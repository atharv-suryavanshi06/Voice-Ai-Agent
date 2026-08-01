"""Bounded background execution for non-response-critical synchronous work."""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple


logger = logging.getLogger(__name__)


@dataclass
class BackgroundJob:
    operation: str
    func: Callable[..., Any]
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    max_attempts: int = 3
    on_success: Optional[Callable[[Any], Any]] = None
    on_failure: Optional[Callable[[BaseException], Any]] = None


class BoundedBackgroundWorker:
    """Runs synchronous side effects in worker threads behind a bounded queue."""

    def __init__(self, name: str, max_queue_size: int = 64) -> None:
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        self.name = name
        self._queue: asyncio.Queue[Optional[BackgroundJob]] = asyncio.Queue(maxsize=max_queue_size)
        self._task: Optional[asyncio.Task] = None
        self._closed = False

    def start(self) -> None:
        if self._closed:
            raise RuntimeError(f"Background worker '{self.name}' is closed")
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name=f"background:{self.name}")

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    def submit(self, job: BackgroundJob) -> bool:
        """Enqueue without waiting; return False and log when the queue is full."""
        if self._closed:
            logger.error(
                "Background operation rejected because worker is closed",
                extra={"worker": self.name, "operation": job.operation, **job.context},
            )
            return False
        self.start()
        try:
            self._queue.put_nowait(job)
            return True
        except asyncio.QueueFull:
            logger.error(
                "Background operation queue is full",
                extra={"worker": self.name, "operation": job.operation, **job.context},
            )
            return False

    async def _notify(self, callback: Optional[Callable[..., Any]], value: Any) -> None:
        if callback is None:
            return
        try:
            result = callback(value)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("Background completion callback failed", extra={"worker": self.name})

    async def _run(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                if job is None:
                    return

                last_error: Optional[BaseException] = None
                result: Any = None
                for attempt in range(1, max(1, job.max_attempts) + 1):
                    try:
                        result = await asyncio.to_thread(job.func, *job.args, **job.kwargs)
                        if result is False:
                            raise RuntimeError(f"Operation '{job.operation}' returned an unsuccessful result")
                        last_error = None
                        break
                    except BaseException as exc:
                        last_error = exc
                        logger.warning(
                            "Background operation failed",
                            extra={
                                "worker": self.name,
                                "operation": job.operation,
                                "attempt": attempt,
                                "max_attempts": job.max_attempts,
                                "error_type": type(exc).__name__,
                                **job.context,
                            },
                        )
                        if attempt < job.max_attempts:
                            await asyncio.sleep(min(0.25 * attempt, 1.0))

                if last_error is None:
                    await self._notify(job.on_success, result)
                else:
                    logger.error(
                        "Background operation exhausted retries",
                        extra={"worker": self.name, "operation": job.operation, **job.context},
                    )
                    await self._notify(job.on_failure, last_error)
            finally:
                self._queue.task_done()

    async def flush(self, timeout: float = 10.0) -> bool:
        """Wait for queued work. False means the timeout elapsed."""
        if self._task is None:
            return True
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.error(
                "Timed out while flushing background operations",
                extra={"worker": self.name, "pending": self.pending_count},
            )
            return False

    async def close(self, timeout: float = 10.0) -> bool:
        """Flush pending work and stop the worker. Safe to call repeatedly."""
        if self._closed:
            return True
        flushed = await self.flush(timeout=timeout)
        self._closed = True
        if self._task is not None and not self._task.done():
            try:
                self._queue.put_nowait(None)
                await asyncio.wait_for(self._task, timeout=1.0)
            except (asyncio.QueueFull, asyncio.TimeoutError):
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)
        return flushed
