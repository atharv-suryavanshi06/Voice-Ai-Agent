"""In-process, best-effort event feed for the optional live call dashboard.

Publishing an event never awaits network I/O.  The voice pipeline only places a
small dictionary in bounded asyncio queues; WebSocket handlers serialize and
send it independently.  A slow or disconnected browser therefore cannot block
STT, LLM, or TTS processing.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, Deque, Dict, List


class LiveEventHub:
    """Fan out dashboard events to connected WebSocket clients."""

    def __init__(self, history_size: int = 100, queue_size: int = 100) -> None:
        self._history: Deque[Dict[str, Any]] = deque(maxlen=history_size)
        self._subscribers: List[asyncio.Queue] = []
        self._queue_size = queue_size
        self._current_activity: str | None = None

    def publish(self, event_type: str, **payload: Any) -> None:
        """Publish without waiting for a connected browser or network send."""
        event = {
            "type": event_type,
            "timestamp": time.time(),
            **payload,
        }
        self._history.append(event)

        for queue in tuple(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Keep the most recent view useful without delaying the call.
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

    def set_activity(self, label: str, detail: str = "") -> None:
        """Record an activity transition, ignoring repeated frame-level states."""
        activity_key = f"{label}|{detail}"
        if activity_key == self._current_activity:
            return
        self._current_activity = activity_key
        self.publish("activity", label=label, detail=detail)

    def publish_message(self, role: str, text: str) -> None:
        """Publish a completed user or assistant transcript message."""
        clean_text = (text or "").strip()
        if clean_text:
            self.publish("message", role=role, text=clean_text)

    def subscribe(self) -> tuple[asyncio.Queue, List[Dict[str, Any]]]:
        """Register a dashboard connection and return its bounded queue/history."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.append(queue)
        return queue, list(self._history)

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Remove a dashboard connection if it is still registered."""
        try:
            self._subscribers.remove(queue)
        except ValueError:
            pass


live_event_hub = LiveEventHub()
