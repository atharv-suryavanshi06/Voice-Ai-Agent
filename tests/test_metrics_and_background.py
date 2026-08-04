import asyncio
import unittest
from types import SimpleNamespace

from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData

from core.background_tasks import BackgroundJob, BoundedBackgroundWorker
from core.metrics_tracker import MetricsTracker
from conversation.conversation_manager import ConversationManager
from main import ConversationManagerProcessor


class MetricsCorrectnessTests(unittest.TestCase):
    def test_conversation_processor_does_not_replace_pipecat_metrics_object(self):
        tracker = MetricsTracker()
        processor = ConversationManagerProcessor(
            ConversationManager(session_id="metrics-contract"),
            SimpleNamespace(),
            SimpleNamespace(),
            metrics_tracker=tracker,
        )

        self.assertIsNot(processor._metrics, tracker)
        self.assertTrue(callable(getattr(processor._metrics, "setup", None)))
        self.assertIs(processor._metrics_tracker, tracker)

    def test_provider_usage_does_not_double_manual_estimate(self):
        tracker = MetricsTracker()
        tracker.record_llm_request("system", "hello")
        tracker.record_llm_response("world")
        usage = LLMUsageMetricsData(
            processor="GoogleLLMService",
            value=LLMTokenUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14),
        )
        tracker.record_metrics_frame(MetricsFrame(data=[usage]))

        self.assertEqual(tracker.llm_prompt_tokens, 10)
        self.assertEqual(tracker.llm_completion_tokens, 4)
        self.assertEqual(tracker.llm_total_tokens, 14)
        self.assertGreater(tracker.llm_estimated_prompt_tokens, 0)

    def test_overlapping_calls_have_isolated_timing_state(self):
        first = MetricsTracker()
        second = MetricsTracker()
        first._llm_start_time = 10.0
        second._llm_start_time = 20.0
        first.llm_calls = 1
        second.llm_calls = 2

        self.assertEqual(first._llm_start_time, 10.0)
        self.assertEqual(second._llm_start_time, 20.0)
        self.assertEqual((first.llm_calls, second.llm_calls), (1, 2))


class BackgroundWorkerTests(unittest.TestCase):
    def test_failure_is_retried_without_crashing_caller(self):
        attempts = []
        failures = []

        def fail():
            attempts.append(1)
            raise RuntimeError("database unavailable")

        async def scenario():
            worker = BoundedBackgroundWorker("test", max_queue_size=2)
            accepted = worker.submit(BackgroundJob(
                operation="persist",
                func=fail,
                max_attempts=2,
                on_failure=lambda error: failures.append(type(error).__name__),
            ))
            self.assertTrue(accepted)
            self.assertTrue(await worker.close(timeout=3.0))

        asyncio.run(scenario())
        self.assertEqual(len(attempts), 2)
        self.assertEqual(failures, ["RuntimeError"])


class ConversationShutdownPersistenceTests(unittest.TestCase):
    def test_shutdown_saves_complete_history_after_flushing_incremental_jobs(self):
        class FakeDatabase:
            enabled = True

            def __init__(self):
                self.incremental_messages = []
                self.full_history = None
                self.ended_session = None

            def persist_conversation_update(self, _session_id, _profile, messages):
                self.incremental_messages.extend(messages)
                return True

            def save_profile(self, _session_id, _profile, history):
                self.full_history = history
                return True

            def end_conversation_session(self, session_id, status):
                self.ended_session = (session_id, status)
                return True

        async def scenario():
            manager = ConversationManager(session_id="shutdown-history")
            manager.process_user_message("hello")
            manager.record_assistant_reply("hi there")
            database = FakeDatabase()
            processor = ConversationManagerProcessor(
                manager,
                SimpleNamespace(),
                SimpleNamespace(),
                db_manager=database,
            )

            await processor.aclose()

            self.assertEqual(
                database.full_history,
                [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi there"}],
            )
            self.assertEqual(database.ended_session, ("shutdown-history", "completed"))

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
