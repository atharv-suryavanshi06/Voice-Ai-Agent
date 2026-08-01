"""
main.py

A minimal, local, real-time voice agent built with Pipecat.

Pipeline:
    Microphone -> Deepgram (STT) -> Gemini (LLM) -> Cartesia (TTS) -> Speaker

Setup:
    pip install -r requirements.txt
    cp .env.example .env   # then fill in your API keys

Run:
    python main.py

Speak into your microphone; the bot's reply plays back through your
speakers. Press Ctrl+C to stop. Headphones are recommended - see the note
on AlwaysUserMuteStrategy below.
"""

import asyncio
import logging
import time
import math


from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, LLMContextFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from conversation.conversation_manager import ConversationManager
from conversation.customer_profile import normalize_stt_aliases
from core import config
from database.db_manager import PostgresDBManager
from services.email_sender import EmailService
from rag.vector_store import PolicyVectorStore
from rag.retriever import PolicyRetriever
from core.metrics_tracker import MetricsCollectorProcessor, MetricsTracker, global_metrics_tracker
from core.background_tasks import BackgroundJob, BoundedBackgroundWorker
from core.live_events import live_event_hub
from rag.service import RAGService


logger = logging.getLogger(__name__)

# Lifecycle state is initialized for every mode so partial startup failures are
# never masked by cleanup-time NameError exceptions.
active_session_manager = None
active_db_manager = None
active_background_worker = None
active_metrics_tracker = None
active_dashboard_server = None
active_dashboard_task = None


async def shutdown_active_resources() -> None:
    """Idempotent cleanup for normal completion and partial startup failures."""
    global active_session_manager, active_db_manager, active_background_worker
    global active_metrics_tracker, active_dashboard_server, active_dashboard_task

    if active_background_worker is not None:
        try:
            await active_background_worker.close(timeout=10.0)
        except Exception:
            logger.exception("Failed to close active background worker")
    if active_db_manager is not None:
        try:
            active_db_manager.close()
        except Exception:
            logger.exception("Failed to close active database manager")
    if active_dashboard_server is not None:
        active_dashboard_server.should_exit = True
    if active_dashboard_task is not None and not active_dashboard_task.done():
        try:
            await asyncio.wait_for(active_dashboard_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            active_dashboard_task.cancel()
            await asyncio.gather(active_dashboard_task, return_exceptions=True)

    active_session_manager = None
    active_db_manager = None
    active_background_worker = None
    active_metrics_tracker = None
    active_dashboard_server = None
    active_dashboard_task = None


# Internal prompt used only to start Pipecat's first LLM turn. It is not a
# caller utterance and must never be rendered as one in the live dashboard.
INITIAL_GREETING_TRIGGER = "Greet me briefly and ask how you can help."


def _get_message_role(msg) -> str:
    if isinstance(msg, dict):
        return msg.get("role", "")
    if hasattr(msg, "message"):
        inner = msg.message
        if isinstance(inner, dict):
            return inner.get("role", "")
        if hasattr(inner, "role"):
            return inner.role
    if hasattr(msg, "role"):
        return msg.role
    return ""


def _get_message_content(msg) -> str:
    if isinstance(msg, dict):
        return msg.get("content", "")
    if hasattr(msg, "message"):
        inner = msg.message
        if isinstance(inner, dict):
            return inner.get("content", "")
        if hasattr(inner, "content"):
            return inner.content
    if hasattr(msg, "content"):
        return msg.content
    return ""


class ConversationManagerProcessor(FrameProcessor):
    """
    Custom FrameProcessor that intercepts LLMContextFrames in the pipeline,
    extracts information from user messages using ConversationManager, and
    dynamically adjusts the system instructions for GoogleLLMService based on
    the current conversation state and customer profile.
    """

    def __init__(
        self,
        manager: ConversationManager,
        llm: GoogleLLMService,
        retriever: PolicyRetriever,
        db_manager: PostgresDBManager = None,
        email_service: EmailService = None,
        metrics_tracker: MetricsTracker = None,
        background_worker: BoundedBackgroundWorker = None,
        rag_service: RAGService = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._manager = manager
        self._llm = llm
        self._retriever = retriever
        self._db_manager = db_manager
        self._email_service = email_service
        # FrameProcessor reserves ``self._metrics`` for Pipecat's own
        # FrameProcessorMetrics object, which implements setup()/cleanup().
        self._metrics_tracker = metrics_tracker or MetricsTracker()
        self._background_worker = background_worker or BoundedBackgroundWorker(
            name=f"session-{manager.session_id}", max_queue_size=64
        )
        self._owns_background_worker = background_worker is None
        self._rag_service = rag_service or RAGService(retriever, top_k=5)
        self._recorded_assistant_count = 0
        self._is_first_turn = True

    def _queue_persistence(self) -> bool:
        if not self._db_manager or not getattr(self._db_manager, "enabled", False):
            return False
        messages = self._manager.pending_persistence_messages()
        if not messages:
            return True
        message_ids = [message["message_id"] for message in messages]
        job = BackgroundJob(
            operation="persist_conversation",
            func=self._db_manager.persist_conversation_update,
            args=(self._manager.session_id, self._manager.profile.to_dict(), messages),
            context={"session_id": self._manager.session_id, "message_count": len(messages)},
            max_attempts=3,
            on_success=lambda _result: self._manager.mark_persistence_complete(message_ids, True),
            on_failure=lambda _error: self._manager.mark_persistence_complete(message_ids, False),
        )
        accepted = self._background_worker.submit(job)
        if accepted:
            self._manager.mark_persistence_queued(message_ids)
        return accepted

    async def aclose(self) -> None:
        """Flush session persistence/tracing work. Safe after partial startup."""
        self._queue_persistence()
        await self._background_worker.flush(timeout=10.0)
        # A queue-full or exhausted-retry callback leaves messages unreserved.
        # Retry them once during shutdown so critical persistence is not silently
        # discarded when the call has no later turn to trigger another enqueue.
        if self._manager.pending_persistence_messages():
            self._queue_persistence()
            await self._background_worker.flush(timeout=10.0)
        if self._db_manager and getattr(self._db_manager, "enabled", False):
            self._background_worker.submit(BackgroundJob(
                operation="end_conversation_session",
                func=self._db_manager.end_conversation_session,
                args=(self._manager.session_id, "completed"),
                context={"session_id": self._manager.session_id},
            ))
            await self._background_worker.flush(timeout=5.0)
        if self._owns_background_worker:
            await self._background_worker.close(timeout=10.0)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            context = frame.context

            assistant_msgs = [m for m in context.messages if _get_message_role(m) == "assistant"]
            current_assistant_count = len(assistant_msgs)
            if current_assistant_count > self._recorded_assistant_count:

                new_assistant_msgs = assistant_msgs[self._recorded_assistant_count:]
                for msg in new_assistant_msgs:
                    reply_content = _get_message_content(msg)
                    self._manager.record_assistant_reply(reply_content)
                    self._queue_persistence()
                    live_event_hub.publish_message("assistant", reply_content)
                    live_event_hub.set_activity("Converting text to speech", "Cartesia is preparing the reply")

                    # Asynchronously log completed turn trace to LangSmith
                    try:
                        from observability.langsmith_tracer import global_langsmith_tracer
                        last_user_msg = getattr(self, "_last_user_msg", "")
                        system_prompt = getattr(self, "_last_system_prompt", "")
                        retrieved_chunks = getattr(self, "_last_retrieved_chunks", None)
                        turn_start_t = getattr(self, "_turn_start_t", time.perf_counter())
                        turn_latency_ms = (time.perf_counter() - turn_start_t) * 1000.0

                        p_tokens = max(1, math.ceil(len(system_prompt + last_user_msg) / 4.0)) if system_prompt else 0
                        c_tokens = max(1, math.ceil(len(reply_content) / 4.0)) if reply_content else 0

                        global_langsmith_tracer.log_voice_turn(
                            session_id=self._manager.session_id,
                            user_message=last_user_msg,
                            system_prompt=system_prompt,
                            llm_response=reply_content,
                            prompt_tokens=p_tokens,
                            completion_tokens=c_tokens,
                            end_to_end_latency_ms=turn_latency_ms,
                            retrieved_chunks=retrieved_chunks,
                        )
                    except Exception:
                        pass

                self._recorded_assistant_count = current_assistant_count

            # Retrieve user messages from context
            user_msgs = [m for m in context.messages if _get_message_role(m) == "user"]
            if user_msgs:
                last_user_msg = normalize_stt_aliases(_get_message_content(user_msgs[-1]))
                self._last_user_msg = last_user_msg
                self._turn_start_t = time.perf_counter()
                is_initial_greeting = last_user_msg.strip() == INITIAL_GREETING_TRIGGER
                if not is_initial_greeting:
                    live_event_hub.publish_message("user", last_user_msg)
                    live_event_hub.set_activity("Understanding request", "Checking the conversation flow")
                else:
                    live_event_hub.set_activity("Starting conversation", "Riya is preparing a greeting")

                # Check if it's the initial greeting turn
                if self._is_first_turn:
                    system_prompt = self._manager.start_conversation()
                    retrieved_chunks = None
                    self._is_first_turn = False
                else:
                    is_policy_q = self._manager.should_retrieve_policy_context(last_user_msg)
                    retrieved_chunks = None
                    retrieval_error = False
                    if is_policy_q:
                        try:
                            policy_id_filter = self._manager.prepare_policy_context(last_user_msg)
                            if self._manager.policy_selection_required:
                                live_event_hub.set_activity("Clarifying policy", "Waiting for the policy name or number")
                            else:
                                live_event_hub.set_activity("Searching policy knowledge", "Retrieving relevant policy details")
                                retrieved_chunks = await asyncio.to_thread(
                                    self._rag_service.retrieve_relevant,
                                    last_user_msg,
                                    policy_id_filter,
                                    5,
                                )
                        except Exception:
                            retrieval_error = True
                            logger.exception(
                                "Policy retrieval failed",
                                extra={
                                    "session_id": self._manager.session_id,
                                    "resolution_status": self._manager.policy_resolution_status,
                                    "selected_policy_id": self._manager.active_policy_id,
                                },
                            )
                            retrieved_chunks = None
                    system_prompt = self._manager.process_user_message(
                        last_user_msg,
                        retrieved_chunks=retrieved_chunks,
                        is_policy_question=is_policy_q,
                        retrieval_error=retrieval_error,
                    )
                    self._manager.complete_policy_turn(is_policy_q)

                    # Email is the only side effect whose result may affect this
                    # turn's spoken text. It runs in a worker thread and is only
                    # awaited once a confirmed address actually needs delivery.
                    confirmed_email = self._manager.profile.email
                    should_attempt_email = bool(
                        confirmed_email
                        and self._manager.profile.email_confirmed
                        and (self._manager.last_sent_email or "") != confirmed_email.strip().lower()
                    )
                    if should_attempt_email:
                        await self._manager.maybe_trigger_email_async(self._email_service, self._db_manager)
                        system_prompt = self._manager.build_system_prompt(
                            retrieved_chunks=retrieved_chunks,
                            retrieval_error=retrieval_error,
                        )

                self._last_system_prompt = system_prompt
                self._last_retrieved_chunks = retrieved_chunks

                # Persistence is append-only and queued; the response never waits
                # for PostgreSQL.
                self._queue_persistence()
                live_event_hub.set_activity("LLM is responding", "Gemini is generating a reply")

                # Dynamically set system instruction on the Gemini LLM service settings
                self._llm._settings.system_instruction = system_prompt




            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


import argparse
import uvicorn

async def run_microphone_agent(dashboard: bool = True) -> None:
    global active_session_manager, active_db_manager, active_background_worker
    global active_metrics_tracker, active_dashboard_server, active_dashboard_task
    config.validate_api_keys()

    dashboard_server = None
    dashboard_task = None
    if dashboard:
        # This server is optional and shares the event loop. Dashboard events
        # are still queued with put_nowait, so browser I/O never blocks audio.
        from server import app
        import uvicorn

        dashboard_server = uvicorn.Server(
            uvicorn.Config(app=app, host="0.0.0.0", port=config.SERVER_PORT, log_level="warning")
        )
        dashboard_task = asyncio.create_task(dashboard_server.serve())
        active_dashboard_server = dashboard_server
        active_dashboard_task = dashboard_task
        # Give Uvicorn one scheduling turn to surface an immediate bind/import
        # failure. A dashboard failure must never prevent the audio agent from
        # starting, because the dashboard is only an observer.
        await asyncio.sleep(0.05)
        if dashboard_task.done():
            try:
                dashboard_task.result()
            except BaseException as exc:
                print(f"[Dashboard Warning] Could not start event server: {exc}")
                dashboard_server = None
                dashboard_task = None
                active_dashboard_server = None
                active_dashboard_task = None
        else:
            print(f"[Dashboard] Event server available at ws://localhost:{config.SERVER_PORT}/events")

    # --- Transport: local microphone in, local speaker out -----------------
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        )
    )

    # --- Speech-to-Text: Deepgram (streaming) -------------------------------
    stt = DeepgramSTTService(
        api_key=config.DEEPGRAM_API_KEY,
        settings=DeepgramSTTService.Settings(
            model=config.DEEPGRAM_MODEL,
            language=Language(config.DEEPGRAM_LANGUAGE),
        ),
    )

    # --- LLM: Google Gemini --------------------------------------------------
    llm = GoogleLLMService(
        api_key=config.GOOGLE_API_KEY,
        settings=GoogleLLMService.Settings(
            model=config.GEMINI_MODEL,
            system_instruction=config.GEMINI_SYSTEM_PROMPT,
        ),
    )

    db_manager = PostgresDBManager()
    active_db_manager = db_manager
    manager = ConversationManager(db_manager=db_manager)
    active_session_manager = manager
    email_service = EmailService()
    session_metrics = MetricsTracker()
    vector_store = PolicyVectorStore()
    retriever = PolicyRetriever(vector_store, metrics_tracker=session_metrics)
    conversation_processor = ConversationManagerProcessor(
        manager,
        llm,
        retriever,
        db_manager=db_manager,
        email_service=email_service,
        metrics_tracker=session_metrics,
    )

    active_background_worker = conversation_processor._background_worker
    active_metrics_tracker = session_metrics


    # --- Text-to-Speech: Cartesia (streaming) -------------------------------
    tts = CartesiaTTSService(
        api_key=config.CARTESIA_API_KEY,
        settings=CartesiaTTSService.Settings(
            voice=config.CARTESIA_VOICE_ID,
            model=config.CARTESIA_MODEL,
        ),
    )

    # --- Metrics Collector Processors -----------------------------------------
    stt._enable_metrics = True
    stt._enable_usage_metrics = True
    llm._enable_metrics = True
    llm._enable_usage_metrics = True
    tts._enable_metrics = True
    tts._enable_usage_metrics = True

    stt_metrics_processor = MetricsCollectorProcessor(
        tracker=session_metrics,
        capture_input_events=True,
        capture_service_events=False,
        capture_native_metrics=False,
    )
    tts_metrics_processor = MetricsCollectorProcessor(
        tracker=session_metrics,
        capture_input_events=False,
        capture_service_events=True,
        capture_native_metrics=True,
    )

    context = LLMContext()
    aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_mute_strategies=[],
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),        # Microphone
            stt_metrics_processor,    # BEFORE STT: Captures microphone audio & VAD speech stops
            stt,                      # Deepgram STT
            aggregator.user(),        # Add user turn to context
            conversation_processor,   # Manage states, extract profiles, dynamic system instruction
            llm,                      # Gemini LLM
            tts,                      # Cartesia TTS
            tts_metrics_processor,    # AFTER TTS: Captures transcriptions, LLM frames & TTS audio frames
            transport.output(),       # Speaker
            aggregator.assistant(),   # Add assistant turn to context
        ]
    )

    task = PipelineTask(pipeline, enable_rtvi=False, idle_timeout_secs=None)

    context.add_message(
        {"role": "user", "content": INITIAL_GREETING_TRIGGER}
    )
    await task.queue_frames([LLMRunFrame()])

    print("\n[Microphone Mode] Voice agent running - speak into your microphone. Press Ctrl+C to stop.\n")
    runner = PipelineRunner()
    interrupted = False
    try:
        await runner.run(task)
    except asyncio.CancelledError:
        # Ctrl+C cancels the microphone coroutine. Consume that cancellation
        # here so the optional Uvicorn dashboard can receive its normal
        # should_exit signal instead of being force-cancelled mid-lifespan.
        interrupted = True
    finally:
        await conversation_processor.aclose()
        db_manager.close()
        session_metrics.print_summary()
        try:
            from observability.langsmith_tracer import global_langsmith_tracer
            await asyncio.to_thread(global_langsmith_tracer.flush, 5.0)
        except Exception:
            logger.exception("LangSmith flush failed", extra={"session_id": manager.session_id})
        active_session_manager = None
        active_db_manager = None
        active_background_worker = None
        active_metrics_tracker = None
        if dashboard_server:
            dashboard_server.should_exit = True
        if dashboard_task:
            try:
                await asyncio.wait_for(asyncio.shield(dashboard_task), timeout=5)
            except asyncio.TimeoutError:
                dashboard_task.cancel()
                await asyncio.gather(dashboard_task, return_exceptions=True)
            except asyncio.CancelledError:
                # A second Ctrl+C is an explicit forced shutdown request.
                dashboard_task.cancel()
                await asyncio.gather(dashboard_task, return_exceptions=True)
            except Exception as exc:
                print(f"[Dashboard Warning] Event server stopped unexpectedly: {exc}")
        active_dashboard_server = None
        active_dashboard_task = None



async def run_twilio_agent(phone_number: str = None) -> None:
    config.validate_api_keys()
    config.validate_twilio_keys()

    from services.twilio_service import TwilioService, setup_ngrok_tunnel
    from server import app, set_public_url

    if not phone_number:
        phone_number = input("Enter target phone number to call (e.g., +919876543210): ").strip()
    
    if not phone_number:
        print("Phone number is required for Twilio mode.")
        return

    # Auto-prefix +91 for 10-digit numbers missing country code
    if not phone_number.startswith("+"):
        if len(phone_number) == 10 and phone_number.isdigit():
            phone_number = f"+91{phone_number}"
        else:
            phone_number = f"+{phone_number}"
    
    print(f"Target recipient phone number: {phone_number}")


    # Start ngrok tunnel or use config.PUBLIC_URL
    public_url = setup_ngrok_tunnel(config.SERVER_PORT)
    set_public_url(public_url)
    twiml_url = f"{public_url}/twiml"

    print(f"\n[Twilio Mode] Public Webhook URL: {twiml_url}")
    
    server = None
    server_task = None
    try:
        # Initialize Uvicorn Server async task
        uvicorn_config = uvicorn.Config(app=app, host="0.0.0.0", port=config.SERVER_PORT, log_level="info")
        server = uvicorn.Server(uvicorn_config)
        server_task = asyncio.create_task(server.serve())

        # Wait briefly for server startup then trigger Twilio call
        await asyncio.sleep(2)
        if server_task.done():
            server_task.result()
        twilio_svc = TwilioService()
        call_sid = twilio_svc.make_outbound_call(to_phone_number=phone_number, twiml_url=twiml_url)
        print(f"\nOutbound call initiated! Call SID: {call_sid}")
        print("Waiting for recipient to answer... Press Ctrl+C to stop.\n")
        await server_task
    finally:
        if server is not None:
            server.should_exit = True
        if server_task is not None and not server_task.done():
            try:
                await asyncio.wait_for(server_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                server_task.cancel()
                await asyncio.gather(server_task, return_exceptions=True)
        try:
            from observability.langsmith_tracer import global_langsmith_tracer
            await asyncio.to_thread(global_langsmith_tracer.flush, 5.0)
        except Exception:
            logger.exception("LangSmith flush failed during Twilio shutdown")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Voice AI Agent Runner")
    parser.add_argument("--mode", choices=["mic", "twilio"], help="Mode: 'mic' for local microphone, 'twilio' for phone call")
    parser.add_argument("--phone", help="Recipient phone number for Twilio mode (e.g. +919876543210)")
    parser.add_argument("--dashboard", dest="dashboard", action="store_true", help="Start the live dashboard event server (enabled by default in microphone mode)")
    parser.add_argument("--no-dashboard", dest="dashboard", action="store_false", help="Run microphone mode without the live dashboard event server")
    parser.set_defaults(dashboard=True)
    args = parser.parse_args()

    mode = args.mode
    if not mode:
        print("========================================")
        print("    Voice AI Agent - Mode Selection    ")
        print("========================================")
        print("1. Local Microphone / Speaker")
        print("2. Twilio Outbound Phone Call")
        choice = input("Select mode (1 or 2, default 1): ").strip()
        if choice == "2":
            mode = "twilio"
        else:
            mode = "mic"

    try:
        if mode == "twilio":
            await run_twilio_agent(phone_number=args.phone)
        else:
            await run_microphone_agent(dashboard=args.dashboard)
    finally:
        await shutdown_active_resources()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
