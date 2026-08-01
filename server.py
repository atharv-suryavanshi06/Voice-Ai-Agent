"""
server.py

FastAPI server providing endpoints for Twilio Webhook & WebSocket Media Streams.
Runs the Pipecat real-time Voice AI agent pipeline over telephone calls.
"""

import asyncio
import json
from fastapi import FastAPI, WebSocket, Request, Response
from fastapi.responses import HTMLResponse

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMContextFrame, LLMRunFrame
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
from pipecat.services.google.llm import GoogleLLMService
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.transcriptions.language import Language

from conversation.conversation_manager import ConversationManager
from core import config
from core.metrics_tracker import MetricsCollectorProcessor, MetricsTracker
from database.db_manager import PostgresDBManager
from rag.retriever import PolicyRetriever
from rag.vector_store import PolicyVectorStore
from services.email_sender import EmailService
from services.twilio_service import TwilioService
from main import ConversationManagerProcessor, INITIAL_GREETING_TRIGGER
from core.live_events import live_event_hub

app = FastAPI(title="Voice AI Agent Twilio Telephony Server")
twilio_service = TwilioService()
public_base_url: str = ""


def set_public_url(url: str):
    global public_base_url
    public_base_url = url.rstrip("/")


@app.websocket("/events")
async def dashboard_events(websocket: WebSocket):
    """Stream best-effort call activity and completed transcripts to the UI."""
    await websocket.accept()
    queue, history = live_event_hub.subscribe()
    try:
        for event in history:
            await websocket.send_json(event)
        while True:
            event = await queue.get()
            await websocket.send_json(event)
    except asyncio.CancelledError:
        # Uvicorn cancels waiting WebSocket tasks during normal shutdown.
        # In Python 3.11 this is not an Exception, so handle it explicitly to
        # avoid an ASGI traceback when the microphone process receives Ctrl+C.
        return
    except Exception:
        # A closed browser must not affect the running call.
        pass
    finally:
        live_event_hub.unsubscribe(queue)


@app.api_route("/twiml", methods=["GET", "POST"])
async def twiml_endpoint(request: Request):
    """
    TwiML webhook endpoint called by Twilio when outbound/inbound call connects.
    Instructs Twilio to open a WebSocket stream to /ws.
    """
    ws_host = public_base_url.replace("https://", "").replace("http://", "")
    ws_url = f"wss://{ws_host}/ws"
    twiml_xml = twilio_service.generate_twiml(ws_url)
    return Response(content=twiml_xml, media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for Twilio Media Streams.
    Receives raw audio over WebSocket, streams to Deepgram STT -> Gemini LLM -> Cartesia TTS -> Twilio Speaker.
    """
    await websocket.accept()
    db_manager = None
    conversation_processor = None
    session_metrics = MetricsTracker()
    session_id = "twilio-startup"
    try:
        stream_sid = None
        call_sid = None
        while not stream_sid:
            message_text = await websocket.receive_text()
            data = json.loads(message_text)
            if data.get("event") == "start":
                start_data = data.get("start", {})
                stream_sid = start_data.get("streamSid")
                call_sid = start_data.get("callSid")
                print(f"Twilio Media Stream started: stream_sid={stream_sid}, call_sid={call_sid}")

        serializer = TwilioFrameSerializer(
            stream_sid=stream_sid,
            call_sid=call_sid,
            account_sid=config.TWILIO_ACCOUNT_SID,
            auth_token=config.TWILIO_AUTH_TOKEN,
        )
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                vad_analyzer=SileroVADAnalyzer(),
                serializer=serializer,
            ),
        )
        stt = DeepgramSTTService(
            api_key=config.DEEPGRAM_API_KEY,
            settings=DeepgramSTTService.Settings(
                model=config.DEEPGRAM_MODEL,
                language=Language(config.DEEPGRAM_LANGUAGE),
            ),
        )
        llm = GoogleLLMService(
            api_key=config.GOOGLE_API_KEY,
            settings=GoogleLLMService.Settings(
                model=config.GEMINI_MODEL,
                system_instruction=config.GEMINI_SYSTEM_PROMPT,
            ),
        )
        tts = CartesiaTTSService(
            api_key=config.CARTESIA_API_KEY,
            settings=CartesiaTTSService.Settings(
                voice=config.CARTESIA_VOICE_ID,
                model=config.CARTESIA_MODEL,
            ),
        )

        # Startup I/O/model loading is kept off FastAPI's event loop.
        db_manager = await asyncio.to_thread(PostgresDBManager)
        manager = ConversationManager(db_manager=db_manager)
        session_id = manager.session_id
        email_svc = EmailService()
        vector_store = await asyncio.to_thread(PolicyVectorStore)
        retriever = await asyncio.to_thread(
            PolicyRetriever,
            vector_store,
            None,
            15,
            None,
            session_metrics,
        )
        conversation_processor = ConversationManagerProcessor(
            manager,
            llm,
            retriever,
            db_manager=db_manager,
            email_service=email_svc,
            metrics_tracker=session_metrics,
        )

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
            user_params=LLMUserAggregatorParams(user_mute_strategies=[]),
        )
        pipeline = Pipeline([
            transport.input(),
            stt_metrics_processor,
            stt,
            aggregator.user(),
            conversation_processor,
            llm,
            tts,
            tts_metrics_processor,
            transport.output(),
            aggregator.assistant(),
        ])
        task = PipelineTask(pipeline, enable_rtvi=False, idle_timeout_secs=None)
        context.add_message({"role": "user", "content": INITIAL_GREETING_TRIGGER})
        await task.queue_frames([LLMRunFrame()])

        print("Twilio call connected - voice agent listening.")
        runner = PipelineRunner()
        await runner.run(task)
    except Exception as e:
        print(f"Twilio pipeline session ended for session '{session_id}': {e}")
    finally:
        if conversation_processor is not None:
            await conversation_processor.aclose()
        if db_manager is not None:
            db_manager.close()
        session_metrics.print_summary()
        try:
            from observability.langsmith_tracer import global_langsmith_tracer
            await asyncio.to_thread(global_langsmith_tracer.flush, 5.0)
        except Exception:
            pass
