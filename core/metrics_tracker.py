"""
metrics_tracker.py

Accurately tracks and reports latency (TTFB & total response time) and usage/token metrics per API Key:
  1. Google Gemini LLM (gemini-3.1-flash-lite / gemini-2.0-flash)
  2. Google Gemini Embedding Model (gemini-embedding-2)
  3. Deepgram STT (nova-3 / nova-2)
  4. Cartesia TTS (sonic-3 / sonic-english)

Prints a detailed latency and token usage table when main.py is stopped.
"""

import atexit
import time
import math
import weakref
from typing import List, Optional
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame, MetricsFrame, TextFrame, UserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame,
    UserStartedSpeakingFrame, VADUserStartedSpeakingFrame, TranscriptionFrame,
    LLMFullResponseStartFrame, LLMFullResponseEndFrame, TTSStartedFrame, TTSStoppedFrame,
    LLMTextFrame, TTSTextFrame, BotStartedSpeakingFrame, TTSAudioRawFrame,
    InputAudioRawFrame, UserAudioRawFrame, AudioRawFrame
)
from pipecat.metrics.metrics import (
    TTFBMetricsData, LLMUsageMetricsData, TTSUsageMetricsData, LLMTokenUsage
)
from core import config
from core.live_events import live_event_hub
_ACTIVE_TRACKERS = weakref.WeakSet()


def _flush_unprinted_trackers() -> None:
    """Print active metrics before the interpreter exits."""
    for tracker in list(_ACTIVE_TRACKERS):
        if not tracker._summary_printed and any((
            tracker.llm_calls, tracker.embedding_calls, tracker.stt_calls,
            tracker.tts_calls, tracker.llm_total_tokens,
        )):
            tracker.print_summary()


atexit.register(_flush_unprinted_trackers)


def mask_api_key(key: Optional[str]) -> str:
    """Safely masks API key showing prefix and suffix."""
    if not key:
        return "NOT_CONFIGURED"
    clean_key = str(key).strip()
    if len(clean_key) <= 8:
        return "****"
    return f"{clean_key[:4]}...{clean_key[-4:]}"


class MetricsTracker:
    """
    Central collector for tracking API request counts, TTFB/generation latencies, 
    and token/character usage across all configured API keys.
    """

    def __init__(self):
        self._summary_printed = False
        _ACTIVE_TRACKERS.add(self)

        # 1. Google Gemini LLM Metrics
        self.llm_calls: int = 0
        self.llm_ttfbs: List[float] = []            # Latency to first token (ms)
        self.llm_total_latencies: List[float] = []  # Total response generation latency (ms)
        self.llm_prompt_tokens: int = 0
        self.llm_completion_tokens: int = 0
        self.llm_total_tokens: int = 0
        self.llm_estimated_prompt_tokens: int = 0
        self.llm_estimated_completion_tokens: int = 0

        # 2. Google Gemini Embedding Metrics
        self.embedding_calls: int = 0
        self.embedding_latencies: List[float] = []  # Embedding call latency (ms)
        self.embedding_query_chars: int = 0
        self.embedding_tokens: int = 0

        # 3. Deepgram STT Metrics
        self.stt_calls: int = 0
        self.stt_latencies: List[float] = []        # STT response latency (ms)

        # 4. Cartesia TTS Metrics
        self.tts_calls: int = 0
        self.tts_ttfbs: List[float] = []            # TTFB audio synthesis latency (ms)
        self.tts_chars: int = 0
        self.tts_estimated_chars: int = 0

        # Internal turn timing state
        self._user_stopped_time: Optional[float] = None
        self._llm_start_time: Optional[float] = None
        self._llm_first_token_time: Optional[float] = None
        self._tts_start_time: Optional[float] = None
        self._llm_active: bool = False
        self._tts_active: bool = False

    def record_stt_transcription(self, text: str, latency_ms: Optional[float] = None):
        """Records an STT transcription event."""
        if not text or not text.strip():
            return
        self.stt_calls += 1

        if latency_ms and latency_ms > 0:
            self.stt_latencies.append(latency_ms)
        else:
            now = time.perf_counter()
            calc_latency = None
            if getattr(self, "_user_stopped_time", None):
                calc_latency = (now - self._user_stopped_time) * 1000.0
            elif getattr(self, "_stt_speech_stop_time", None):
                calc_latency = (now - self._stt_speech_stop_time) * 1000.0
            elif getattr(self, "_stt_speech_start_time", None):
                calc_latency = (now - self._stt_speech_start_time) * 1000.0

            if calc_latency and calc_latency > 0:
                self.stt_latencies.append(calc_latency)

    def record_llm_request(self, system_instruction: str, user_message: str):
        """Records a manual LLM request using a local, non-network estimate."""
        self._llm_start_time = time.perf_counter()
        self._llm_first_token_time = None
        self.llm_calls += 1

        prompt_text = f"{system_instruction}\nUser: {user_message}"
        tokens = max(1, math.ceil(len(prompt_text) / 4.0)) if prompt_text.strip() else 0
        self.llm_estimated_prompt_tokens += tokens

    def record_llm_response(self, reply_text: str, ttfb_ms: Optional[float] = None, total_latency_ms: Optional[float] = None):
        """Records an LLM completion response and computes output tokens."""
        if ttfb_ms and ttfb_ms > 0:
            self.llm_ttfbs.append(ttfb_ms)
        if total_latency_ms and total_latency_ms > 0:
            self.llm_total_latencies.append(total_latency_ms)

        tokens = max(1, math.ceil(len(reply_text) / 4.0)) if reply_text and reply_text.strip() else 0
        self.llm_estimated_completion_tokens += tokens

    def record_embedding_call(self, query: str, duration_sec: float):
        """Records an embedding request call to Google Gemini Embeddings."""
        if not query or not query.strip():
            return
        self.embedding_calls += 1
        latency_ms = duration_sec * 1000.0
        self.embedding_latencies.append(latency_ms)
        char_len = len(query)
        self.embedding_query_chars += char_len

        self.embedding_tokens += max(1, math.ceil(char_len / 4.0))

    def record_tts_call(self, text: str, ttfb_ms: Optional[float] = None):
        """Records a Cartesia TTS synthesis call."""
        if not text or not text.strip():
            return
        self.tts_calls += 1
        self.tts_estimated_chars += len(text)
        if ttfb_ms and ttfb_ms > 0:
            self.tts_ttfbs.append(ttfb_ms)

    def record_metrics_frame(self, frame: MetricsFrame):
        """Processes Pipecat's native MetricsFrame events if emitted."""
        for data in frame.data:
            processor_name = str(getattr(data, "processor", "")).lower()

            if isinstance(data, TTFBMetricsData):
                val_ms = data.value * 1000.0
                if ("google" in processor_name or "gemini" in processor_name or "llm" in processor_name) and val_ms > 0:
                    self.llm_ttfbs.append(val_ms)
                elif ("deepgram" in processor_name or "stt" in processor_name) and val_ms > 0:
                    self.stt_latencies.append(val_ms)
                elif ("cartesia" in processor_name or "tts" in processor_name) and val_ms > 0:
                    self.tts_ttfbs.append(val_ms)

            elif isinstance(data, LLMUsageMetricsData):
                usage: LLMTokenUsage = data.value
                if usage.prompt_tokens:
                    self.llm_prompt_tokens += usage.prompt_tokens
                if usage.completion_tokens:
                    self.llm_completion_tokens += usage.completion_tokens
                if usage.total_tokens:
                    self.llm_total_tokens += usage.total_tokens
                else:
                    self.llm_total_tokens += (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)

            elif isinstance(data, TTSUsageMetricsData):
                if isinstance(data.value, (int, float)):
                    self.tts_chars += int(data.value)

    def print_summary(self):
        """Prints a comprehensive latency & token usage report per API key to stdout."""
        if self._summary_printed:
            return
        self._summary_printed = True
        def avg(lst: List[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        def mn(lst: List[float]) -> float:
            return min(lst) if lst else 0.0

        def mx(lst: List[float]) -> float:
            return max(lst) if lst else 0.0

        print("\n" + "=" * 74)
        print("           VOICE AI AGENT - LATENCY & TOKEN USAGE REPORT           ")
        print("=" * 74)

        # 1. Google Gemini LLM
        google_key = mask_api_key(getattr(config, "GOOGLE_API_KEY", None))
        llm_model = getattr(config, "GEMINI_MODEL", "gemini-2.0-flash")
        print(f"\n1. GOOGLE GEMINI LLM")
        print(f"   - Service / Model   : GoogleLLMService ({llm_model})")
        print(f"   - API Key           : {google_key}")
        print(f"   - Total Responses   : {self.llm_calls}")
        if self.llm_ttfbs:
            print(f"   - TTFB Latency      : Avg = {avg(self.llm_ttfbs):.2f} ms | Min = {mn(self.llm_ttfbs):.2f} ms | Max = {mx(self.llm_ttfbs):.2f} ms")
        else:
            print(f"   - TTFB Latency      : N/A")
        if self.llm_total_latencies:
            print(f"   - Total Latency     : Avg = {avg(self.llm_total_latencies):.2f} ms | Min = {mn(self.llm_total_latencies):.2f} ms | Max = {mx(self.llm_total_latencies):.2f} ms")
        else:
            print(f"   - Total Latency     : N/A")
        if self.llm_total_tokens:
            print(f"   - Token Usage       : Prompt = {self.llm_prompt_tokens} tokens | Completion = {self.llm_completion_tokens} tokens | Total = {self.llm_total_tokens} tokens (provider reported)")
        else:
            estimated_total = self.llm_estimated_prompt_tokens + self.llm_estimated_completion_tokens
            print(f"   - Token Usage       : Prompt = {self.llm_estimated_prompt_tokens} tokens | Completion = {self.llm_estimated_completion_tokens} tokens | Total = {estimated_total} tokens (local estimate)")

        # 2. Google Gemini Embedding
        emb_model = "gemini-embedding-2"
        print(f"\n2. GOOGLE GEMINI EMBEDDING")
        print(f"   - Service / Model   : Google GenAI Embeddings ({emb_model})")
        print(f"   - API Key           : {google_key}")
        print(f"   - Total API Calls   : {self.embedding_calls}")
        if self.embedding_latencies:
            print(f"   - Latency           : Avg = {avg(self.embedding_latencies):.2f} ms | Min = {mn(self.embedding_latencies):.2f} ms | Max = {mx(self.embedding_latencies):.2f} ms")
        else:
            print(f"   - Latency           : N/A (no embedding requests)")
        print(f"   - Usage             : Embedded Chars = {self.embedding_query_chars} | Est. Tokens = {self.embedding_tokens}")

        # 3. Deepgram STT
        deepgram_key = mask_api_key(getattr(config, "DEEPGRAM_API_KEY", None))
        stt_model = getattr(config, "DEEPGRAM_MODEL", "nova-2")
        print(f"\n3. DEEPGRAM STT")
        print(f"   - Service / Model   : DeepgramSTTService ({stt_model})")
        print(f"   - API Key           : {deepgram_key}")
        print(f"   - Transcriptions    : {self.stt_calls}")
        if self.stt_latencies:
            print(f"   - STT Latency       : Avg = {avg(self.stt_latencies):.2f} ms | Min = {mn(self.stt_latencies):.2f} ms | Max = {mx(self.stt_latencies):.2f} ms")
        else:
            print(f"   - STT Latency       : N/A")

        # 4. Cartesia TTS
        cartesia_key = mask_api_key(getattr(config, "CARTESIA_API_KEY", None))
        tts_model = getattr(config, "CARTESIA_MODEL", "sonic-english")
        tts_voice = str(getattr(config, "CARTESIA_VOICE_ID", "default"))
        print(f"\n4. CARTESIA TTS")
        print(f"   - Service / Voice   : CartesiaTTSService ({tts_model} - {tts_voice[:8]}...)")
        print(f"   - API Key           : {cartesia_key}")
        print(f"   - Synthesizations   : {self.tts_calls}")
        if self.tts_ttfbs:
            print(f"   - TTS TTFB Latency  : Avg = {avg(self.tts_ttfbs):.2f} ms | Min = {mn(self.tts_ttfbs):.2f} ms | Max = {mx(self.tts_ttfbs):.2f} ms")
        else:
            print(f"   - TTS TTFB Latency  : N/A")
        chars = self.tts_chars or self.tts_estimated_chars
        source = "provider reported" if self.tts_chars else "local estimate"
        print(f"   - Usage             : Total Synthesized Chars = {chars} ({source})")

        print("\n" + "=" * 74 + "\n")


# Global singleton instance
global_metrics_tracker = MetricsTracker()


class MetricsCollectorProcessor(FrameProcessor):
    """
    Pipecat FrameProcessor that captures pipeline event timings and MetricsFrames
    to measure real-time latency and usage for Deepgram STT, Gemini LLM, and Cartesia TTS.
    """

    def __init__(
        self,
        tracker: MetricsTracker = global_metrics_tracker,
        capture_input_events: bool = True,
        capture_service_events: bool = True,
        capture_native_metrics: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tracker = tracker
        self.capture_input_events = capture_input_events
        self.capture_service_events = capture_service_events
        self.capture_native_metrics = capture_native_metrics

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self.capture_native_metrics and isinstance(frame, MetricsFrame):
            self.tracker.record_metrics_frame(frame)

        # Microphone Audio Input Streaming
        elif self.capture_input_events and isinstance(frame, (InputAudioRawFrame, UserAudioRawFrame, AudioRawFrame)):
            self.tracker._last_audio_frame_time = time.perf_counter()

        # STT Speech Start
        elif self.capture_input_events and isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
            self.tracker._stt_speech_start_time = time.perf_counter()
            self.tracker._stt_speech_stop_time = None
            live_event_hub.set_activity("Listening", "Caller is speaking")

        # STT Speech Stop
        elif self.capture_input_events and isinstance(frame, (UserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame)):
            self.tracker._user_stopped_time = time.perf_counter()
            self.tracker._stt_speech_stop_time = time.perf_counter()
            live_event_hub.set_activity("Converting speech to text", "Deepgram is transcribing")

        # STT Transcription Ready
        elif self.capture_service_events and isinstance(frame, TranscriptionFrame):
            latency_ms = None
            now = time.perf_counter()
            if getattr(self.tracker, "_user_stopped_time", None):
                latency_ms = (now - self.tracker._user_stopped_time) * 1000.0
            elif getattr(self.tracker, "_stt_speech_stop_time", None):
                latency_ms = (now - self.tracker._stt_speech_stop_time) * 1000.0
            elif getattr(self.tracker, "_last_audio_frame_time", None):
                latency_ms = (now - self.tracker._last_audio_frame_time) * 1000.0
            elif getattr(self.tracker, "_stt_speech_start_time", None):
                latency_ms = (now - self.tracker._stt_speech_start_time) * 1000.0

            self.tracker.record_stt_transcription(frame.text, latency_ms=latency_ms)
            live_event_hub.set_activity("Transcript ready", "Preparing the next response")

        # LLM Request Start
        elif self.capture_service_events and isinstance(frame, LLMFullResponseStartFrame):
            self.tracker._llm_start_time = time.perf_counter()
            self.tracker._llm_first_token_time = None
            if not self.tracker._llm_active:
                self.tracker.llm_calls += 1
                self.tracker._llm_active = True
            live_event_hub.set_activity("LLM is responding", "Gemini is generating a reply")

        # LLM Response Tokens & Start of TTS Text Input
        elif self.capture_service_events and isinstance(frame, (LLMTextFrame, TextFrame, TTSTextFrame)):
            live_event_hub.set_activity("Converting text to speech", "Cartesia is preparing audio")
            # Record LLM TTFB
            if getattr(self.tracker, "_llm_start_time", None) and not getattr(self.tracker, "_llm_first_token_time", None):
                self.tracker._llm_first_token_time = time.perf_counter()
                llm_ttfb_ms = (self.tracker._llm_first_token_time - self.tracker._llm_start_time) * 1000.0
                if llm_ttfb_ms > 0:
                    self.tracker.llm_ttfbs.append(llm_ttfb_ms)

            # Start TTS timer when LLM outputs text for TTS
            if not getattr(self.tracker, "_tts_start_time", None):
                self.tracker._tts_start_time = time.perf_counter()

        # LLM Response Complete
        elif self.capture_service_events and isinstance(frame, LLMFullResponseEndFrame):
            if getattr(self.tracker, "_llm_start_time", None):
                llm_total_ms = (time.perf_counter() - self.tracker._llm_start_time) * 1000.0
                if llm_total_ms > 0:
                    self.tracker.llm_total_latencies.append(llm_total_ms)
                self.tracker._llm_start_time = None
            self.tracker._llm_active = False

            if not getattr(self.tracker, "_tts_start_time", None):
                self.tracker._tts_start_time = time.perf_counter()

        # TTS Audio Output Start (TTFB)
        elif self.capture_service_events and isinstance(frame, (TTSStartedFrame, BotStartedSpeakingFrame, TTSAudioRawFrame)):
            live_event_hub.set_activity("Speaking", "Playing the assistant response")
            if not self.tracker._tts_active:
                self.tracker.tts_calls += 1
                self.tracker._tts_active = True
            if getattr(self.tracker, "_tts_start_time", None):
                tts_ttfb_ms = (time.perf_counter() - self.tracker._tts_start_time) * 1000.0
                if tts_ttfb_ms > 0:
                    self.tracker.tts_ttfbs.append(tts_ttfb_ms)
                self.tracker._tts_start_time = None

        # TTS Output Stop
        elif self.capture_service_events and isinstance(frame, TTSStoppedFrame):
            self.tracker._tts_start_time = None
            self.tracker._tts_active = False
            live_event_hub.set_activity("Listening", "Waiting for the caller")

        await self.push_frame(frame, direction)
