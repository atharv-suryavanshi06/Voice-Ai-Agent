# Live Conversation Dashboard

This React/Vite frontend connects to the Python backend's `/events` WebSocket and displays completed caller/assistant messages plus live STT, retrieval, LLM, and TTS activity.

From `frontend/`:

```powershell
npm install
npm run dev
```

Start the backend from the repository root first:

```powershell
python main.py --mode mic
```

The frontend defaults to `ws://localhost:8765/events`. To use another backend:

```powershell
$env:VITE_EVENTS_URL='wss://example.com/events'
npm run dev
```

Build the production assets with `npm run build`.
