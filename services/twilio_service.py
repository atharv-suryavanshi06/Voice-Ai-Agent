"""
twilio_service.py

Provides helper functions and service integration for Twilio telephonic calling
and TwiML response generation.
"""

from typing import Optional
import sys
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect

from core import config


class TwilioService:
    def __init__(self):
        self.account_sid = config.TWILIO_ACCOUNT_SID
        self.auth_token = config.TWILIO_AUTH_TOKEN
        self.from_number = config.TWILIO_PHONE_NUMBER
        self._client: Optional[Client] = None

    @property
    def client(self) -> Client:
        if not self._client:
            if not self.account_sid or not self.auth_token:
                raise ValueError("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must be configured in .env")
            self._client = Client(self.account_sid, self.auth_token)
        return self._client

    def generate_twiml(self, websocket_url: str) -> str:
        """
        Generate TwiML XML instructing Twilio to stream call audio to the WebSocket URL.
        """
        response = VoiceResponse()
        connect = Connect()
        connect.stream(url=websocket_url)
        response.append(connect)
        return str(response)

    def make_outbound_call(self, to_phone_number: str, twiml_url: str) -> str:
        """
        Initiate an outbound call to the target phone number.
        When answered, Twilio fetches TwiML from twiml_url.
        """
        if not self.from_number:
            raise ValueError("TWILIO_PHONE_NUMBER must be set in .env")

        print(f"Initiating Twilio outbound call to {to_phone_number} from {self.from_number}...")
        call = self.client.calls.create(
            to=to_phone_number,
            from_=self.from_number,
            url=twiml_url,
        )
        print(f"Twilio Call SID created: {call.sid}")
        return call.sid


def setup_ngrok_tunnel(port: int) -> str:
    """
    Starts an ngrok tunnel to expose local port to public internet if PUBLIC_URL is not set.
    """
    if config.PUBLIC_URL:
        public_url = config.PUBLIC_URL.strip().rstrip("/")
        print(f"Using explicitly configured PUBLIC_URL: {public_url}")
        return public_url

    try:
        from pyngrok import ngrok
        if config.NGROK_AUTHTOKEN:
            ngrok.set_auth_token(config.NGROK_AUTHTOKEN)
        
        tunnel = ngrok.connect(port, "http")
        public_url = tunnel.public_url.rstrip("/")
        print(f"Started ngrok tunnel: {public_url} -> http://localhost:{port}")
        return public_url
    except Exception as e:
        err_msg = str(e)
        print("\n==========================================================================", file=sys.stderr)
        print(" [!] NGROK TUNNEL AUTHENTICATION ERROR", file=sys.stderr)
        print("==========================================================================", file=sys.stderr)
        if "ERR_NGROK_4018" in err_msg or "authentication failed" in err_msg:
            print(" ngrok requires a free auth token to start tunnels.", file=sys.stderr)
            print(" How to fix (pick one):", file=sys.stderr)
            print("  1. Get your free token: https://dashboard.ngrok.com/get-started/your-authtoken", file=sys.stderr)
            print("     Paste it in your .env file: NGROK_AUTHTOKEN=your_token_here", file=sys.stderr)
            print("  2. OR if you already have a public URL, set in .env: PUBLIC_URL=https://your-url.ngrok-free.app", file=sys.stderr)
        else:
            print(f" ngrok error: {e}", file=sys.stderr)
            print(" Please set NGROK_AUTHTOKEN or PUBLIC_URL in your .env file.", file=sys.stderr)
        print("==========================================================================\n", file=sys.stderr)
        raise SystemExit(1)

