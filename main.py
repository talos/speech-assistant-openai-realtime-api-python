import os
import json
import base64
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
from dotenv import load_dotenv
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
import io
from googleapiclient.http import MediaIoBaseDownload
import requests
from datetime import datetime

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PORT = int(os.getenv('PORT', 5050))
# SYSTEM_MESSAGE = (
#     "You are a helpful and bubbly AI assistant who loves to chat about "
#     "anything the user is interested in and is prepared to offer them facts. "
#     "You have a penchant for dad jokes, owl jokes, and rickrolling – subtly. "
#     "Always stay positive, but work in a joke when appropriate."
# )
SYSTEM_MESSAGE = None
VOICE = 'alloy'
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated',
    'response.done', 'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
    'session.created',
    'conversation.item.input_audio_transcription.completed'
]
SHOW_TIMING_MATH = False

app = FastAPI()

if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')

def get_instructions():
    if SYSTEM_MESSAGE is None:
        raise ValueError("The SYSTEM_MESSAGE has not been initialized yet!")
    return SYSTEM_MESSAGE

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    response = VoiceResponse()
    # <Say> punctuation to improve text-to-speech flow
    response.say("O.K.")
    host = request.url.hostname
    print('handle_incoming_call params', request.query_params)
    phone_from = request.query_params['From']
    print('call FROM', phone_from)
    connect = Connect()
    stream = connect.stream(url=f'wss://{host}/media-stream')
    stream.parameter(name="foo", value='bar')
    stream.parameter(name="From", value=phone_from)
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket, instructions=Depends(get_instructions)):
    """Handle WebSocket connections between Twilio and OpenAI."""
    print("Client connected")
    await websocket.accept()

    print(f'websocket.query_params {websocket.query_params}')

    async with websockets.connect(
        'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17',
        extra_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
    ) as openai_ws:
        await initialize_session(openai_ws, instructions)

        # Connection specific state
        stream_sid = None
        latest_media_timestamp = 0
        last_assistant_item = None
        mark_queue = []
        response_start_timestamp_twilio = None
        transcription = []
        ai_generated_summary = ''
        twilio_disconnected = False
        awaiting_final_transcript = False

        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
            nonlocal stream_sid, latest_media_timestamp, transcription, twilio_disconnected
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    # print('twilio message', data)
                    if data['event'] == 'media' and openai_ws.open:
                        latest_media_timestamp = int(data['media']['timestamp'])
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }
                        await openai_ws.send(json.dumps(audio_append))
                    elif data['event'] == 'start':
                        stream_sid = data['start']['streamSid']
                        parameters = data['start']['customParameters']
                        foo = parameters.get("foo")
                        phone_from = parameters.get("From")
                        print(f"Incoming stream has started {stream_sid}, {parameters}")
                        response_start_timestamp_twilio = None
                        latest_media_timestamp = 0
                        last_assistant_item = None
                    elif data['event'] == 'mark':
                        if mark_queue:
                            mark_queue.pop(0)
                    elif data['event'] == 'stop':
                        print("Twilio 'stop' transcription", transcription)
                        twilio_disconnected = True
                        await send_summary_item(openai_ws)
                        awaiting_final_transcript = True
            except WebSocketDisconnect:
                print("Client disconnected.")
                #await send_summary_item(openai_ws)
                if openai_ws.open:
                    #await openai_ws.close()
                    twilio_disconnected = True
            except Exception as e:
                print('Other exception from twilio', e)

        async def send_to_twilio():
            """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
            nonlocal stream_sid, last_assistant_item, response_start_timestamp_twilio, transcription, twilio_disconnected, awaiting_final_transcript, ai_generated_summary
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)
                    if response['type'] in LOG_EVENT_TYPES:
                        print(f"Received event: {response['type']}", response)

                    if response.get('type') == 'response.audio.delta' and 'delta' in response:
                        audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode('utf-8')
                        audio_delta = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": audio_payload
                            }
                        }
                        await websocket.send_json(audio_delta)

                        if response_start_timestamp_twilio is None:
                            response_start_timestamp_twilio = latest_media_timestamp
                            if SHOW_TIMING_MATH:
                                print(f"Setting start timestamp for new response: {response_start_timestamp_twilio}ms")

                        # Update last_assistant_item safely
                        if response.get('item_id'):
                            last_assistant_item = response['item_id']

                        await send_mark(websocket, stream_sid)

                    # Trigger an interruption. Your use case might work better using `input_audio_buffer.speech_stopped`, or combining the two.
                    if response.get('type') == 'input_audio_buffer.speech_started':
                        print("Speech started detected.")
                        if last_assistant_item:
                            print(f"Interrupting response with id: {last_assistant_item}")
                            await handle_speech_started_event()

                    # Handle transcriptions
                    if response.get('type') == 'conversation.item.input_audio_transcription.completed' and 'transcript' in response:
                        transcription.append({ 'role': 'user', 'transcript': response['transcript'] })
                        print('transcription', transcription)

                    if response.get('type') == 'response.done' and response.get('response')['object'] == 'realtime.response' and response.get('response')['status'] == 'completed' and 'transcript' in response.get('response').get('output')[0].get('content')[0]:
                        transcription.append({ 'role': 'assistant', 'transcript': response.get('response').get('output')[0].get('content')[0]['transcript'] })
                        print('transcription', transcription)

                    if response.get('type') == 'response.done' and twilio_disconnected and not awaiting_final_transcript:
                        awaiting_final_transcript = True
                        print('response done and twilio disconnected and not yet awaiting final transcript', response)
                        await send_summary_item(openai_ws)

                    if response.get('type') == 'response.done' and twilio_disconnected and awaiting_final_transcript:
                        print('response done and twilio disconnected and awaiting final transcript', response)
                        ai_generated_summary = response.get('response').get('output')[0].get('content')[0]['text']
                        print('AI generated summary', ai_generated_summary)

                        await openai_ws.close()

                    # if response.get('type') == 'response.done' and response.get('response')['object'] == 'realtime.response' and response.get('response')['status'] == 'cancelled':
                    #     print('response.done cancelled', response)

            except Exception as e:
                print(f"Error in send_to_twilio: {e}")
                #await send_summary_item(openai_ws)

        async def handle_speech_started_event():
            """Handle interruption when the caller's speech starts."""
            nonlocal response_start_timestamp_twilio, last_assistant_item
            print("Handling speech started event.")
            if mark_queue and response_start_timestamp_twilio is not None:
                elapsed_time = latest_media_timestamp - response_start_timestamp_twilio
                if SHOW_TIMING_MATH:
                    print(f"Calculating elapsed time for truncation: {latest_media_timestamp} - {response_start_timestamp_twilio} = {elapsed_time}ms")

                if last_assistant_item:
                    if SHOW_TIMING_MATH:
                        print(f"Truncating item with ID: {last_assistant_item}, Truncated at: {elapsed_time}ms")

                    truncate_event = {
                        "type": "conversation.item.truncate",
                        "item_id": last_assistant_item,
                        "content_index": 0,
                        "audio_end_ms": elapsed_time
                    }
                    await openai_ws.send(json.dumps(truncate_event))

                await websocket.send_json({
                    "event": "clear",
                    "streamSid": stream_sid
                })

                mark_queue.clear()
                last_assistant_item = None
                response_start_timestamp_twilio = None

        async def send_mark(connection, stream_sid):
            if stream_sid:
                mark_event = {
                    "event": "mark",
                    "streamSid": stream_sid,
                    "mark": {"name": "responsePart"}
                }
                await connection.send_json(mark_event)
                mark_queue.append('responsePart')

        try:
            await asyncio.gather(receive_from_twilio(), send_to_twilio())
        except Exception as e:
            print('Error from asyncio.gather', e)

        send_webhook('some number', transcription, ai_generated_summary)

async def send_initial_conversation_item(openai_ws):
    """Send initial conversation item if AI talks first."""
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Greet the user with 'Hello there! How can I help you?'"
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))

async def send_summary_item(openai_ws):
    """Summarize conversation"""
    session_update = { "type": "session.update", "session": { "modalities": ["text"] } }
    print('Sending session update for summary:', json.dumps(session_update))

    await openai_ws.send(json.dumps(session_update))
    summary_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Provide a transcript of the entire conversation, notating yourself as 'Donna:' and the user as 'User:'. Provide a commentary on top about anything that was edited or removed. "
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(summary_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))
    print('Sent summary item', summary_item)

async def initialize_session(openai_ws, instructions):
    """Control initial session with OpenAI."""
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": instructions,
            "modalities": ["text", "audio"],
            # Otherwise, input audio won't be transcribed as text
            "input_audio_transcription": { "model": "whisper-1" },
            "temperature": 0.8,
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

    # Uncomment the next line to have the AI speak first
    await send_initial_conversation_item(openai_ws)

def authenticate_with_env():
    """Authenticate using environment variables."""
    scopes = ['https://www.googleapis.com/auth/drive']
    client_email = os.getenv('GOOGLE_CLIENT_EMAIL')
    private_key = os.getenv('GOOGLE_PRIVATE_KEY').replace('\\n', '\n')  # Handle newline escape in env
    project_id = os.getenv('GOOGLE_PROJECT_ID')

    if not all([client_email, private_key, project_id]):
        raise EnvironmentError("Missing one or more required environment variables: GOOGLE_CLIENT_EMAIL, GOOGLE_PRIVATE_KEY, GOOGLE_PROJECT_ID")

    credentials = Credentials.from_service_account_info(
        {
            "type": "service_account",
            "project_id": project_id,
            "private_key_id": "chat_pkey_id",  # Private key ID is optional for manual setup
            "private_key": private_key,
            "client_email": client_email,
            "client_id": "chat_pkey_client_id",  # Optional
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{client_email}"
        },
        scopes=scopes
    )
    return build('drive', 'v3', credentials=credentials)

def get_file_as_markdown(service, file_id):
    """Retrieve a Google Docs file as Markdown and return it as a string."""
    request = service.files().export_media(fileId=file_id, mimeType='text/markdown')
    markdown_content = io.BytesIO()
    downloader = MediaIoBaseDownload(markdown_content, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")

    # Decode the content from bytes to string
    return markdown_content.getvalue().decode('utf-8')

def send_webhook(from_phone, transcription, ai_generated_summary):
    webhook_data = {
        'Incoming phone number': from_phone,
        'Transcript': transcription,
        'AI generated summary': ai_generated_summary,
        'Date time stamp': datetime.now().isoformat()
    }
    print('sent webhook', webhook_data)
    requests.post(os.getenv('MAKE_URL'), json=webhook_data)

if __name__ == "__main__":
    file_id = os.getenv('GOOGLE_FILE_ID')

    # Authenticate and build the service
    service = authenticate_with_env()

    # Get the file content as Markdown
    #global SYSTEM_MESSAGE
    SYSTEM_MESSAGE = get_file_as_markdown(service, file_id)
    #print("File content as Markdown:")
    #print(SYSTEM_MESSAGE)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
