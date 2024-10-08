import os
import json
import base64
import asyncio
import websockets
from pydub import AudioSegment
from dotenv import load_dotenv
import sounddevice as sd
from scipy.io.wavfile import write
import requests
import concurrent.futures

# Load environment variables
load_dotenv()

# Set your OpenAI API key and other credentials
API_KEY = os.getenv('API_KEY')  # Ensure to set this in your .env file
SKYFLOW_ACCOUNT_ID = os.getenv('SKYFLOW_ACCOUNT_ID')
BEARER_TOKEN = os.getenv('BEARER_TOKEN')
VAULT_ID = os.getenv('VAULT_ID')
URL = os.getenv('URL')  # Base URL for the Detect API
URL_WS = os.getenv('URL_WS') # WebSocket URL for OpenAI Realtime API

# Function to record audio from microphone
def record_audio(duration=10, fs=44100):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write('input.wav', fs, recording)  # Save as WAV file
    print("Recording complete.")
    return 'input.wav'

# Function to convert WAV to base64
def convert_wav_to_base64(file_path):
    with open(file_path, "rb") as wav_file:
        base64_encoded = base64.b64encode(wav_file.read()).decode('utf-8')
    return base64_encoded

# Function to convert audio to 16-bit PCM, 24kHz, mono, and base64 encode it
def audio_to_base64(audio_file_path):
    # Load audio using pydub
    audio = AudioSegment.from_file(audio_file_path)
    
    # Resample to 24kHz mono and convert to 16-bit PCM
    audio = audio.set_frame_rate(24000).set_channels(1).set_sample_width(2)
    pcm_audio = audio.raw_data  # Get raw 16-bit PCM data
    
    # Encode the PCM audio data to base64
    pcm_base64 = base64.b64encode(pcm_audio).decode('utf-8')
    
    return pcm_base64

# Function to send audio via WebSocket
async def send_audio(ws, base64_audio):
    event = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "audio": base64_audio
                }
            ]
        }
    }
    
    # Send the event containing the base64-encoded audio
    await ws.send(json.dumps(event))
    await ws.send(json.dumps({"type": "response.create"}))

# Function to detect audio using the Detect API
def detect_audio(file_path):
    base64_audio = convert_wav_to_base64(file_path)
    
    url = f'{URL}/v1/detect/file'
    headers = {
        'Content-Type': 'application/json',
        'x-skyflow-account-id': SKYFLOW_ACCOUNT_ID,
        'Authorization': f'Bearer {BEARER_TOKEN}',
    }
    
    payload = {
        "file": base64_audio,
        "data_format": "wav",
        "audio": {
            "output_processed_audio": True,
            "options": {
                "bleep_gain": -30,
                "bleep_start_padding": 0,
                "bleep_stop_padding": 0
            }
        },
        "accuracy": "high_multilingual",
        "restrict_entity_types": ["location", "ssn","account_number"],
        "input_type": "BASE64",
        "vault_id": VAULT_ID
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Function to check processing status
def check_status(status_id):
    url = f'{URL}/v1/detect/status/{status_id}?vault_id={VAULT_ID}'
    headers = {'Authorization': f'Bearer {BEARER_TOKEN}'}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Main function to handle the entire process
async def main():
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor()
    
    # Step 1: Record audio
    audio_file_path = await loop.run_in_executor(executor, record_audio, 10, 44100)
    
    # Step 2: Call Detect API
    print("Sending audio for detection...")
    detect_response = await loop.run_in_executor(executor, detect_audio, audio_file_path)
    
    if detect_response and 'status_url' in detect_response:
        status_id = detect_response['status_url'].split('/')[-1]
        
        # Check status until processing is complete
        while True:
            status_response = await loop.run_in_executor(executor, check_status, status_id)
            
            if status_response and status_response.get("status") == "SUCCESS":
                for output in status_response["output"]:
                    if output["processedFileType"] == "redacted_audio":
                        processed_audio_base64 = output["processedFile"]
                        processed_audio_path = 'processed_output.wav'
                        with open(processed_audio_path, "wb") as f:
                            f.write(base64.b64decode(processed_audio_base64))
                        print(f"Processed audio saved as {processed_audio_path}")
                        
                        # Now, use processed_audio_path as input to OpenAI API
                        base64_audio = audio_to_base64(processed_audio_path)
                        
                        async with websockets.connect(
                            URL_WS,
                            extra_headers={
                                "Authorization": f"Bearer {API_KEY}",
                                "OpenAI-Beta": "realtime=v1"
                            }
                        ) as ws:
                            print("Connected to OpenAI Realtime API.")
                            
                            # Send the encoded audio
                            await send_audio(ws, base64_audio)
                            
                            accumulated_audio = ""
                            async for message in ws:
                                parsed_message = json.loads(message)
                                
                                # Accumulate the audio deltas in the response
                                if parsed_message.get("type") == "response.audio.delta":
                                    delta = parsed_message.get("delta")
                                    accumulated_audio += delta
                                
                                # When the response is complete, save the full base64 audio
                                elif parsed_message.get("type") == "response.audio.done":
                                    print("Streaming completed")
                                    
                                    pcm_audio = base64.b64decode(accumulated_audio)
    
                                    # Reconstruct the AudioSegment from the raw PCM data
                                    audio = AudioSegment(
                                        data=pcm_audio,
                                        sample_width=2,      # 16-bit PCM corresponds to 2 bytes
                                        frame_rate=24000,    # The sample rate you used for encoding
                                        channels=1           # Mono audio
                                    )
    
                                    # Export the audio to a file (e.g., WAV format)
                                    audio.export("output.wav", format="wav")
                                    
                                    print("Assistant's reply saved to output.wav")
                                    break
                            break  # Break out of the async for loop
                break  # Break out of the while loop
            elif status_response.get("status") == "FAILED":
                print("Audio processing failed.")
                break
            else:
                print("Processing... checking again in 2 seconds.")
                await asyncio.sleep(2)
    else:
        print("Error during audio detection.")

if __name__ == "__main__":
    asyncio.run(main())
