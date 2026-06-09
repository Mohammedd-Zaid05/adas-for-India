import os
from groq import Groq
from dotenv import load_dotenv
import io

load_dotenv()

def transcribe_audio(audio_bytes):
    """
    Transcribes audio bytes using Groq's Whisper API.
    
    Args:
        audio_bytes (bytes): The raw audio data.
        
    Returns:
        dict: Transcription text and success status.
    """
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return {"text": "GROQ_API_KEY not found in environment", "success": False}

        client = Groq(api_key=api_key)
        
        # Groq's transcription API expects a file-like object
        audio_file = io.BytesIO(audio_bytes)
        
        transcription = client.audio.transcriptions.create(
            file=("audio.wav", audio_file),
            model="whisper-large-v3-turbo",
            response_format="json"
        )
        
        return {
            "text": transcription.text,
            "success": True
        }
    except Exception as e:
        return {
            "text": str(e),
            "success": False
        }
