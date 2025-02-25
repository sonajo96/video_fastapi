from fastapi import FastAPI
from pydantic import BaseModel
import whisper
import yt_dlp
import os
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
import warnings
warnings.simplefilter("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=UserWarning)

# Initialize FastAPI
app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
     allow_origins=["http://127.0.0.1:5501"], # Allow all origins (change this to your frontend URL in production)
    allow_credentials=False,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


# Load Whisper model
model = whisper.load_model("base")

# Set up Google Gemini API Key
GEMINI_API_KEY = "AIzaSyCP9glaUSILORsFPKXU9xqf3MTWpEsheGE"
genai.configure(api_key=GEMINI_API_KEY)

# Variable to store the latest transcript
latest_transcript = None

class VideoURL(BaseModel):
    url: str

class QuestionRequest(BaseModel):
    question: str  # Only question field, no URL

def download_audio(youtube_url):
    """Download audio from a YouTube video."""
    output_path = "audio.mp3"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "audio.%(ext)s",
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    if not os.path.exists(output_path):
        raise FileNotFoundError("Audio download failed.")

    return output_path

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper."""
    result = model.transcribe(audio_path)
    os.remove(audio_path) 
    return result["text"]

def generate_answer(question, context):
    """Generate an AI response using Gemini."""
    prompt = f"Here is the transcript of a YouTube video:\n\n{context}\n\nNow, answer this question: {question}"
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text

@app.post("/transcribe/")
async def transcribe_video(video: VideoURL):
    """Download and transcribe a YouTube video."""
    global latest_transcript  # Store only the latest transcript

    try:
        audio_path = download_audio(video.url)
        latest_transcript = transcribe_audio(audio_path)  # Store latest transcript

        return {"message": "Transcription stored successfully.", "transcript": latest_transcript}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    """Answer questions based on the latest transcribed video."""
    try:
        if not latest_transcript:
            return {"error": "No transcript available. Please transcribe a video first."}

        # Generate answer using Gemini
        answer = generate_answer(request.question, latest_transcript)
        return {"answer": answer}  # Only return the answer
    except Exception as e:
        return {"error": str(e)}