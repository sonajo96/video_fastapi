import os
import whisper
import yt_dlp
from pydub import AudioSegment
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import google.generativeai as genai

app = FastAPI()

class VideoRequest(BaseModel):
    youtube_url: str

class ChatRequest(BaseModel):
    question: str

# Define transcript_memory as a global variable
transcript_memory: str = ""

gemini_api = "AIzaSyCP9glaUSILORsFPKXU9xqf3MTWpEsheGE"
genai.configure(api_key=gemini_api)

def download_audio(youtube_url):
    """Download YouTube audio and convert to WAV"""
    output_path = "audio.mp3"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    if os.path.exists("audio.mp3.mp3"):
        os.rename("audio.mp3.mp3", "audio.mp3")

    audio = AudioSegment.from_file("audio.mp3", format="mp3")
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export("audio1.wav", format="wav")
    return "audio1.wav"

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

@app.post("/download_transcribe")
async def download_transcribe(request: VideoRequest):
    global transcript_memory  # Access the global variable
    try:
        audio_file = download_audio(request.youtube_url)
        transcript = transcribe_audio(audio_file)
        transcript_memory = transcript  # Store the transcript in the global variable
        return {"transcript": transcript}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def ask_gemini(question, transcript):
    """Generate AI response using Gemini API"""
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"The following is a transcript of a video:\n\n{transcript}\n\nBased on this transcript, answer the question:\n{question}"
    response = model.generate_content(prompt)
    return response.text if response else "Sorry, I couldn't generate a response."

@app.post("/chat")
async def chat(request: ChatRequest):
    global transcript_memory
    print("Current Transcript Memory:", transcript_memory)  # Debugging log

    if not transcript_memory:
        raise HTTPException(status_code=400, detail="No transcript available. Please process a video first.")
    
    answer = ask_gemini(request.question, transcript_memory)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)