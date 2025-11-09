"""FastAPI server for Kani TTS + Whisper (OpenAI-compatible)"""

import io
import os
import time
import tempfile
import threading
import queue
import struct
import numpy as np
import requests

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field
from typing import Optional, Literal

from scipy.io.wavfile import write as wav_write
from faster_whisper import WhisperModel

from audio import LLMAudioPlayer, StreamingAudioWriter
from generation import TTSGenerator
from config import CHUNK_SIZE, LOOKBACK_FRAMES, TEMPERATURE, TOP_P, MAX_TOKENS
from nemo.utils.nemo_logging import Logger

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
nemo_logger = Logger()
nemo_logger.remove_stream_handlers()

# -----------------------------------------------------------------------------
# FASTAPI APP SETUP
# -----------------------------------------------------------------------------
app = FastAPI(title="Kani TTS + Whisper API (OpenAI Compatible)", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# GLOBAL MODELS
# -----------------------------------------------------------------------------
generator = None
player = None
whisper_model = None

# -----------------------------------------------------------------------------
# VOICE MAPPING
# -----------------------------------------------------------------------------
VOICE_MAPPING = {
    "natalie-fr": "natalie",
    "makati-fr": "makati",
    "andrew-en": "andrew",
    # "kore-en": "kore",
    "shimmer-ht": "shimmer",
    "onyx-ht": "onyx",
    "kore-ht": "kore",
    "daniel-ht": "daniel",
    "bob-ht": "bob",
    "jean-ht": "jean",
    "zefi-ht": "zefi",
    "charles-ht": "charles",
}

# -----------------------------------------------------------------------------
# REQUEST MODELS
# -----------------------------------------------------------------------------
class TTSRequest(BaseModel):
    text: str
    temperature: Optional[float] = TEMPERATURE
    max_tokens: Optional[int] = MAX_TOKENS
    top_p: Optional[float] = TOP_P
    chunk_size: Optional[int] = CHUNK_SIZE
    lookback_frames: Optional[int] = LOOKBACK_FRAMES
    speaker_id: Optional[str] = None


class OpenAITTSRequest(BaseModel):
    model: str = Field(default="tts-1", description="TTS model to use")
    input: str = Field(..., description="Text to generate audio for", max_length=4096)
    voice: str = Field(..., description="Voice name")
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = "mp3"
    speed: Optional[float] = Field(default=1.0, ge=0.25, le=4.0)
    temperature: Optional[float] = TEMPERATURE
    max_tokens: Optional[int] = MAX_TOKENS
    top_p: Optional[float] = TOP_P
    chunk_size: Optional[int] = CHUNK_SIZE
    lookback_frames: Optional[int] = LOOKBACK_FRAMES


class WhisperTranscribeRequest(BaseModel):
    audio_url: Optional[str] = None


# -----------------------------------------------------------------------------
# STARTUP EVENT
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize TTS + Whisper models"""
    global generator, player, whisper_model
    print("🚀 Initializing TTS and Whisper models...")
    generator = TTSGenerator()
    player = LLMAudioPlayer(generator.tokenizer)
    whisper_model = WhisperModel("small")
    print("✅ Models initialized successfully!")


# -----------------------------------------------------------------------------
# HEALTH CHECK
# -----------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "tts_initialized": generator is not None,
        "whisper_initialized": whisper_model is not None,
    }


# -----------------------------------------------------------------------------
# TTS ENDPOINTS (OPENAI COMPATIBLE)
# -----------------------------------------------------------------------------
@app.post("/v1/audio/speech")
async def openai_speech(request: OpenAITTSRequest):
    """Generate audio from text input using Kani TTS"""
    if not generator or not player:
        raise HTTPException(status_code=503, detail="TTS not initialized")

    try:
        speaker_id = VOICE_MAPPING.get(request.voice, "makati")
        temperature = request.temperature or TEMPERATURE

        audio_writer = StreamingAudioWriter(
            player,
            output_file=None,
            chunk_size=request.chunk_size,
            lookback_frames=request.lookback_frames,
        )
        audio_writer.start()
        print(request.input)
        generator.generate(
            request.input,
            audio_writer,
            max_tokens=request.max_tokens,
            speaker_id=speaker_id,
            temperature=temperature,
        )

        audio_writer.finalize()
        if not audio_writer.audio_chunks:
            raise HTTPException(status_code=500, detail="No audio generated")

        full_audio = np.concatenate(audio_writer.audio_chunks)
        if request.speed != 1.0:
            full_audio = adjust_speed(full_audio, request.speed)

        audio_buffer = convert_audio_format(full_audio, request.response_format)
        media_types = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "application/octet-stream",
        }

        return Response(
            content=audio_buffer.read(),
            media_type=media_types.get(request.response_format, "audio/mpeg"),
            headers={"Content-Disposition": f"attachment; filename=speech.{request.response_format}"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/speech/stream")
async def openai_speech_stream(request: OpenAITTSRequest):
    """Stream audio chunks as they’re generated"""
    if not generator or not player:
        raise HTTPException(status_code=503, detail="TTS not initialized")

    async def audio_chunk_generator():
        chunk_queue = queue.Queue()
        speaker_id = VOICE_MAPPING.get(request.voice, "makati")

        class ChunkList(list):
            def append(self, chunk):
                super().append(chunk)
                chunk_queue.put(("chunk", chunk))

        audio_writer = StreamingAudioWriter(
            player,
            output_file=None,
            chunk_size=request.chunk_size,
            lookback_frames=request.lookback_frames,
        )
        audio_writer.audio_chunks = ChunkList()

        def generate():
            try:
                audio_writer.start()
                generator.generate(
                    request.input,
                    audio_writer,
                    max_tokens=request.max_tokens,
                    speaker_id=speaker_id,
                )
                audio_writer.finalize()
                chunk_queue.put(("done", None))
            except Exception as e:
                chunk_queue.put(("error", str(e)))

        gen_thread = threading.Thread(target=generate)
        gen_thread.start()

        try:
            while True:
                msg_type, data = chunk_queue.get(timeout=30)
                if msg_type == "chunk":
                    if request.speed != 1.0:
                        data = adjust_speed(data, request.speed)
                    pcm_data = (data * 32767).astype(np.int16)
                    chunk_bytes = pcm_data.tobytes()
                    yield struct.pack("<I", len(chunk_bytes)) + chunk_bytes
                elif msg_type == "done":
                    yield struct.pack("<I", 0)
                    break
                elif msg_type == "error":
                    yield struct.pack("<I", 0xFFFFFFFF)
                    break
        finally:
            gen_thread.join()

    return StreamingResponse(
        audio_chunk_generator(),
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": "22050", "X-Channels": "1", "X-Bit-Depth": "16"},
    )


# -----------------------------------------------------------------------------
# WHISPER ENDPOINT
# -----------------------------------------------------------------------------
@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(None),
    model: str = Form("small"),
    language: Optional[str] = Form(None),
    task: Optional[Literal["transcribe", "translate"]] = Form("transcribe"),
    request: WhisperTranscribeRequest = None,
):
    """Transcribe or translate audio using Whisper (OpenAI-compatible)"""
    global whisper_model
    if not whisper_model:
        whisper_model = WhisperModel(model)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        if file:
            content = await file.read()
            tmp.write(content)
        elif request and request.audio_url:
            r = requests.get(request.audio_url)
            tmp.write(r.content)
        else:
            raise HTTPException(status_code=400, detail="No audio file or URL provided")
        tmp_path = tmp.name

    try:
        segments, info = whisper_model.transcribe(tmp_path, language=language, task=task)
        text_output = ""
        segments_list = []
        for segment in segments:
            segments_list.append(
                {"start": segment.start, "end": segment.end, "text": segment.text.strip()}
            )
            text_output += segment.text.strip() + " "

        os.remove(tmp_path)
        return {
            "model": model,
            "language": info.language,
            "duration": info.duration,
            "text": text_output.strip(),
            "segments": segments_list,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------
def adjust_speed(audio: np.ndarray, speed: float) -> np.ndarray:
    if speed == 1.0:
        return audio
    try:
        from scipy import signal
        new_length = int(len(audio) / speed)
        return signal.resample(audio, new_length)
    except ImportError:
        indices = np.linspace(0, len(audio) - 1, int(len(audio) / speed))
        return np.interp(indices, np.arange(len(audio)), audio)


def convert_audio_format(audio: np.ndarray, format: str) -> io.BytesIO:
    buffer = io.BytesIO()
    if format == "wav":
        wav_write(buffer, 22050, audio)
    elif format == "pcm":
        pcm_data = (audio * 32767).astype(np.int16)
        buffer.write(pcm_data.tobytes())
    else:
        print(f"Warning: Format '{format}' not fully implemented, returning WAV")
        wav_write(buffer, 22050, audio)
    buffer.seek(0)
    return buffer


# -----------------------------------------------------------------------------
# ROOT
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "name": "Kani TTS + Whisper API (OpenAI Compatible)",
        "version": "1.1.0",
        "endpoints": {
            "OpenAI Compatible": {
                "/v1/audio/speech": "POST - Generate TTS audio",
                "/v1/audio/speech/stream": "POST - Stream TTS audio",
                "/v1/audio/transcriptions": "POST - Transcribe audio using Whisper",
            },
            "System": {
                "/health": "GET - Check service health",
            },
        },
        "supported_voices": list(VOICE_MAPPING.keys()),
        "supported_formats": ["mp3", "opus", "aac", "flac", "wav", "pcm"],
    }


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("🎤 Starting Kani TTS + Whisper API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
