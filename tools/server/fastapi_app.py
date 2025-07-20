from fastapi import FastAPI, Path, Body, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Annotated
import os
import requests
from http import HTTPStatus
from tools.server.inference import inference_wrapper as inference

from tools.server.model_manager import ModelManager
from tools.server.model_utils import (
    batch_vqgan_decode,
    cached_vqgan_batch_encode,
)
from tools.server.db import fetchVoice
from fish_speech.utils.schema import (
    ServeTTSRequest,
    ServeReferenceAudio,
    VapiTTSRequest,
    ServeVQGANDecodeRequest,
    ServeVQGANDecodeResponse,
    ServeVQGANEncodeRequest,
    ServeVQGANEncodeResponse,
    CachedAudio
)
import io
import time
import numpy as np
import ormsgpack
import soundfile as sf
import torch
from loguru import logger
from tools.server.db import connect, disconnect
from tools.server.api_utils import (
    buffer_to_async_generator,
    get_content_type,
    inference_async,
    inference_async_vapi
)


app = FastAPI()

@app.on_event("startup")
async def startup_event():
    await connect()

    app.state.model_manager = ModelManager(
        mode="tts",
        device="cuda",
        half=True,
        compile=True,
        llama_checkpoint_path="checkpoints/openaudio-s1-mini",
        decoder_checkpoint_path="checkpoints/openaudio-s1-mini/codec.pth",
        decoder_config_name="modded_dac_vq",
    )

    logger.info(f"Startup done")

audio_cache = {}  # Dict[str, CachedAudio]
audio_cache_dir = "audio_cache"



# ------------------- Helper Function -------------------
async def get_cached_or_fetch_voice(voice_id: str):
    cached_voice = audio_cache.get(voice_id)

    if cached_voice is None:
        print("Audio not in cache, fetching from db")
        voice = await fetchVoice(voice_id)
        uri = voice.uri if voice and voice.uri else "https://parrot-samples.s3.amazonaws.com/gargamel/Nigel.wav"
        reference_text = voice.transcription if voice and voice.transcription else "hey hows it going mate i would love to catch up" 
        filename = uri.split("/")[-1]
        save_path = os.path.join(audio_cache_dir, filename)

        response = requests.get(uri)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            f.write(response.content)

        cached_voice = CachedAudio(path=save_path, transcription=reference_text)
        audio_cache[voice_id] = cached_voice

    return cached_voice

# @app.middleware("http")
# async def log_request_body(request: Request, call_next):
#     body = await request.body()
#     print(f"Request body (raw): {body.decode('utf-8') if body else '(empty)'}")
#     response = await call_next(request)
#     return response

# ------------------- GET and POST Audio -------------------
@app.get("/v1/tts/vapi/{voice_id}")
@app.post("/v1/tts/vapi/{voice_id}")
async def tts_vapi(
    request: Request,
    voice_id: str = Path(...),
    req: VapiTTSRequest = Body(...)
):
    print("voice_id", voice_id)
    print("vapi message", req.message)

    cached_voice = await get_cached_or_fetch_voice(voice_id)

    with open(cached_voice.path, "rb") as f:
        audio_bytes = f.read()

    reference_audio = ServeReferenceAudio(
        audio=audio_bytes,
        text=cached_voice.transcription
    )

    model_manager: ModelManager = request.app.state.model_manager
    engine = model_manager.tts_inference_engine

    print(f"Generating audio for {req.message.text}")

    ttsRequest = ServeTTSRequest(
        text=req.message.text,
        format="pcm",
        streaming=True,
        references=[reference_audio]
    )

    stream = inference_async_vapi(ttsRequest, engine, req.message.sampleRate)

    return StreamingResponse(
        stream,
        headers={"Content-Disposition": "attachment; filename=audio.pcm"},
        media_type="application/octet-stream"
    )

# ------------------- Cache Audio -------------------
@app.post('/v1/tts/cache/{voice_id}')
async def cache_audio(voice_id: str = Path(...)):
    print("Caching audio")
    voice = await fetchVoice(voice_id)

    if voice is None or voice.uri is None:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Voice not found or voice has no audio sample")

    os.makedirs(audio_cache_dir, exist_ok=True)

    response = requests.get(voice.uri)
    response.raise_for_status()

    filename = voice.uri.split("/")[-1]
    save_path = os.path.join(audio_cache_dir, filename)

    with open(save_path, "wb") as f:
        f.write(response.content)

    audio_cache[voice_id] = CachedAudio(path=save_path, transcription=voice.transcription)

    return JSONResponse({"message": "Success"})

# ------------------- Delete Cached Audio -------------------
@app.delete("/v1/tts/cache/delete/{voice_id}")
async def delete_cache_audio(voice_id: str = Path(...)):
    print("Deleting cached audio")
    audio = audio_cache.get(voice_id)

    if audio is None:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Audio not in cache")

    os.remove(audio.path)
    del audio_cache[voice_id]

    return JSONResponse({"message": "Success"})

# uvicorn tools.server.fastapi_app:app --reload --host 0.0.0.0 --port 8080
