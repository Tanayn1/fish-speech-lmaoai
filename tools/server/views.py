import io
import os
import time
from http import HTTPStatus

import numpy as np
import ormsgpack
import soundfile as sf
import torch
from kui.asgi import (
    Body,
    HTTPException,
    HttpView,
    JSONResponse,
    Routes,
    StreamResponse,
    request,
)
from loguru import logger
from typing_extensions import Annotated
from kui.wsgi import Path


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
from tools.server.api_utils import (
    buffer_to_async_generator,
    get_content_type,
    inference_async,
    inference_async_vapi
)
from tools.server.inference import inference_wrapper as inference

from tools.server.model_manager import ModelManager
from tools.server.model_utils import (
    batch_vqgan_decode,
    cached_vqgan_batch_encode,
)
from tools.server.db import fetchVoice
import requests

MAX_NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", 1))

routes = Routes()

audio_cache: dict[str, CachedAudio] = {}
audio_cache_dir = "audio_cache"


@routes.http("/v1/health")
class Health(HttpView):
    @classmethod
    async def get(cls):
        return JSONResponse({"status": "ok"})

    @classmethod
    async def post(cls):
        return JSONResponse({"status": "ok"})


@routes.http.post("/v1/vqgan/encode")
async def vqgan_encode(req: Annotated[ServeVQGANEncodeRequest, Body(exclusive=True)]):
    # Get the model from the app
    model_manager: ModelManager = request.app.state.model_manager
    decoder_model = model_manager.decoder_model

    # Encode the audio
    start_time = time.time()
    tokens = cached_vqgan_batch_encode(decoder_model, req.audios)
    logger.info(f"[EXEC] VQGAN encode time: {(time.time() - start_time) * 1000:.2f}ms")

    # Return the response
    return ormsgpack.packb(
        ServeVQGANEncodeResponse(tokens=[i.tolist() for i in tokens]),
        option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
    )


@routes.http.post("/v1/vqgan/decode")
async def vqgan_decode(req: Annotated[ServeVQGANDecodeRequest, Body(exclusive=True)]):
    # Get the model from the app
    model_manager: ModelManager = request.app.state.model_manager
    decoder_model = model_manager.decoder_model

    # Decode the audio
    tokens = [torch.tensor(token, dtype=torch.int) for token in req.tokens]
    start_time = time.time()
    audios = batch_vqgan_decode(decoder_model, tokens)
    logger.info(f"[EXEC] VQGAN decode time: {(time.time() - start_time) * 1000:.2f}ms")
    audios = [audio.astype(np.float16).tobytes() for audio in audios]

    # Return the response
    return ormsgpack.packb(
        ServeVQGANDecodeResponse(audios=audios),
        option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
    )


@routes.http.post("/v1/tts")
async def tts(req: Annotated[ServeTTSRequest, Body(exclusive=True)]):
    # Get the model from the app
    app_state = request.app.state
    model_manager: ModelManager = app_state.model_manager
    engine = model_manager.tts_inference_engine
    sample_rate = engine.decoder_model.sample_rate

    # Check if the text is too long
    if app_state.max_text_length > 0 and len(req.text) > app_state.max_text_length:
        raise HTTPException(
            HTTPStatus.BAD_REQUEST,
            content=f"Text is too long, max length is {app_state.max_text_length}",
        )

    # Check if streaming is enabled
    if req.streaming and req.format != "wav":
        raise HTTPException(
            HTTPStatus.BAD_REQUEST,
            content="Streaming only supports WAV format",
        )

    # Perform TTS
    if req.streaming:
        return StreamResponse(
            iterable=inference_async(req, engine),
            headers={
                "Content-Disposition": f"attachment; filename=audio.{req.format}",
            },
            content_type=get_content_type(req.format),
        )
    else:
        fake_audios = next(inference(req, engine))
        buffer = io.BytesIO()
        sf.write(
            buffer,
            fake_audios,
            sample_rate,
            format=req.format,
        )

        return StreamResponse(
            iterable=buffer_to_async_generator(buffer.getvalue()),
            headers={
                "Content-Disposition": f"attachment; filename=audio.{req.format}",
            },
            content_type=get_content_type(req.format),
        )
    

@routes.http.post('/v1/tts/vapi/{voice_id}')
async def ttsVapi(
    req: Annotated[VapiTTSRequest, Body(exclusive=True)],
    voice_id: Annotated[str, Path()]
):
    voice_id = request.path_params["voice_id"]
    print("voice_id", voice_id)
    print("vapi message", req.message)

    print("Fetching Voice from cache")


    cached_voice = audio_cache.get(voice_id)

    if cached_voice is None:
        print("Audio not in cache fetching from db")
        voice = await fetchVoice(voice_id)
        uri = voice.uri if voice and voice.uri else "https://parrot-samples.s3.amazonaws.com/gargamel/Nigel.wav"
        reference_text = voice.transcription if voice and voice.transcription else "hey hows it going mate i would love to catch up" 
        filename = uri.split("/")[-1]
        save_path = os.path.join(audio_cache_dir, filename)

        response = requests.get(uri)
        response.raise_for_status() 

        with open(save_path, "wb") as f:
            f.write(response.content)

        audio_cache[voice_id] = CachedAudio(path=save_path, transcription=reference_text)
        cached_voice = CachedAudio(path=save_path, transcription=reference_text)

    with open(cached_voice.path, "rb") as f:
        audio_bytes = f.read()

    reference_audio = ServeReferenceAudio(
        audio=audio_bytes,
        text=cached_voice.transcription
    )

    app_state = request.app.state
    model_manager: ModelManager = app_state.model_manager
    engine = model_manager.tts_inference_engine

    print(f"Generating audio for {req.message.text}")

    ttsRequest = ServeTTSRequest(text=req.message.text, format="pcm", streaming=True, references=[reference_audio])


    return StreamResponse(
        iterable=inference_async_vapi(ttsRequest, engine, req.message.sampleRate),
        headers={
            "Content-Disposition": f"attachment; filename=audio.pcm",
        },
        content_type="application/octet-stream",
    )

@routes.http.post('/v1/tts/cache/{voice_id}')
async def cache_audio(
    voice_id: Annotated[str, Path()]
):
    print("Caching audio")

    voice = await fetchVoice(voice_id)

    if voice is None or voice.uri is None:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Voice not found or voice has no audio sample")
    

    os.makedirs(audio_cache_dir, exist_ok=True)

    response = requests.get(voice.uri)
    filename = voice.uri.split("/")[-1]
    save_path = os.path.join(audio_cache_dir, filename)

    response = requests.get(voice.uri)
    response.raise_for_status() 

    with open(save_path, "wb") as f:
        f.write(response.content)

    audio_cache[voice_id] = CachedAudio(path=save_path, transcription=voice.transcription)
    
    return JSONResponse({"message": "Success"})

@routes.http.delete("/v1/tts/cache/{voice_id}")
async def delete_cache_audio(
    voice_id: Annotated[str, Path()]
):
    print("Deleting cached audio")
    audio = audio_cache[voice_id]

    if audio is None:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Audio not in cache")
    
    os.remove(audio.path)

    return JSONResponse({"message": "Success"})








    

    
