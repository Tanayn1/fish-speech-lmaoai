from http import HTTPStatus

import numpy as np
from kui.asgi import HTTPException
import torch
import torchaudio.functional as F

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.utils.schema import ServeTTSRequest


def resample_audio(audio_np, orig_sr, target_sr):
    audio_tensor = torch.from_numpy(audio_np)
    resampled_tensor = F.resample(audio_tensor, orig_sr, target_sr)
    return resampled_tensor.numpy()

AMPLITUDE = 32768  # Needs an explaination


def inference_wrapper(req: ServeTTSRequest, engine: TTSInferenceEngine):
    """
    Wrapper for the inference function.
    Used in the API server.
    """
    count = 0
    for result in engine.inference(req):
        match result.code:
            case "header":
                if isinstance(result.audio, tuple):
                    yield result.audio[1]

            case "error":
                raise HTTPException(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    content=str(result.error),
                )

            case "segment":
                count += 1
                if isinstance(result.audio, tuple):
                    yield (result.audio[1] * AMPLITUDE).astype(np.int16).tobytes()

            case "final":
                count += 1
                if isinstance(result.audio, tuple):
                    yield result.audio[1]
                return None  # Stop the generator

    if count == 0:
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            content="No audio generated, please check the input text.",
        )


def vapi_inference_wrapper(req: ServeTTSRequest, engine: TTSInferenceEngine, sample_rate: int):
    """
    Wrapper for the inference function.
    Used in the API server.
    """
    count = 0
    for result in engine.inference(req):
        match result.code:
            case "header":
                if isinstance(result.audio, tuple):
                    yield result.audio[1]

            case "error":
                raise HTTPException(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    content=str(result.error),
                )

            case "segment" | "final":
                count += 1
                if isinstance(result.audio, tuple):
                    orig_sr, segment = result.audio
                    segment = resample_audio(segment, orig_sr, sample_rate)
                    segment = (segment * AMPLITUDE).astype(np.int16)
                    yield segment.tobytes()

                if result.code == "final":
                    return

    if count == 0:
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            content="No audio generated, please check the input text.",
        )