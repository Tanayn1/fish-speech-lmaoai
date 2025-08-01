FROM python:3.12-slim-bookworm AS stage-1
ARG TARGETARCH

ARG HF_TOKEN
ARG HUGGINGFACE_MODEL=openaudio-s1-mini
ARG HF_ENDPOINT=https://huggingface.co

WORKDIR /opt/fish-speech

RUN set -ex \
    && pip install huggingface_hub \
    && huggingface-cli login --token ${HF_TOKEN} \
    && HF_ENDPOINT=${HF_ENDPOINT} huggingface-cli download --resume-download fishaudio/${HUGGINGFACE_MODEL} --local-dir checkpoints/${HUGGINGFACE_MODEL}

FROM python:3.12-slim-bookworm
ARG TARGETARCH

ARG DEPENDENCIES="  \
    ca-certificates \
    libsox-dev \
    build-essential \
    cmake \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -ex \
    && rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' >/etc/apt/apt.conf.d/keep-cache \
    && apt-get update \
    && apt-get -y install --no-install-recommends ${DEPENDENCIES} \
    && echo "no" | dpkg-reconfigure dash

WORKDIR /opt/fish-speech

COPY . .

RUN --mount=type=cache,target=/root/.cache,sharing=locked \
    set -ex \
    && pip install -e .[stable]

COPY --from=stage-1 /opt/fish-speech/checkpoints /opt/fish-speech/checkpoints

ENV GRADIO_SERVER_NAME="0.0.0.0"

EXPOSE 8080

CMD [ "python", "-m", "tools.api_server"]