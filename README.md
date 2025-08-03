# Voice Cloning API

Production-grade API for voice cloning using multiple TTS (Text-to-Speech) models, including Zonos and ChatterboxTTS.

## Features

- REST API for voice cloning
- Supports reference audio in WAV or MP3 format
- Multiple TTS models: Zonos, ChatterboxTTS
- Intelligent text chunking for long inputs
- Streaming audio responses
- Configurable via environment variables
- Docker and Docker Compose support

## Quickstart

### 1. Clone the repository

```sh
git clone <your-repo-url>
cd cloning_api
```

### 2. Install dependencies

#### Using setup script

```sh
bash setup.sh
```

### 3. Run the API

#### Locally

```sh
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

#### With Docker

```sh
docker-compose up --build
```

## API Endpoints

- `GET /` — Health check
- `GET /models` — List available models
- `POST /clone/{model_name}` — Clone voice using specified model

### Example Request

```sh
curl -X 'POST' 'http://localhost:8000/clone/chatterbox'
        -H 'accept: application/json'
        -H 'Content-Type: multipart/form-data'
        -F 'audio_file=@Test.mp3;type=audio/mpeg'
        -F 'text=write anything here ... '
        -F 'max_chunk_length=500'
        -F 'language=en-us'
        -F 'temperature=0.5'
        -F 'speed=1'
```

## Configuration

Settings can be adjusted in [`config/settings.py`](config/settings.py) or via environment variables:

- `MAX_WORKERS`: Number of thread pool workers
- `LOG_LEVEL`: Logging level
