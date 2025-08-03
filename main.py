import asyncio
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field, validator
import uvicorn

from models.chatterbox_model import ChatterboxTTSModel
from models.zonos_model import ZonosTTSModel
from utils.audio_utils import AudioProcessor
from utils.exceptions import ModelError, AudioProcessingError
from config.settings import Settings


class TTSRequest(BaseModel):
    """Request model for TTS generation."""

    text: str = Field(
        ..., min_length=1, max_length=10000, description="Text to synthesize"
    )
    max_chunk_length: Optional[int] = Field(
        500, ge=100, le=2000, description="Maximum chunk length for processing"
    )
    language: Optional[str] = Field("en-us", description="Language code")
    temperature: Optional[float] = Field(
        0.5, ge=0.1, le=1.0, description="Temperature for generation"
    )
    speed: Optional[float] = Field(
        1.0, ge=0.5, le=2.0, description="Speech speed multiplier"
    )

    @validator("text")
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class TTSResponse(BaseModel):
    """Response model for TTS generation."""

    status: str
    message: str
    audio_length_seconds: Optional[float] = None
    processing_time_seconds: Optional[float] = None


# Global variables for models and thread pool
models = {}
thread_pool = None
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    global models, thread_pool

    logger.info("Starting FastAPI TTS application...")

    # Initialize thread pool
    thread_pool = ThreadPoolExecutor(max_workers=settings.max_workers)
    logger.info(f"Thread pool initialized with {settings.max_workers} workers")

    # Initialize models
    try:
        logger.info("Loading TTS models...")
        models["chatterbox"] = ChatterboxTTSModel(device=settings.device)
        models["zonos"] = ZonosTTSModel(device=settings.device)
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down application...")
    if thread_pool:
        thread_pool.shutdown(wait=True)
        logger.info("Thread pool shut down")

    # Cleanup models
    for model_name, model in models.items():
        try:
            model.cleanup()
            logger.info(f"Model {model_name} cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up model {model_name}: {e}")


app = FastAPI(
    title="Voice Cloning API",
    description="Production-grade API for voice cloning using multiple TTS models",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Voice Cloning API is running",
        "available_models": list(models.keys()),
        "device": settings.device,
    }


@app.get("/models")
async def list_models():
    """List available models and their status."""
    model_status = {}
    for name, model in models.items():
        try:
            model_status[name] = {
                "status": "ready",
                "device": model.device,
                "model_type": type(model).__name__,
            }
        except Exception as e:
            model_status[name] = {"status": "error", "error": str(e)}

    return {"models": model_status}


@app.post("/clone/{model_name}")
async def clone_voice(
    model_name: str,
    audio_file: UploadFile = File(..., description="Reference audio file (WAV or MP3)"),
    text: str = Form(..., description="Text to synthesize"),
    max_chunk_length: Optional[int] = Form(500, description="Maximum chunk length"),
    language: Optional[str] = Form("en-us", description="Language code"),
    temperature: Optional[float] = Form(0.5, description="Temperature for generation"),
    speed: Optional[float] = Form(1.0, description="Speech speed multiplier"),
):
    """
    Clone voice using the specified model.

    Args:
        model_name: Name of the model to use ('chatterbox' or 'zonos')
        audio_file: Reference audio file for voice cloning
        text: Text to synthesize
        max_chunk_length: Maximum characters per chunk for long text processing
        language: Language code (default: en-us)
        temperature: Generation temperature (0.1-1.0)
        speed: Speech speed multiplier (0.5-2.0)

    Returns:
        StreamingResponse with the generated audio file
    """
    start_time = asyncio.get_event_loop().time()
    request_id = str(uuid.uuid4())

    logger.info(f"Request {request_id}: Starting voice cloning with {model_name}")

    # Validate model name
    if model_name not in models:
        logger.warning(f"Request {request_id}: Invalid model name: {model_name}")
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not available. Available models: {list(models.keys())}",
        )

    # Validate request data
    try:
        request_data = TTSRequest(
            text=text,
            max_chunk_length=max_chunk_length,
            language=language,
            temperature=temperature,
            speed=speed,
        )
    except Exception as e:
        logger.warning(f"Request {request_id}: Validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Validation error: {e}")

    # Validate audio file
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")

    file_extension = Path(audio_file.filename).suffix.lower()
    if file_extension not in [".wav", ".mp3"]:
        raise HTTPException(
            status_code=400, detail="Audio file must be in WAV or MP3 format"
        )

    # Create temporary files
    temp_dir = Path(tempfile.gettempdir()) / "tts_api" / request_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    input_audio_path = temp_dir / f"input{file_extension}"
    output_audio_path = temp_dir / "output.wav"

    try:
        # Save uploaded audio file
        logger.info(f"Request {request_id}: Saving uploaded audio file")
        with open(input_audio_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)

        # Validate audio file
        try:
            audio_processor = AudioProcessor()
            await audio_processor.validate_audio_file(input_audio_path)
        except AudioProcessingError as e:
            logger.warning(f"Request {request_id}: Audio validation failed: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

        # Generate audio using the specified model
        logger.info(f"Request {request_id}: Starting audio generation")
        model = models[model_name]

        # Run model inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        output_path = await loop.run_in_executor(
            thread_pool,
            model.generate_audio,
            str(input_audio_path),
            str(output_audio_path),
            request_data.text,
            request_data.max_chunk_length,
            request_data.language,
            request_data.temperature,
            request_data.speed,
        )

        if not output_path or not Path(output_path).exists():
            raise ModelError("Model failed to generate audio")

        # Move output to our temp directory
        final_output_path = output_audio_path
        if output_path != str(final_output_path):
            import shutil

            shutil.move(output_path, final_output_path)

        # Get audio duration for response metadata
        try:
            waveform, sample_rate = torchaudio.load(final_output_path)
            duration = waveform.shape[1] / sample_rate
        except Exception:
            duration = None

        processing_time = asyncio.get_event_loop().time() - start_time

        logger.info(
            f"Request {request_id}: Audio generation completed in {processing_time:.2f}s, "
            f"duration: {duration:.2f}s"
            if duration
            else f"Request {request_id}: Audio generation completed"
        )

        # Create streaming response
        def generate_audio_stream():
            try:
                with open(final_output_path, "rb") as audio_file:
                    while True:
                        chunk = audio_file.read(8192)  # 8KB chunks
                        if not chunk:
                            break
                        yield chunk
            finally:
                # Cleanup temp files after streaming
                try:
                    import shutil

                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    logger.error(
                        f"Request {request_id}: Error cleaning up temp files: {e}"
                    )

        return StreamingResponse(
            generate_audio_stream(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=cloned_audio_{request_id}.wav",
                "X-Processing-Time-Seconds": str(round(processing_time, 2)),
                "X-Audio-Duration-Seconds": str(round(duration, 2))
                if duration
                else "unknown",
                "X-Model-Used": model_name,
                "X-Request-ID": request_id,
            },
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ModelError as e:
        logger.error(f"Request {request_id}: Model error: {e}")
        raise HTTPException(status_code=500, detail=f"Model error: {e}")
    except AudioProcessingError as e:
        logger.error(f"Request {request_id}: Audio processing error: {e}")
        raise HTTPException(status_code=400, detail=f"Audio processing error: {e}")
    except Exception as e:
        logger.error(f"Request {request_id}: Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,  # Single worker to share models in memory
        log_level="info",
    )
