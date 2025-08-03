import asyncio
from pathlib import Path
import torchaudio
from .exceptions import AudioProcessingError


class AudioProcessor:
    """Utility class for audio processing operations."""

    @staticmethod
    async def validate_audio_file(file_path: Path) -> bool:
        """Validate audio file format and readability."""
        try:
            # Run validation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, AudioProcessor._sync_validate_audio_file, file_path
            )
        except Exception as e:
            raise AudioProcessingError(f"Audio validation failed: {e}")

    @staticmethod
    def _sync_validate_audio_file(file_path: Path) -> bool:
        """Synchronous audio file validation."""
        try:
            # Try to load the audio file
            waveform, sample_rate = torchaudio.load(str(file_path))

            # Basic validation
            if waveform.shape[1] == 0:
                raise AudioProcessingError("Audio file appears to be empty")

            if sample_rate <= 0:
                raise AudioProcessingError("Invalid sample rate")

            # Check duration (should be between 1 second and 10 minutes)
            duration = waveform.shape[1] / sample_rate
            if duration < 1.0:
                raise AudioProcessingError("Audio file too short (minimum 1 second)")

            if duration > 600.0:  # 10 minutes
                raise AudioProcessingError("Audio file too long (maximum 10 minutes)")

            return True

        except Exception as e:
            if isinstance(e, AudioProcessingError):
                raise
            raise AudioProcessingError(f"Cannot read audio file: {e}")
