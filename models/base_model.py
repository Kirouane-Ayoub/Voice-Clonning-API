from abc import ABC, abstractmethod


class BaseTTSModel(ABC):
    """Abstract base class for TTS models."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None

    @abstractmethod
    def load_model(self):
        """Load the TTS model."""
        pass

    @abstractmethod
    def generate_audio(
        self,
        audio_file_path: str,
        text: str,
        max_chunk_length: int = 500,
        language: str = "en-us",
        temperature: float = 0.5,
        speed: float = 1.0,
    ) -> str:
        """Generate audio from text using reference audio."""
        pass

    def cleanup(self):
        """Cleanup model resources."""
        if hasattr(self, "model") and self.model is not None:
            del self.model
            if self.device == "cuda":
                import torch

                torch.cuda.empty_cache()
