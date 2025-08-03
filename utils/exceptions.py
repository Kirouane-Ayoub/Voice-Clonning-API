class TTSAPIException(Exception):
    """Base exception for TTS API."""

    pass


class ModelError(TTSAPIException):
    """Exception raised when model operations fail."""

    pass


class AudioProcessingError(TTSAPIException):
    """Exception raised when audio processing fails."""

    pass


class ValidationError(TTSAPIException):
    """Exception raised when input validation fails."""

    pass
