import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from pydub import AudioSegment
import re
import os

from .base_model import BaseTTSModel
from utils.exceptions import ModelError


class ChatterboxTTSModel(BaseTTSModel):
    """ChatterboxTTS model implementation."""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.load_model()

    def load_model(self):
        """Load the ChatterboxTTS model."""
        try:
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
        except Exception as e:
            raise ModelError(f"Failed to load ChatterboxTTS model: {e}")

    def generate_audio(
        self,
        audio_file_path: str,
        output_file: str,
        text: str,
        max_chunk_length: int = 500,
        language: str = "en-us",
        temperature: float = 0.5,
        speed: float = 1.0,
    ) -> str:
        """Generate audio from text using ChatterboxTTS."""
        try:
            # Convert MP3 to WAV if needed
            wav_file_path = self._convert_to_wav(audio_file_path)

            # Split text into chunks
            chunks = self._split_text_intelligently(text, max_chunk_length)

            # Generate audio for each chunk
            audio_chunks = []
            temp_files = []

            for i, chunk in enumerate(chunks):
                # Generate audio for this chunk
                wav = self.model.generate(
                    chunk,
                    audio_prompt_path=wav_file_path,
                    cfg_weight=0.3,
                    exaggeration=0.7,
                    temperature=temperature,
                )

                # Save temporary file
                temp_file = f"temp_chunk_{i}_{id(self)}.wav"
                ta.save(temp_file, wav, self.model.sr)
                temp_files.append(temp_file)
                audio_chunks.append(wav)

            # Concatenate all audio chunks
            final_audio = torch.cat(audio_chunks, dim=-1)

            # Save final concatenated audio
            ta.save(str(output_file), final_audio, self.model.sr)
            print(f"===========> Audio saved to {output_file} <===========")

            # Clean up temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

            # Clean up converted WAV file if it was created
            if wav_file_path != audio_file_path and os.path.exists(wav_file_path):
                os.remove(wav_file_path)

            return output_file

        except Exception as e:
            raise ModelError(f"ChatterboxTTS generation failed: {e}")

    def _convert_to_wav(self, audio_file_path: str) -> str:
        """Convert MP3 to WAV format if needed."""
        if audio_file_path.lower().endswith(".wav"):
            return audio_file_path

        try:
            audio = AudioSegment.from_file(audio_file_path)
            wav_file_path = audio_file_path.rsplit(".", 1)[0] + "_converted.wav"
            audio.export(wav_file_path, format="wav")
            return wav_file_path
        except Exception as e:
            raise ModelError(f"Failed to convert audio to WAV: {e}")

    def _split_text_intelligently(self, text: str, max_length: int = 500) -> list:
        """Split text into chunks intelligently."""
        if len(text) <= max_length:
            return [text]

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Single sentence is too long, split by words
                    words = sentence.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 > max_length:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                                temp_chunk = word
                            else:
                                chunks.append(word)
                        else:
                            temp_chunk += " " + word if temp_chunk else word
                    if temp_chunk:
                        current_chunk = temp_chunk
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
