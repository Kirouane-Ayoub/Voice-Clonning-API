import os
import re

import torch
import torchaudio
from zonos.conditioning import make_cond_dict
from zonos.model import Zonos

from utils.exceptions import ModelError

from .base_model import BaseTTSModel


class ZonosTTSModel(BaseTTSModel):
    """Zonos TTS model implementation."""

    def __init__(
        self, model_name: str = "Zyphra/Zonos-v0.1-transformer", device: str = "cuda"
    ):
        self.model_name = model_name
        super().__init__(device)

    def load_model(self):
        """Load the Zonos model."""
        try:
            self.model = Zonos.from_pretrained(self.model_name, device=self.device)
        except Exception as e:
            raise ModelError(f"Failed to load Zonos model: {e}")

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
        """Generate audio from text using Zonos."""
        try:
            if self.model is None:
                self.load_model()
            # Load reference audio and create speaker embedding
            wav, sampling_rate = torchaudio.load(audio_file_path)
            speaker = self.model.make_speaker_embedding(wav, sampling_rate)

            # Split text into chunks
            chunks = self._split_text_intelligently(text, max_chunk_length)

            # Generate audio for each chunk
            audio_chunks = []
            temp_files = []

            for i, chunk in enumerate(chunks):
                # Create conditioning for this chunk
                cond_dict = make_cond_dict(
                    text=chunk,
                    speaker=speaker,
                    language=language,
                )
                conditioning = self.model.prepare_conditioning(cond_dict)

                # Generate codes and decode to audio
                codes = self.model.generate(conditioning)
                wavs = self.model.autoencoder.decode(codes).cpu()

                # Ensure proper tensor shape
                if wavs.dim() == 2:
                    current_wav = wavs.unsqueeze(0)
                elif wavs.dim() == 3:
                    current_wav = wavs[0:1]
                else:
                    current_wav = wavs.unsqueeze(0).unsqueeze(0)

                # Save temporary file
                temp_file = f"temp_zonos_chunk_{i}_{id(self)}.wav"
                torchaudio.save(
                    temp_file, current_wav[0], self.model.autoencoder.sampling_rate
                )
                temp_files.append(temp_file)

                # Store for concatenation
                audio_chunks.append(current_wav[0])

            # Concatenate all audio chunks
            if audio_chunks:
                target_channels = audio_chunks[0].shape[0]
                processed_chunks = []

                for chunk in audio_chunks:
                    if chunk.shape[0] != target_channels:
                        if chunk.shape[0] == 1 and target_channels == 2:
                            chunk = chunk.repeat(2, 1)
                        elif chunk.shape[0] == 2 and target_channels == 1:
                            chunk = chunk.mean(dim=0, keepdim=True)
                    processed_chunks.append(chunk)

                final_audio = torch.cat(processed_chunks, dim=1)

                # Save final audio
                torchaudio.save(
                    str(output_file), final_audio, self.model.autoencoder.sampling_rate
                )
                print(f"===========> Audio saved to {output_file} <===========")
            else:
                raise ModelError("No audio chunks generated")

            # Clean up temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

            return output_file

        except Exception as e:
            raise ModelError(f"Zonos generation failed: {e}")

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
