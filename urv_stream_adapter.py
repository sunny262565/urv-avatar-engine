"""In-memory MuseTalk 1.5 adapter for the URV WebSocket worker."""

import copy
import glob
import os
import pickle
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import WhisperModel

from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.blending import get_image_blending
from musetalk.utils.preprocessing import read_imgs
from musetalk.utils.utils import datagen, load_all_model


_MODELS = {}


def _load_models(device: torch.device):
    key = str(device)
    if key in _MODELS:
        return _MODELS[key]
    home = Path(os.getenv("MUSETALK_HOME", "/opt/MuseTalk"))
    previous_cwd = os.getcwd()
    try:
        os.chdir(home)
        vae, unet, pe = load_all_model(
            unet_model_path=str(home / "models/musetalkV15/unet.pth"),
            vae_type="sd-vae",
            unet_config=str(home / "models/musetalkV15/musetalk.json"),
            device=device,
        )
    finally:
        os.chdir(previous_cwd)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pe = pe.to(device=device, dtype=dtype)
    vae.vae = vae.vae.to(device=device, dtype=dtype)
    unet.model = unet.model.to(device=device, dtype=dtype)
    whisper_dir = home / "models/whisper"
    processor = AudioProcessor(feature_extractor_path=str(whisper_dir))
    whisper = WhisperModel.from_pretrained(str(whisper_dir))
    whisper = whisper.to(device=device, dtype=dtype).eval()
    whisper.requires_grad_(False)
    value = (vae, unet, pe, processor, whisper, dtype)
    _MODELS[key] = value
    return value


class StreamingRenderer:
    def __init__(self, avatar_dir: Path, device: str):
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = requested
        self.vae, self.unet, self.pe, self.processor, self.whisper, self.dtype = _load_models(requested)
        self.timesteps = torch.tensor([0], device=requested)
        self.batch_size = int(os.getenv("URV_MUSETALK_BATCH_SIZE", "8"))
        self.sample_rate = int(os.getenv("URV_PCM_SAMPLE_RATE", "24000"))
        self.index = 0
        self._load_avatar(Path(avatar_dir))

    def _load_avatar(self, root: Path):
        with (root / "coords.pkl").open("rb") as handle:
            self.coords = pickle.load(handle)
        with (root / "mask_coords.pkl").open("rb") as handle:
            self.mask_coords = pickle.load(handle)
        self.latents = torch.load(root / "latents.pt", map_location="cpu")
        frame_paths = sorted(glob.glob(str(root / "full_imgs" / "*.[jpJP][pnPN]*[gG]")))
        mask_paths = sorted(glob.glob(str(root / "mask" / "*.[jpJP][pnPN]*[gG]")))
        self.frames = read_imgs(frame_paths)
        self.masks = read_imgs(mask_paths)
        lengths = {len(self.coords), len(self.mask_coords), len(self.latents), len(self.frames), len(self.masks)}
        if len(lengths) != 1 or not self.frames:
            raise RuntimeError("prepared avatar artifacts are empty or inconsistent")

    def _write_pcm(self, samples: np.ndarray, path: str):
        pcm = np.asarray(samples, dtype="<i2")
        with wave.open(path, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(self.sample_rate)
            output.writeframes(pcm.tobytes())

    @torch.inference_mode()
    def render_pcm(self, samples: np.ndarray, motion: dict) -> list[np.ndarray]:
        del motion  # MuseTalk derives mouth motion from audio; expression control is separate work.
        if np.asarray(samples).size == 0:
            return []
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            self._write_pcm(samples, audio_file.name)
            features, audio_length = self.processor.get_audio_feature(audio_file.name, weight_dtype=self.dtype)
        chunks = self.processor.get_whisper_chunk(
            features, self.device, self.dtype, self.whisper, audio_length,
            fps=25, audio_padding_length_left=2, audio_padding_length_right=2,
        )
        output = []
        phase = self.index % len(self.latents)
        phased_latents = self.latents[phase:] + self.latents[:phase]
        generator = datagen(chunks, phased_latents, self.batch_size, device=str(self.device))
        for whisper_batch, latent_batch in generator:
            encoded = self.pe(whisper_batch.to(self.device))
            latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)
            predicted = self.unet.model(
                latent_batch, self.timesteps, encoder_hidden_states=encoded
            ).sample.to(device=self.device, dtype=self.vae.vae.dtype)
            for face in self.vae.decode_latents(predicted):
                slot = self.index % len(self.frames)
                original = copy.deepcopy(self.frames[slot])
                x1, y1, x2, y2 = self.coords[slot]
                resized = cv2.resize(face.astype(np.uint8), (x2 - x1, y2 - y1))
                blended = get_image_blending(
                    original, resized, self.coords[slot], self.masks[slot], self.mask_coords[slot]
                )
                output.append(cv2.cvtColor(blended, cv2.COLOR_BGR2RGBA))
                self.index += 1
        return output


def create_renderer(avatar_dir: Path, device: str = "cuda") -> StreamingRenderer:
    return StreamingRenderer(avatar_dir=avatar_dir, device=device)
