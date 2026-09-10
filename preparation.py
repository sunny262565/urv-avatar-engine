import json
import os
import shutil
import subprocess
import wave
from pathlib import Path


MUSETALK_HOME = Path(os.getenv("MUSETALK_HOME", "/opt/MuseTalk"))
PERSISTENT_ROOT = Path(os.getenv("URV_DATA_ROOT", "/runpod-volume"))
SOURCE_ROOT = Path(os.getenv("URV_SOURCE_ROOT", "/app/avatars"))
RESULT_ROOT = PERSISTENT_ROOT / "musetalk-results" / "v15" / "avatars"
OFFICIAL_RESULT_ROOT = MUSETALK_HOME / "results" / "v15" / "avatars"
REGISTERED = {"teacher_meera", "dr_arjun_mehta", "professor_arya", "rohan_verma"}


def _status_path(avatar_id: str) -> Path:
    return RESULT_ROOT / avatar_id / "urv_preparation.json"


def status(avatar_id: str) -> dict:
    if avatar_id not in REGISTERED:
        raise ValueError("unknown avatar_id")
    path = _status_path(avatar_id)
    if not path.exists():
        return {"avatar_id": avatar_id, "state": "missing", "preprocessed": False}
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_layout() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    OFFICIAL_RESULT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    if OFFICIAL_RESULT_ROOT.exists() and not OFFICIAL_RESULT_ROOT.is_symlink():
        raise RuntimeError(f"Refusing to replace non-symlink {OFFICIAL_RESULT_ROOT}")
    if not OFFICIAL_RESULT_ROOT.exists():
        OFFICIAL_RESULT_ROOT.symlink_to(RESULT_ROOT, target_is_directory=True)


def _silence_wav(path: Path, seconds: float = 0.5, sample_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x00" * int(seconds * sample_rate))


def prepare(avatar_id: str) -> dict:
    if avatar_id not in REGISTERED:
        raise ValueError("unknown avatar_id")
    source = SOURCE_ROOT / avatar_id / "source.png"
    if not source.exists():
        raise FileNotFoundError(f"missing source portrait: {source}")
    if not (MUSETALK_HOME / "scripts" / "realtime_inference.py").exists():
        raise RuntimeError("MuseTalk runtime is not installed")

    _ensure_layout()
    target = RESULT_ROOT / avatar_id
    if target.exists():
        current = status(avatar_id)
        if current.get("preprocessed"):
            return current
        raise RuntimeError("partial preparation exists; remove it explicitly before retrying")

    temp = Path("/tmp/urv-avatar-preparation") / avatar_id
    if temp.exists():
        shutil.rmtree(temp)
    image_dir = temp / "frames"
    image_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, image_dir / "00000000.png")
    silence = temp / "silence.wav"
    config = temp / "realtime.yaml"
    _silence_wav(silence)
    config.write_text(
        f"{avatar_id}:\n"
        "  preparation: true\n"
        f"  video_path: {image_dir}\n"
        "  bbox_shift: 0\n"
        "  audio_clips:\n"
        f"    calibration: {silence}\n",
        encoding="utf-8",
    )

    command = [
        "python3", "-m", "scripts.realtime_inference",
        "--version", "v15",
        "--inference_config", str(config),
        "--unet_model_path", "models/musetalkV15/unet.pth",
        "--unet_config", "models/musetalkV15/musetalk.json",
        "--fps", "25",
        "--batch_size", "8",
        "--skip_save_images",
    ]
    completed = subprocess.run(
        command,
        cwd=MUSETALK_HOME,
        check=False,
        text=True,
        capture_output=True,
        timeout=int(os.getenv("URV_PREPARE_TIMEOUT_SECONDS", "3600")),
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-4000:] or completed.stdout[-4000:])

    required = ["coords.pkl", "latents.pt", "mask_coords.pkl", "full_imgs", "mask"]
    missing = [name for name in required if not (target / name).exists()]
    if missing:
        raise RuntimeError(f"MuseTalk preparation missing artifacts: {missing}")

    result = {
        "avatar_id": avatar_id,
        "state": "ready",
        "preprocessed": True,
        "renderer": "musetalk-v1.5",
        "fps": 25,
    }
    _status_path(avatar_id).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
