from abc import ABC, abstractmethod
from pathlib import Path
import importlib.util, os, time
import numpy as np

class BaseAvatarRenderer(ABC):
    @abstractmethod
    def load(self,avatar_dir: Path): ...
    @abstractmethod
    def render(self,audio: np.ndarray,motion: dict) -> list[np.ndarray]: ...

class MuseTalkRenderer(BaseAvatarRenderer):
    """Streaming adapter around a mounted, unmodified MuseTalk checkout.

    Model code and weights are intentionally not vendored. The checkout must expose
    urv_stream_adapter.create_renderer(), which performs cached, in-memory inference.
    """
    def __init__(self): self.engine=None; self.last_ms=0.0; self.fps=0.0
    def load(self,avatar_dir: Path):
        checkout=Path(os.environ.get("MUSETALK_HOME","/opt/MuseTalk"))
        adapter=checkout/"urv_stream_adapter.py"
        if not adapter.exists(): raise RuntimeError(f"MuseTalk streaming adapter missing: {adapter}")
        spec=importlib.util.spec_from_file_location("urv_musetalk",adapter)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.engine=module.create_renderer(avatar_dir=avatar_dir,device=os.getenv("URV_DEVICE","cuda"))
    def render(self,audio: np.ndarray,motion: dict) -> list[np.ndarray]:
        if self.engine is None: raise RuntimeError("renderer not loaded")
        started=time.perf_counter(); frames=self.engine.render_pcm(audio,motion)
        self.last_ms=(time.perf_counter()-started)*1000
        self.fps=len(frames)/(self.last_ms/1000) if self.last_ms else 0
        return frames

def create_renderer(name: str) -> BaseAvatarRenderer:
    if name in {"musetalk","musetalk-v1.5"}: return MuseTalkRenderer()
    raise ValueError(f"Unknown URV avatar renderer: {name}")
