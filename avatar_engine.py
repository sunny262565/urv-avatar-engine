import asyncio, time
from pathlib import Path
import numpy as np
from motion import MotionEngine
from renderer import create_renderer

class AvatarEngine:
    def __init__(self,root: Path,renderer_name: str,fps: int):
        self.root=root; self.renderer=create_renderer(renderer_name); self.motion=MotionEngine(); self.fps=fps
        self.queue=asyncio.Queue(maxsize=16); self.cancelled=False; self.sequence=0
    def load(self,avatar_id: str):
        avatar_dir=(self.root/avatar_id).resolve()
        if self.root.resolve() not in avatar_dir.parents: raise ValueError("invalid avatar id")
        if not (avatar_dir/"metadata.json").exists(): raise FileNotFoundError("preprocessed avatar not found")
        self.renderer.load(avatar_dir)
    async def push_audio(self,samples: np.ndarray,timestamp_ms: int):
        if self.queue.full(): self.queue.get_nowait()
        await self.queue.put((samples,timestamp_ms))
    async def interrupt(self):
        self.cancelled=True
        while not self.queue.empty(): self.queue.get_nowait()
        self.motion.set("idle")
    async def frames(self):
        while True:
            item=await self.queue.get()
            if item is None: return
            audio,audio_timestamp_ms=item
            self.cancelled=False; self.motion.set("speaking"); started=time.perf_counter()
            for index,frame in enumerate(await asyncio.to_thread(self.renderer.render,audio,self.motion.parameters())):
                if self.cancelled: break
                yield self.sequence,audio_timestamp_ms+round(index*1000/self.fps),frame
                self.sequence+=1
            self.motion.set("idle")
