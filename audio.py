import base64
from dataclasses import dataclass
import numpy as np

@dataclass(slots=True)
class AudioChunk:
    sequence: int; timestamp_ms: int; samples: np.ndarray

def decode_chunk(event: dict) -> AudioChunk:
    raw=base64.b64decode(event["audio"])
    return AudioChunk(int(event["sequence"]),int(event["timestamp_ms"]),np.frombuffer(raw,dtype="<i2").copy())
