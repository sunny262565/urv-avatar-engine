"""Verify authenticated PCM-to-RGBA streaming against a deployed avatar worker."""

import argparse
import asyncio
import base64
import json
import math
import os
import struct
import time

from websockets.asyncio.client import connect


async def run(url: str, api_key: str, avatar_token: str, avatar_id: str, timeout: int) -> None:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-URV-Avatar-Token": avatar_token,
    }
    started = time.perf_counter()
    async with connect(
        url,
        additional_headers=headers,
        max_size=32 * 1024 * 1024,
        open_timeout=timeout,
        proxy=None,
        compression=None,
    ) as socket:
        session_id = f"smoke-{int(time.time())}"
        await socket.send(json.dumps({
            "type": "start", "session_id": session_id, "avatar_id": avatar_id, "fps": 25,
        }))
        ready = json.loads(await asyncio.wait_for(socket.recv(), timeout))
        if ready.get("type") != "ready":
            raise RuntimeError(f"expected ready event, received: {ready}")

        # A short audible-frequency tone ensures Whisper produces inference chunks;
        # absolute silence may be trimmed and correctly yield no video frames.
        tone_pcm = b"".join(
            struct.pack("<h", int(2_000 * math.sin(2 * math.pi * 220 * i / 24_000)))
            for i in range(24_000)
        )
        await socket.send(json.dumps({
            "type": "audio",
            "sequence": 0,
            "timestamp_ms": 0,
            "sample_rate": 24_000,
            "encoding": "pcm_s16le",
            "audio": base64.b64encode(tone_pcm).decode("ascii"),
        }))
        while True:
            event = json.loads(await asyncio.wait_for(socket.recv(), timeout))
            if event.get("type") != "frame":
                continue
            rgba = base64.b64decode(event["rgba"], validate=True)
            expected = int(event["width"]) * int(event["height"]) * 4
            if len(rgba) != expected:
                raise RuntimeError(f"invalid RGBA byte count: {len(rgba)} != {expected}")
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            print(json.dumps({
                "ok": True,
                "avatar_id": avatar_id,
                "first_frame_ms": elapsed_ms,
                "width": event["width"],
                "height": event["height"],
                "server_frame_generation_ms": event.get("frame_generation_ms"),
                "server_fps": event.get("fps"),
            }))
            await socket.send(json.dumps({"type": "stop"}))
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("URV_AVATAR_WS_URL"))
    parser.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"))
    parser.add_argument("--avatar-token", default=os.getenv("URV_AVATAR_TOKEN"))
    parser.add_argument("--avatar-id", default="teacher_meera")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    missing = [name for name, value in {
        "--url/URV_AVATAR_WS_URL": args.url,
        "--api-key/RUNPOD_API_KEY": args.api_key,
        "--avatar-token/URV_AVATAR_TOKEN": args.avatar_token,
    }.items() if not value]
    if missing:
        parser.error("missing " + ", ".join(missing))
    asyncio.run(run(args.url, args.api_key, args.avatar_token, args.avatar_id, args.timeout))


if __name__ == "__main__":
    main()
