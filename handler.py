import asyncio, base64, json, os, time
from pathlib import Path
import cv2, torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from audio import decode_chunk
from avatar_engine import AvatarEngine

app=FastAPI(title="URV Avatar Engine",version="0.1.0")
ROOT=Path(os.getenv("URV_AVATAR_ROOT","/models/avatars"))

@app.get("/health")
async def health():
    return {"ok":True,"engine":"urv-avatar","renderer":os.getenv("URV_AVATAR_RENDERER","musetalk-v1.5"),
            "cuda":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

@app.websocket("/ws/avatar")
async def avatar_socket(ws: WebSocket):
    expected=os.getenv("URV_AVATAR_TOKEN","")
    if not expected or ws.headers.get("authorization") != f"Bearer {expected}":
        await ws.close(code=status.WS_1008_POLICY_VIOLATION); return
    await ws.accept(); engine=None; sender=None
    async def send_frames():
        async for seq,ts,rgba in engine.frames():
            height,width=rgba.shape[:2]
            await ws.send_json({"type":"frame","sequence":seq,"timestamp_ms":ts,"width":width,"height":height,
                "rgba":base64.b64encode(rgba.tobytes()).decode("ascii"),"fps":engine.renderer.fps,
                "frame_generation_ms":engine.renderer.last_ms,
                "vram_mb":round(torch.cuda.memory_allocated()/1048576,1) if torch.cuda.is_available() else 0})
    try:
        while True:
            event=json.loads(await ws.receive_text())
            if event.get("type")=="start":
                engine=AvatarEngine(ROOT,os.getenv("URV_AVATAR_RENDERER","musetalk-v1.5"),int(event.get("fps",25)))
                engine.load(event.get("avatar_id","professor_arya")); sender=asyncio.create_task(send_frames())
                await ws.send_json({"type":"ready","session_id":event["session_id"]})
            elif event.get("type")=="audio" and engine:
                chunk=decode_chunk(event)
                await engine.push_audio(chunk.samples,chunk.timestamp_ms)
            elif event.get("type")=="interrupt" and engine: await engine.interrupt()
            elif event.get("type")=="stop" and engine: await engine.queue.put(None)
    except WebSocketDisconnect: pass
    finally:
        if sender: sender.cancel()
