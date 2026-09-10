# URV Avatar Engine

RunPod-ready streaming GPU service. It receives PCM at `/ws/avatar`, loads
preprocessed avatar assets once, and returns timestamped RGBA frames.
It never creates an MP4.

The Docker image installs MuseTalk 1.5 at `/opt/MuseTalk` and adds a thin
`urv_stream_adapter.py` whose
`create_renderer(avatar_dir, device)` returns an object implementing
`render_pcm(samples, motion)`. Persistent preprocessing output belongs under
`/runpod-volume/musetalk-results/v15/avatars/<avatar_id>` for all four registered identities.

Authenticated administration routes are `GET /admin/preparation-status` and
`POST /admin/prepare-avatars`. The POST performs GPU preprocessing and must only
be called after the Network Volume is mounted at `/runpod-volume`.
Behind a RunPod load balancer, use `Authorization: Bearer <RUNPOD_API_KEY>` for
the gateway and `X-URV-Avatar-Token: <URV_AVATAR_TOKEN>` for the application.
The same two headers are required when opening `/ws/avatar` from the LiveKit agent.

```bash
docker build -t urv-avatar-engine:0.1 avatar-worker
docker run --gpus all --rm -p 8000:8000 \
 -e URV_AVATAR_TOKEN=replace-me \
 -e URV_AVATAR_RENDERER=musetalk-v1.5 \
 -v urv-avatar-data:/runpod-volume urv-avatar-engine:0.1
```

Use a normal 24 GB GPU Pod first and expose port 8000 behind TLS. Frame events
report FPS, generation latency and VRAM. MuseTalk's published 30 FPS+ figure is
hardware-specific; measure on the selected Pod. Its 256px face region,
identity-loss and jitter limitations mean sharper teeth, expressive upper-face
motion and production temporal stability remain URV Avatar V2 work.
