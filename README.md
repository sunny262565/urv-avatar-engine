# URV Avatar Engine

RunPod-ready streaming GPU service. It receives PCM at `/ws/avatar`, loads
preprocessed Professor Arya assets once, and returns timestamped RGBA frames.
It never creates an MP4.

Mount MuseTalk 1.5 at `/opt/MuseTalk` with a thin `urv_stream_adapter.py` whose
`create_renderer(avatar_dir, device)` returns an object implementing
`render_pcm(samples, motion)`. Preprocess once into
`avatars/professor_arya/{identity,frames,masks,latents}`.

```bash
docker build -t urv-avatar-engine:0.1 avatar-worker
docker run --gpus all --rm -p 8000:8000 \
 -e URV_AVATAR_TOKEN=replace-me \
 -e URV_AVATAR_RENDERER=musetalk-v1.5 \
 -v /runpod-volume/avatars:/models/avatars \
 -v /runpod-volume/MuseTalk:/opt/MuseTalk urv-avatar-engine:0.1
```

Use a normal 24 GB GPU Pod first and expose port 8000 behind TLS. Frame events
report FPS, generation latency and VRAM. MuseTalk's published 30 FPS+ figure is
hardware-specific; measure on the selected Pod. Its 256px face region,
identity-loss and jitter limitations mean sharper teeth, expressive upper-face
motion and production temporal stability remain URV Avatar V2 work.
