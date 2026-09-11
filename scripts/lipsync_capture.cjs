const fs = require("fs");
const WebSocket = require("ws");

const [url, apiKey, avatarToken, pcmPath, rgbaPath, metaPath, avatarId = "professor_arya"] = process.argv.slice(2);
if (![url, apiKey, avatarToken, pcmPath, rgbaPath, metaPath].every(Boolean)) {
  throw new Error("usage: node lipsync_capture.js URL API_KEY AVATAR_TOKEN PCM RGBA META [AVATAR_ID]");
}

const pcm = fs.readFileSync(pcmPath);
const expectedFrames = Math.max(1, Math.round((pcm.length / 2 / 24000) * 25));
const output = fs.createWriteStream(rgbaPath);
const started = Date.now();
let width = 0;
let height = 0;
let frames = 0;
let firstFrameMs = null;
let finished = false;

const ws = new WebSocket(url, {
  headers: {
    Authorization: `Bearer ${apiKey}`,
    "X-URV-Avatar-Token": avatarToken,
  },
  maxPayload: 32 * 1024 * 1024,
});

const timeout = setTimeout(() => finish(new Error(`timeout after ${frames}/${expectedFrames} frames`)), 240000);

function finish(error) {
  if (finished) return;
  finished = true;
  clearTimeout(timeout);
  output.end(() => {
    if (error) {
      console.error(error.message);
      process.exitCode = 1;
      ws.terminate();
      return;
    }
    const metadata = { width, height, frames, fps: 25, first_frame_ms: firstFrameMs };
    fs.writeFileSync(metaPath, JSON.stringify(metadata, null, 2));
    console.log(JSON.stringify(metadata));
    ws.send(JSON.stringify({ type: "stop" }));
    ws.close();
  });
}

ws.on("open", () => ws.send(JSON.stringify({
  type: "start",
  session_id: `lipsync-${Date.now()}`,
  avatar_id: avatarId,
  fps: 25,
})));

ws.on("message", (data) => {
  const event = JSON.parse(data.toString());
  if (event.type === "ready") {
    ws.send(JSON.stringify({
      type: "audio",
      sequence: 0,
      timestamp_ms: 0,
      sample_rate: 24000,
      encoding: "pcm_s16le",
      audio: pcm.toString("base64"),
    }));
    return;
  }
  if (event.type !== "frame") return;
  const rgba = Buffer.from(event.rgba, "base64");
  if (!width) {
    width = Number(event.width);
    height = Number(event.height);
    firstFrameMs = Date.now() - started;
  }
  if (Number(event.width) !== width || Number(event.height) !== height || rgba.length !== width * height * 4) {
    finish(new Error("invalid or inconsistent RGBA frame"));
    return;
  }
  output.write(rgba);
  frames += 1;
  if (frames >= expectedFrames) finish();
});

ws.on("error", finish);
ws.on("close", (code) => {
  if (!finished) finish(new Error(`socket closed with code ${code} after ${frames}/${expectedFrames} frames`));
});
