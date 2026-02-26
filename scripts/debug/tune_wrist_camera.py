"""Interactive wrist camera tuner — browser-based slider UI.

Run this script, then open http://localhost:8000 in your browser.
Drag sliders to adjust camera pos/quat/fov and see the rendered image update live.
"""

import base64
import http.server
import io
import json
import os
import sys
import urllib.parse

import mujoco
import numpy as np
from PIL import Image

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from mujoco_manip.env import PickPlaceEnv  # noqa: E402

SCENE_XML = os.path.join(_PROJECT_ROOT, "pick_and_place_scene.xml")
IMAGE_SIZE = 480

# Global state
env = PickPlaceEnv(SCENE_XML, add_wrist_camera=True)
env.reset_to_keyframe("scene_start")
mujoco.mj_forward(env.model, env.data)
renderer = mujoco.Renderer(env.model, IMAGE_SIZE, IMAGE_SIZE)
cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")

HTML = """<!DOCTYPE html>
<html>
<head>
<title>Wrist Camera Tuner</title>
<style>
  body { font-family: monospace; background: #1a1a2e; color: #e0e0e0; margin: 20px; }
  .container { display: flex; gap: 30px; }
  .controls { min-width: 400px; }
  .slider-row { margin: 8px 0; display: flex; align-items: center; gap: 10px; }
  .slider-row label { min-width: 80px; }
  .slider-row input[type=range] { flex: 1; }
  .slider-row .val { min-width: 70px; text-align: right; color: #00ff88; }
  img { border: 2px solid #444; image-rendering: pixelated; }
  h2 { color: #00ff88; }
  h3 { color: #888; margin-top: 20px; }
  #output { background: #0a0a1a; padding: 10px; margin-top: 15px; border: 1px solid #333;
            font-size: 14px; white-space: pre; color: #00ff88; }
  button { background: #00ff88; color: #1a1a2e; border: none; padding: 8px 16px;
           cursor: pointer; font-family: monospace; font-weight: bold; margin: 5px; }
</style>
</head>
<body>
<h2>Wrist Camera Tuner</h2>
<div class="container">
  <div class="controls">
    <h3>Position (hand frame)</h3>
    <div class="slider-row">
      <label>X:</label>
      <input type="range" id="px" min="-0.15" max="0.15" step="0.005" value="0.0">
      <span class="val" id="px_val">0.000</span>
    </div>
    <div class="slider-row">
      <label>Y:</label>
      <input type="range" id="py" min="-0.15" max="0.15" step="0.005" value="0.05">
      <span class="val" id="py_val">0.050</span>
    </div>
    <div class="slider-row">
      <label>Z:</label>
      <input type="range" id="pz" min="-0.15" max="0.15" step="0.005" value="0.0">
      <span class="val" id="pz_val">0.000</span>
    </div>

    <h3>Look direction (tilt from +Z toward axis)</h3>
    <div class="slider-row">
      <label>Tilt X:</label>
      <input type="range" id="tilt_x" min="-60" max="60" step="1" value="30">
      <span class="val" id="tilt_x_val">30&deg;</span>
    </div>
    <div class="slider-row">
      <label>Tilt Y:</label>
      <input type="range" id="tilt_y" min="-60" max="60" step="1" value="0">
      <span class="val" id="tilt_y_val">0&deg;</span>
    </div>

    <h3>Roll &amp; FOV</h3>
    <div class="slider-row">
      <label>Roll:</label>
      <input type="range" id="roll" min="-180" max="180" step="1" value="0">
      <span class="val" id="roll_val">0&deg;</span>
    </div>
    <div class="slider-row">
      <label>FOV Y:</label>
      <input type="range" id="fovy" min="30" max="150" step="1" value="90">
      <span class="val" id="fovy_val">90&deg;</span>
    </div>

    <div>
      <button onclick="resetDefaults()">Reset</button>
      <button onclick="copyOutput()">Copy Output</button>
    </div>

    <div id="output">Loading...</div>
  </div>
  <div>
    <img id="cam_img" width="480" height="480" src="">
  </div>
</div>
<script>
const sliders = ['px','py','pz','tilt_x','tilt_y','roll','fovy'];
let debounceTimer = null;

function getParams() {
  const p = {};
  sliders.forEach(s => { p[s] = parseFloat(document.getElementById(s).value); });
  return p;
}

function updateLabels() {
  const p = getParams();
  document.getElementById('px_val').textContent = p.px.toFixed(3);
  document.getElementById('py_val').textContent = p.py.toFixed(3);
  document.getElementById('pz_val').textContent = p.pz.toFixed(3);
  document.getElementById('tilt_x_val').innerHTML = p.tilt_x.toFixed(0) + '&deg;';
  document.getElementById('tilt_y_val').innerHTML = p.tilt_y.toFixed(0) + '&deg;';
  document.getElementById('roll_val').innerHTML = p.roll.toFixed(0) + '&deg;';
  document.getElementById('fovy_val').innerHTML = p.fovy.toFixed(0) + '&deg;';
}

function fetchImage() {
  const p = getParams();
  const qs = new URLSearchParams(p).toString();
  fetch('/render?' + qs)
    .then(r => r.json())
    .then(data => {
      document.getElementById('cam_img').src = 'data:image/png;base64,' + data.image;
      document.getElementById('output').textContent =
        'cam.pos = [' + data.pos.map(v=>v.toFixed(4)).join(', ') + ']\\n' +
        'cam.quat = [' + data.quat.map(v=>v.toFixed(4)).join(', ') + ']\\n' +
        'cam.fovy = ' + data.fovy.toFixed(1);
    });
}

function onSliderChange() {
  updateLabels();
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(fetchImage, 100);
}

function resetDefaults() {
  document.getElementById('px').value = 0;
  document.getElementById('py').value = 0.05;
  document.getElementById('pz').value = 0;
  document.getElementById('tilt_x').value = 30;
  document.getElementById('tilt_y').value = 0;
  document.getElementById('roll').value = 0;
  document.getElementById('fovy').value = 90;
  onSliderChange();
}

function copyOutput() {
  navigator.clipboard.writeText(document.getElementById('output').textContent);
}

sliders.forEach(s => {
  document.getElementById(s).addEventListener('input', onSliderChange);
});

updateLabels();
fetchImage();
</script>
</body>
</html>
"""


def quat_from_look_up(look, up):
    """Build MuJoCo camera quat (wxyz) from look direction and up vector."""
    look = look / np.linalg.norm(look)
    right = np.cross(look, up)
    right = right / np.linalg.norm(right)
    up_orth = np.cross(right, look)
    R = np.column_stack([right, up_orth, -look])
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w, x, y, z = (
            0.25 / s,
            (R[2, 1] - R[1, 2]) * s,
            (R[0, 2] - R[2, 0]) * s,
            (R[1, 0] - R[0, 1]) * s,
        )
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (
            (R[2, 1] - R[1, 2]) / s,
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
        )
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (
            (R[0, 2] - R[2, 0]) / s,
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s,
        )
    else:
        s = 2.0 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (
            (R[1, 0] - R[0, 1]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s,
        )
    return np.array([w, x, y, z])


def compute_camera(px, py, pz, tilt_x_deg, tilt_y_deg, roll_deg, fovy):
    """Compute camera pos, quat, fovy from intuitive parameters."""
    # Look direction: start at hand +Z, tilt toward +X by tilt_x, toward +Y by tilt_y
    tx = np.radians(tilt_x_deg)
    ty = np.radians(tilt_y_deg)
    look = np.array([np.sin(tx), np.sin(ty), np.cos(tx) * np.cos(ty)])
    look = look / np.linalg.norm(look)

    # Base up: -X in hand frame (perpendicular to default look +Z)
    up_base = np.array([-1.0, 0.0, 0.0])

    # Apply roll around look axis
    roll = np.radians(roll_deg)
    # Rodrigues rotation of up_base around look by roll angle
    up = (
        up_base * np.cos(roll)
        + np.cross(look, up_base) * np.sin(roll)
        + look * np.dot(look, up_base) * (1 - np.cos(roll))
    )

    quat = quat_from_look_up(look, up)
    return np.array([px, py, pz]), quat, fovy


def render_image(px, py, pz, tilt_x, tilt_y, roll, fovy):
    """Render and return PNG bytes + camera params."""
    pos, quat, fovy = compute_camera(px, py, pz, tilt_x, tilt_y, roll, fovy)
    env.model.cam_pos[cam_id] = pos
    env.model.cam_quat[cam_id] = quat
    env.model.cam_fovy[cam_id] = fovy
    mujoco.mj_forward(env.model, env.data)
    renderer.update_scene(env.data, camera="wrist")
    img = renderer.render()
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue(), pos.tolist(), quat.tolist(), float(fovy)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path.startswith("/render"):
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            px = float(params.get("px", [0])[0])
            py = float(params.get("py", [0.05])[0])
            pz = float(params.get("pz", [0])[0])
            tilt_x = float(params.get("tilt_x", [30])[0])
            tilt_y = float(params.get("tilt_y", [0])[0])
            roll = float(params.get("roll", [0])[0])
            fovy = float(params.get("fovy", [90])[0])
            png_bytes, pos, quat, fovy_out = render_image(
                px, py, pz, tilt_x, tilt_y, roll, fovy
            )
            b64 = base64.b64encode(png_bytes).decode()
            result = json.dumps(
                {"image": b64, "pos": pos, "quat": quat, "fovy": fovy_out}
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(result.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress request logs


def main():
    port = 8000
    print(f"Wrist camera tuner running at http://localhost:{port}")
    print("Open in browser, adjust sliders, copy the output values.")
    print("Press Ctrl+C to stop.")
    server = http.server.HTTPServer(("", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
