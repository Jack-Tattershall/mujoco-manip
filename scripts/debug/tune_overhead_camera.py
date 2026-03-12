"""Interactive overhead camera tuner for the bottle packing scene.

Run this script, then open http://localhost:8001 in your browser.
Drag sliders to adjust camera pos / look-at / fov and see the rendered image update live.
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
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from mujoco_manip.tasks.bottle_packing.env import BottlePackingEnv  # noqa: E402

IMAGE_SIZE = 480

# --- Scene setup ----------------------------------------------------------- #
env = BottlePackingEnv()
env.reset_to_keyframe("scene_start")
env.colorize_bottles()
# Place a few bottles in wells and on conveyor for a representative view
env.setup_scene(num_prepacked=6)
env.load_conveyor(list(range(6, 11)))
# Advance conveyor partway so bottles are visible on the belt
for _ in range(500):
    env.tick_conveyor()
mujoco.mj_forward(env.model, env.data)

renderer = mujoco.Renderer(env.model, IMAGE_SIZE, IMAGE_SIZE)
cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")

# Read current defaults from the model
_DEFAULT_POS = env.model.cam_pos[cam_id].copy()
_DEFAULT_FOVY = float(env.model.cam_fovy[cam_id])

# --- Quaternion helpers ---------------------------------------------------- #


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


def xyaxes_from_quat(q):
    """Convert quaternion (wxyz) to MuJoCo xyaxes (6,) = [x_axis, y_axis]."""
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return np.concatenate([R[:, 0], R[:, 1]])


# --- Compute + render ----------------------------------------------------- #


def compute_camera(px, py, pz, tx, ty, tz, roll_deg, fovy):
    """Compute camera pos/quat from position and look-at target."""
    pos = np.array([px, py, pz])
    target = np.array([tx, ty, tz])
    look = target - pos
    if np.linalg.norm(look) < 1e-6:
        look = np.array([0, 0, -1])
    look = look / np.linalg.norm(look)

    # World up, then apply roll around look axis
    up_base = np.array([0.0, 0.0, 1.0])
    roll = np.radians(roll_deg)
    up = (
        up_base * np.cos(roll)
        + np.cross(look, up_base) * np.sin(roll)
        + look * np.dot(look, up_base) * (1 - np.cos(roll))
    )
    quat = quat_from_look_up(look, up)
    return pos, quat, fovy


def render_image(px, py, pz, tx, ty, tz, roll, fovy):
    pos, quat, fovy = compute_camera(px, py, pz, tx, ty, tz, roll, fovy)
    env.model.cam_pos[cam_id] = pos
    env.model.cam_quat[cam_id] = quat
    env.model.cam_fovy[cam_id] = fovy
    mujoco.mj_forward(env.model, env.data)
    renderer.update_scene(env.data, camera="overhead")
    img = renderer.render()
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    xy = xyaxes_from_quat(quat)
    return buf.getvalue(), pos.tolist(), quat.tolist(), xy.tolist(), float(fovy)


# --- HTML UI --------------------------------------------------------------- #

HTML = (
    """<!DOCTYPE html>
<html>
<head>
<title>Overhead Camera Tuner</title>
<style>
  body { font-family: monospace; background: #1a1a2e; color: #e0e0e0; margin: 20px; }
  .container { display: flex; gap: 30px; }
  .controls { min-width: 420px; }
  .slider-row { margin: 8px 0; display: flex; align-items: center; gap: 10px; }
  .slider-row label { min-width: 90px; }
  .slider-row input[type=range] { flex: 1; }
  .slider-row .val { min-width: 70px; text-align: right; color: #00ff88; }
  img { border: 2px solid #444; image-rendering: pixelated; }
  h2 { color: #00ff88; }
  h3 { color: #888; margin-top: 20px; }
  #output { background: #0a0a1a; padding: 10px; margin-top: 15px; border: 1px solid #333;
            font-size: 13px; white-space: pre; color: #00ff88; min-height: 80px; }
  button { background: #00ff88; color: #1a1a2e; border: none; padding: 8px 16px;
           cursor: pointer; font-family: monospace; font-weight: bold; margin: 5px; }
</style>
</head>
<body>
<h2>Overhead Camera Tuner (Bottle Packing)</h2>
<div class="container">
  <div class="controls">
    <h3>Camera Position (world frame)</h3>
    <div class="slider-row">
      <label>Pos X:</label>
      <input type="range" id="px" min="-0.5" max="0.5" step="0.01" value="DPOS_X">
      <span class="val" id="px_val"></span>
    </div>
    <div class="slider-row">
      <label>Pos Y:</label>
      <input type="range" id="py" min="0.0" max="1.5" step="0.01" value="DPOS_Y">
      <span class="val" id="py_val"></span>
    </div>
    <div class="slider-row">
      <label>Pos Z:</label>
      <input type="range" id="pz" min="0.3" max="3.0" step="0.01" value="DPOS_Z">
      <span class="val" id="pz_val"></span>
    </div>

    <h3>Look-at Target (world frame)</h3>
    <div class="slider-row">
      <label>Target X:</label>
      <input type="range" id="tx" min="-0.5" max="0.5" step="0.01" value="0.10">
      <span class="val" id="tx_val"></span>
    </div>
    <div class="slider-row">
      <label>Target Y:</label>
      <input type="range" id="ty" min="0.0" max="1.0" step="0.01" value="0.45">
      <span class="val" id="ty_val"></span>
    </div>
    <div class="slider-row">
      <label>Target Z:</label>
      <input type="range" id="tz" min="0.0" max="0.5" step="0.01" value="0.28">
      <span class="val" id="tz_val"></span>
    </div>

    <h3>Roll &amp; FOV</h3>
    <div class="slider-row">
      <label>Roll:</label>
      <input type="range" id="roll" min="-180" max="180" step="1" value="0">
      <span class="val" id="roll_val"></span>
    </div>
    <div class="slider-row">
      <label>FOV Y:</label>
      <input type="range" id="fovy" min="20" max="120" step="1" value="DFOVY">
      <span class="val" id="fovy_val"></span>
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
const sliders = ['px','py','pz','tx','ty','tz','roll','fovy'];
const defaults = {
  px: DPOS_X, py: DPOS_Y, pz: DPOS_Z,
  tx: 0.10, ty: 0.45, tz: 0.28,
  roll: 0, fovy: DFOVY
};
let debounceTimer = null;

function getParams() {
  const p = {};
  sliders.forEach(s => { p[s] = parseFloat(document.getElementById(s).value); });
  return p;
}

function updateLabels() {
  const p = getParams();
  ['px','py','pz','tx','ty','tz'].forEach(s => {
    document.getElementById(s + '_val').textContent = p[s].toFixed(2);
  });
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
        '# XML attribute format:\\n' +
        'pos="' + data.pos.map(v=>v.toFixed(4)).join(' ') + '"\\n' +
        'xyaxes="' + data.xyaxes.map(v=>v.toFixed(4)).join(' ') + '"\\n' +
        'fovy="' + data.fovy.toFixed(1) + '"\\n\\n' +
        '# Python format:\\n' +
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
  sliders.forEach(s => { document.getElementById(s).value = defaults[s]; });
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
""".replace("DPOS_X", f"{_DEFAULT_POS[0]:.2f}")
    .replace("DPOS_Y", f"{_DEFAULT_POS[1]:.2f}")
    .replace("DPOS_Z", f"{_DEFAULT_POS[2]:.2f}")
    .replace("DFOVY", f"{_DEFAULT_FOVY:.0f}")
)

# --- HTTP server ----------------------------------------------------------- #


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

            def g(name, default):
                return float(params.get(name, [default])[0])

            px = g("px", _DEFAULT_POS[0])
            py = g("py", _DEFAULT_POS[1])
            pz = g("pz", _DEFAULT_POS[2])
            tx = g("tx", 0.10)
            ty = g("ty", 0.45)
            tz = g("tz", 0.28)
            roll = g("roll", 0)
            fovy = g("fovy", _DEFAULT_FOVY)

            png_bytes, pos, quat, xyaxes, fovy_out = render_image(
                px, py, pz, tx, ty, tz, roll, fovy
            )
            b64 = base64.b64encode(png_bytes).decode()
            result = json.dumps(
                {
                    "image": b64,
                    "pos": pos,
                    "quat": quat,
                    "xyaxes": xyaxes,
                    "fovy": fovy_out,
                }
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(result.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    port = 8001
    print(f"Overhead camera tuner running at http://localhost:{port}")
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
