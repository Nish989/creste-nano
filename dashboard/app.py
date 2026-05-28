import asyncio
import json
import os
import subprocess
import signal
import time
import threading
from pathlib import Path
from aiohttp import web
import aiohttp
import yaml

# ROS2 bridge — optional (only works on Jetson where rclpy is installed)
try:
    import rclpy
    from rclpy.node import Node as RclpyNode
    from sensor_msgs.msg import NavSatFix, CompressedImage
    from std_msgs.msg import Float64, Bool, String
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False

# ── Config ────────────────────────────────────────────────────────────────────
WS_DIR = os.path.expanduser('~/mapless_nav_ws')
DATA_DIR = os.path.expanduser('~/mapless_nav_data')
ROUTE_FILE = os.path.join(DATA_DIR, 'current_route.yaml')
PORT = 8080

# ── State ─────────────────────────────────────────────────────────────────────
state = {
    'mode': 'idle',           # idle | teleop | data_collection | autonomous
    'process': None,          # current ROS2 subprocess
    'log_lines': [],          # last 100 log lines
    'ros2_connected': False,  # True when rclpy bridge is receiving data
    'metrics': {
        'nir': 0.0,
        'interventions': 0,
        'autonomous_meters': 0.0,
        'gps_lat': None,
        'gps_lon': None,
        'gps_sats': 0,
        'gps_fix': -1,
        'speed': 0.0,
        'heading': 0.0,
        'steering': 0.0,
        'throttle': 0.0,
        'autonomous': False,
        'recording': False,
        'online_updates': 0,
    },
    'waypoints': [],
    'ws_clients': set(),
    'latest_jpeg': None,  # latest camera frame as JPEG bytes
}

# ── ROS2 Topic Bridge ────────────────────────────────────────────────────────

def start_ros2_bridge():
    if not HAS_RCLPY:
        return
    try:
        rclpy.init()
    except RuntimeError:
        pass  # already initialized

    class BridgeNode(RclpyNode):
        def __init__(self):
            super().__init__('dashboard_bridge')
            self.create_subscription(NavSatFix, '/gps/fix', self.gps_cb, 10)
            self.create_subscription(Float64, '/gps/speed', self.speed_cb, 10)
            self.create_subscription(Float64, '/gps/course', self.heading_cb, 10)
            self.create_subscription(Float64, '/cmd_steering', self.steer_cb, 10)
            self.create_subscription(Float64, '/cmd_throttle', self.throttle_cb, 10)
            self.create_subscription(Bool, '/autonomous_mode', self.auto_cb, 10)
            self.create_subscription(Bool, '/recording', self.recording_cb, 10)
            self.create_subscription(String, '/intervention_stats', self.stats_cb, 10)
            self.create_subscription(CompressedImage, '/camera/image_raw/compressed', self.cam_cb, 5)
            self._last_push = 0

            # Publishers for phone drive controls
            self.cmd_steer_pub = self.create_publisher(Float64, '/cmd_steering', 10)
            self.cmd_thr_pub = self.create_publisher(Float64, '/cmd_throttle', 10)
            self.cmd_estop_pub = self.create_publisher(Bool, '/estop', 10)
            self.cmd_record_pub = self.create_publisher(Bool, '/record_toggle', 10)
            self.cmd_auto_pub = self.create_publisher(Bool, '/autonomous_mode', 10)
            state['phone_pubs'] = {
                'steer': self.cmd_steer_pub,
                'throttle': self.cmd_thr_pub,
                'estop': self.cmd_estop_pub,
                'record': self.cmd_record_pub,
                'auto': self.cmd_auto_pub,
            }
            state['last_phone_cmd'] = 0

            # Push metrics to WS clients at 2Hz
            self.create_timer(0.5, self.push_metrics)

        def gps_cb(self, msg):
            state['ros2_connected'] = True
            if msg.status.status >= 0:
                state['metrics']['gps_lat'] = msg.latitude
                state['metrics']['gps_lon'] = msg.longitude
                state['metrics']['gps_fix'] = msg.status.status

        def speed_cb(self, msg):
            state['metrics']['speed'] = msg.data

        def heading_cb(self, msg):
            state['metrics']['heading'] = msg.data

        def steer_cb(self, msg):
            state['metrics']['steering'] = msg.data

        def throttle_cb(self, msg):
            state['metrics']['throttle'] = msg.data

        def auto_cb(self, msg):
            state['metrics']['autonomous'] = msg.data

        def recording_cb(self, msg):
            state['metrics']['recording'] = msg.data

        def cam_cb(self, msg):
            state['latest_jpeg'] = bytes(msg.data)

        def stats_cb(self, msg):
            try:
                d = json.loads(msg.data)
                state['metrics']['interventions'] = d.get('interventions', 0)
                state['metrics']['autonomous_meters'] = d.get('meters', 0.0)
                state['metrics']['nir'] = d.get('nir', 0.0)
                state['metrics']['online_updates'] = d.get('updates', 0)
            except Exception:
                pass

        def push_metrics(self):
            if loop is None:
                return
            # Phone control watchdog — stop car if phone disconnects
            lpc = state.get('last_phone_cmd', 0)
            if lpc > 0 and (time.time() - lpc) > 0.5:
                pubs = state.get('phone_pubs')
                if pubs:
                    pubs['steer'].publish(Float64(data=0.0))
                    pubs['throttle'].publish(Float64(data=0.0))
                state['last_phone_cmd'] = 0
            asyncio.run_coroutine_threadsafe(
                broadcast({
                    'type': 'metrics',
                    'data': state['metrics'],
                    'ros2': state['ros2_connected'],
                    'mode': state['mode'],
                }),
                loop,
            )

    node = BridgeNode()
    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        node.destroy_node()


# ── ROS2 Process Management ────────────────────────────────────────────────────

LAUNCH_COMMANDS = {
    'teleop': f'bash -c "source {WS_DIR}/install/setup.bash && ros2 launch mapless_nav teleop_launch.py"',
    'data_collection': f'bash -c "source {WS_DIR}/install/setup.bash && ros2 launch mapless_nav data_collection_launch.py"',
    'autonomous': f'bash -c "source {WS_DIR}/install/setup.bash && ros2 launch mapless_nav autonomous_launch.py route_file:={ROUTE_FILE}"',
}

def start_mode(mode):
    stop_current()
    time.sleep(2)  # let camera device release before restarting
    if mode not in LAUNCH_COMMANDS:
        return

    cmd = LAUNCH_COMMANDS[mode]
    proc = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        text=True,
        bufsize=1,
    )
    state['process'] = proc
    state['mode'] = mode
    add_log(f'[DASHBOARD] Started {mode} mode (PID {proc.pid})')

    # Stream logs in background thread
    def stream_logs():
        for line in proc.stdout:
            add_log(line.rstrip())
        add_log(f'[DASHBOARD] {mode} process exited')
        state['mode'] = 'idle'
        state['process'] = None

    t = threading.Thread(target=stream_logs, daemon=True)
    t.start()

def stop_current():
    proc = state.get('process')
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=3)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        add_log('[DASHBOARD] Stopped current mode')
    state['process'] = None
    state['mode'] = 'idle'

def add_log(line):
    state['log_lines'].append(line)
    if len(state['log_lines']) > 200:
        state['log_lines'] = state['log_lines'][-200:]
    # Broadcast to all websocket clients
    asyncio.run_coroutine_threadsafe(
        broadcast({'type': 'log', 'line': line}),
        loop
    )

# ── Waypoint Management ────────────────────────────────────────────────────────

def load_waypoints():
    if os.path.exists(ROUTE_FILE):
        with open(ROUTE_FILE) as f:
            data = yaml.safe_load(f)
            state['waypoints'] = data.get('waypoints', [])

def save_waypoints():
    os.makedirs(DATA_DIR, exist_ok=True)
    data = {'waypoints': state['waypoints']}
    with open(ROUTE_FILE, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

# ── WebSocket Broadcast ────────────────────────────────────────────────────────

async def broadcast(msg):
    dead = set()
    for ws in state['ws_clients']:
        try:
            await ws.send_str(json.dumps(msg))
        except Exception:
            dead.add(ws)
    state['ws_clients'] -= dead

# ── Metrics polling (reads intervention stats from ROS2 topic via file) ────────

def poll_metrics():
    interventions_log = os.path.join(DATA_DIR, 'interventions', 'interventions.jsonl')
    last_size = 0
    while True:
        try:
            if os.path.exists(interventions_log):
                size = os.path.getsize(interventions_log)
                if size != last_size:
                    last_size = size
                    with open(interventions_log) as f:
                        lines = f.readlines()
                    if lines:
                        last = json.loads(lines[-1])
                        state['metrics']['interventions'] = last.get('intervention_n', 0)
                        state['metrics']['autonomous_meters'] = last.get('autonomous_meters', 0.0)
                        m = state['metrics']['autonomous_meters']
                        i = state['metrics']['interventions']
                        state['metrics']['nir'] = (i / max(m, 1)) * 100
                        asyncio.run_coroutine_threadsafe(
                            broadcast({'type': 'metrics', 'data': state['metrics']}),
                            loop
                        )
        except Exception:
            pass
        time.sleep(1.0)

# ── HTTP Handlers ──────────────────────────────────────────────────────────────

async def handle_index(request):
    return web.Response(text=HTML, content_type='text/html')

async def handle_stream(request):
    resp = web.StreamResponse()
    resp.content_type = 'multipart/x-mixed-replace; boundary=frame'
    await resp.prepare(request)
    while True:
        jpeg = state.get('latest_jpeg')
        if jpeg:
            await resp.write(
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
            )
        await asyncio.sleep(0.1)  # ~10fps

async def handle_ws(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    state['ws_clients'].add(ws)

    # Send initial state
    await ws.send_str(json.dumps({
        'type': 'init',
        'mode': state['mode'],
        'metrics': state['metrics'],
        'ros2': state['ros2_connected'],
        'waypoints': state['waypoints'],
        'logs': state['log_lines'][-50:],
    }))

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            data = json.loads(msg.data)
            await handle_ws_message(ws, data)
        elif msg.type == aiohttp.WSMsgType.ERROR:
            break

    state['ws_clients'].discard(ws)
    return ws

async def handle_ws_message(ws, data):
    action = data.get('action')

    if action == 'start':
        mode = data.get('mode')
        threading.Thread(target=start_mode, args=(mode,), daemon=True).start()
        await broadcast({'type': 'mode', 'mode': mode})

    elif action == 'stop':
        threading.Thread(target=stop_current, daemon=True).start()
        await broadcast({'type': 'mode', 'mode': 'idle'})

    elif action == 'estop':
        threading.Thread(target=stop_current, daemon=True).start()
        # Also publish estop to ROS2
        subprocess.Popen(
            f'bash -c "source {WS_DIR}/install/setup.bash && '
            f'ros2 topic pub --once /estop std_msgs/msg/Bool data:\ true"',
            shell=True
        )
        await broadcast({'type': 'mode', 'mode': 'idle'})
        await broadcast({'type': 'log', 'line': '[DASHBOARD] ⚠️ E-STOP TRIGGERED'})

    elif action == 'add_waypoint':
        wp = {'lat': data['lat'], 'lon': data['lon']}
        state['waypoints'].append(wp)
        save_waypoints()
        await broadcast({'type': 'waypoints', 'waypoints': state['waypoints']})

    elif action == 'clear_waypoints':
        state['waypoints'] = []
        save_waypoints()
        await broadcast({'type': 'waypoints', 'waypoints': []})

    elif action == 'remove_waypoint':
        idx = data.get('idx', -1)
        if 0 <= idx < len(state['waypoints']):
            state['waypoints'].pop(idx)
            save_waypoints()
            await broadcast({'type': 'waypoints', 'waypoints': state['waypoints']})

    elif action == 'phone_control':
        pubs = state.get('phone_pubs')
        if pubs:
            s = max(-1.0, min(1.0, float(data.get('steering', 0))))
            t = max(-1.0, min(1.0, float(data.get('throttle', 0))))
            pubs['steer'].publish(Float64(data=s))
            pubs['throttle'].publish(Float64(data=t))
            state['last_phone_cmd'] = time.time()

    elif action == 'phone_record':
        pubs = state.get('phone_pubs')
        if pubs:
            pubs['record'].publish(Bool(data=True))

    elif action == 'phone_auto':
        pubs = state.get('phone_pubs')
        if pubs:
            auto = not state['metrics'].get('autonomous', False)
            pubs['auto'].publish(Bool(data=auto))

# ── HTML / CSS / JS (single file dashboard) ───────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CREStE-Nano Dashboard</title>
<script>
  // On hotspot (10.42.0.x) skip Leaflet — no internet. On home WiFi, load it.
  window.L = null;
  if (location.hostname !== '10.42.0.1') {
    document.write('<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>');
    document.write('<scr'+'ipt src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"><\/scr'+'ipt>');
  }
</script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0a0a0f;
    color: #e0e0e0;
    min-height: 100vh;
  }

  /* Header */
  .header {
    background: #111118;
    border-bottom: 1px solid #222;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .header h1 {
    font-size: 18px;
    font-weight: 700;
    color: #76c7ff;
    letter-spacing: 0.5px;
  }
  .header h1 span { color: #fff; }
  .status-pill {
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .status-idle    { background: #1a1a2e; color: #666; border: 1px solid #333; }
  .status-teleop  { background: #1a2e1a; color: #4caf50; border: 1px solid #4caf50; }
  .status-data_collection { background: #2e2a1a; color: #ffc107; border: 1px solid #ffc107; }
  .status-autonomous { background: #1a1e2e; color: #76c7ff; border: 1px solid #76c7ff; }

  /* Layout */
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: auto auto auto;
    gap: 12px;
    padding: 16px;
    max-width: 1200px;
    margin: 0 auto;
  }
  @media (max-width: 700px) {
    .grid { grid-template-columns: 1fr; }
    .span2 { grid-column: span 1 !important; }
  }
  .span2 { grid-column: span 2; }

  /* Cards */
  .card {
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 16px;
  }
  .card h2 {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #555;
    margin-bottom: 14px;
  }

  /* Control buttons */
  .btn-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }
  .btn {
    padding: 14px;
    border-radius: 10px;
    border: none;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
  }
  .btn:hover { transform: translateY(-1px); filter: brightness(1.1); }
  .btn:active { transform: translateY(0); }
  .btn .icon { font-size: 22px; }
  .btn-teleop    { background: #1a2e1a; color: #4caf50; border: 1px solid #4caf50; }
  .btn-data      { background: #2e2a1a; color: #ffc107; border: 1px solid #ffc107; }
  .btn-auto      { background: #1a1e2e; color: #76c7ff; border: 1px solid #76c7ff; }
  .btn-stop      { background: #1e1a1a; color: #888; border: 1px solid #333; }
  .btn-estop     {
    grid-column: span 2;
    background: #ff3b30;
    color: #fff;
    font-size: 16px;
    padding: 16px;
    border-radius: 10px;
    border: none;
    cursor: pointer;
    font-weight: 800;
    letter-spacing: 1px;
    transition: all 0.15s;
  }
  .btn-estop:hover { background: #ff6b6b; }
  .btn.active { filter: brightness(1.3); box-shadow: 0 0 12px currentColor; }

  /* Metrics */
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
  }
  .metric {
    background: #0d0d15;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
  }
  .metric .value {
    font-size: 24px;
    font-weight: 700;
    color: #76c7ff;
    line-height: 1;
  }
  .metric .label {
    font-size: 10px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
  }
  .metric.good .value { color: #4caf50; }
  .metric.warn .value { color: #ffc107; }

  /* Map */
  #map {
    height: 280px;
    border-radius: 8px;
    overflow: hidden;
  }
  .map-controls {
    display: flex;
    gap: 8px;
    margin-top: 10px;
  }
  .map-btn {
    flex: 1;
    padding: 8px;
    background: #1a1a2e;
    border: 1px solid #333;
    border-radius: 8px;
    color: #aaa;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
  }
  .map-btn:hover { border-color: #76c7ff; color: #76c7ff; }
  .waypoint-list {
    margin-top: 10px;
    max-height: 80px;
    overflow-y: auto;
  }
  .waypoint-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 8px;
    background: #0d0d15;
    border-radius: 6px;
    margin-bottom: 4px;
    font-size: 11px;
    color: #888;
  }
  .waypoint-item button {
    background: none;
    border: none;
    color: #ff3b30;
    cursor: pointer;
    font-size: 14px;
  }

  /* Camera */
  .camera-feed {
    background: #000;
    border-radius: 8px;
    aspect-ratio: 16/9;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #333;
    font-size: 13px;
    overflow: hidden;
  }
  .camera-feed img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 8px;
  }

  /* Terminal */
  .terminal {
    background: #050508;
    border-radius: 8px;
    padding: 12px;
    height: 200px;
    overflow-y: auto;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 11px;
    line-height: 1.6;
    color: #4caf50;
  }
  .terminal .log-warn  { color: #ffc107; }
  .terminal .log-error { color: #ff3b30; }
  .terminal .log-info  { color: #76c7ff; }
  .terminal .log-dash  { color: #888; font-style: italic; }

  /* Connection indicator */
  .conn-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #ff3b30;
    display: inline-block;
    margin-right: 6px;
  }
  .conn-dot.connected { background: #4caf50; animation: pulse 2s infinite; }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  /* Phone drive overlay — fixed at bottom when driving */
  .drive-overlay {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    height: 200px;
    background: rgba(10,10,15,0.97);
    border-top: 2px solid #76c7ff;
    display: none;
    align-items: center;
    justify-content: space-around;
    z-index: 99;
    padding: 10px 20px;
  }
  .drive-overlay.active { display: flex; }
  .drive-joy-wrap { display: flex; flex-direction: column; align-items: center; gap: 4px; }
  .joy-label { font-size: 9px; color: #444; text-transform: uppercase; letter-spacing: 2px; }
  .drive-info { display: flex; flex-direction: column; gap: 10px; align-items: center; }
  .drive-speed { font-size: 28px; font-weight: 700; color: #76c7ff; font-variant-numeric: tabular-nums; }
  .drive-steer-val { font-size: 12px; color: #555; font-variant-numeric: tabular-nums; }
  .drive-estop {
    background: #ff3b30; color: white; border: none; border-radius: 50%;
    width: 80px; height: 80px; font-size: 13px; font-weight: 800; cursor: pointer;
    -webkit-tap-highlight-color: transparent; touch-action: manipulation;
  }
  .drive-estop:active { background: #cc0000; transform: scale(0.95); }
  .drive-toggles { display: flex; gap: 8px; }
  .drive-toggle {
    padding: 8px 16px; background: #1a1a2e; border: 1px solid #333;
    border-radius: 8px; color: #888; font-size: 11px; font-weight: 600; cursor: pointer;
    touch-action: manipulation;
  }
  .drive-toggle.on { border-color: #4caf50; color: #4caf50; background: #1a2e1a; }
</style>
</head>
<body>

<div class="header">
  <h1>CREStE<span>-Nano</span> 🤖</h1>
  <div style="display:flex;align-items:center;gap:10px">
    <span><span class="conn-dot" id="connDot"></span><span id="connText" style="font-size:11px;color:#555">WS</span></span>
    <span><span class="conn-dot" id="ros2Dot"></span><span id="ros2Text" style="font-size:11px;color:#555">ROS2</span></span>
    <span class="status-pill status-idle" id="statusPill">IDLE</span>
  </div>
</div>

<div class="grid">

  <!-- Control -->
  <div class="card">
    <h2>🎮 Control</h2>
    <div class="btn-grid">
      <button class="btn btn-teleop" onclick="startMode('teleop')">
        <span class="icon">🕹️</span>Teleop
      </button>
      <button class="btn btn-data" onclick="startMode('data_collection')">
        <span class="icon">⏺️</span>Record
      </button>
      <button class="btn btn-auto" onclick="startMode('autonomous')">
        <span class="icon">🤖</span>Autonomous
      </button>
      <button class="btn btn-stop" onclick="stopMode()">
        <span class="icon">⏹️</span>Stop
      </button>
      <button class="btn-estop" onclick="estop()">⚠️ EMERGENCY STOP</button>
    </div>
  </div>

  <!-- Metrics -->
  <div class="card">
    <h2>📊 Live Metrics</h2>
    <div class="metrics-grid">
      <div class="metric" id="m-nir">
        <div class="value" id="val-nir">—</div>
        <div class="label">NIR/100m</div>
      </div>
      <div class="metric">
        <div class="value" id="val-speed">0.0</div>
        <div class="label">Speed m/s</div>
      </div>
      <div class="metric">
        <div class="value" id="val-heading">—</div>
        <div class="label">Heading °</div>
      </div>
      <div class="metric">
        <div class="value" id="val-steering">0.0</div>
        <div class="label">Steering</div>
      </div>
      <div class="metric">
        <div class="value" id="val-interventions">0</div>
        <div class="label">Interventions</div>
      </div>
      <div class="metric good">
        <div class="value" id="val-meters">0</div>
        <div class="label">Auto Meters</div>
      </div>
    </div>
  </div>

  <!-- Map -->
  <div class="card span2">
    <h2>📍 GPS Route Planner — tap map to add waypoints</h2>
    <div id="map"></div>
    <div id="gpsCoords" style="padding:10px;text-align:center;font-size:14px;color:#76c7ff;display:none">
      <span id="gpsText">Waiting for GPS fix...</span>
    </div>
    <div class="map-controls">
      <button class="map-btn" onclick="clearWaypoints()">🗑️ Clear Waypoints</button>
      <button class="map-btn" onclick="centerOnCar()">🎯 Center on Car</button>
      <span style="flex:2;padding:8px;font-size:11px;color:#555;text-align:right" id="wpCount">0 waypoints</span>
    </div>
    <div class="waypoint-list" id="wpList"></div>
  </div>

  <!-- Camera -->
  <div class="card">
    <h2>📷 Camera Feed</h2>
    <div class="camera-feed">
      <img id="cameraImg"
           src=""
           onerror="this.style.display='none';document.getElementById('camOffline').style.display='flex'"
           style="display:block"/>
      <div id="camOffline" style="display:none;flex-direction:column;align-items:center;gap:8px;color:#333">
        <span style="font-size:32px">📷</span>
        <span>Camera offline</span>
        <span style="font-size:10px">Start a mode to activate</span>
      </div>
    </div>
  </div>

  <!-- Terminal -->
  <div class="card">
    <h2>🖥️ ROS2 Log</h2>
    <div class="terminal" id="terminal"></div>
  </div>

</div>

<!-- Phone Drive Controls — fixed at bottom when a mode is active -->
<div class="drive-overlay" id="driveOverlay">
  <div class="drive-joy-wrap">
    <canvas id="joystick" width="160" height="160"></canvas>
    <div class="joy-label">DRAG TO DRIVE</div>
  </div>
  <div class="drive-info">
    <div class="drive-speed" id="driveSpeed">0.0 m/s</div>
    <div class="drive-steer-val" id="driveSteer">STR 0.00</div>
    <button class="drive-estop" id="driveEstop">E-STOP</button>
    <div class="drive-toggles">
      <button class="drive-toggle" id="drvRec" onclick="toggleRecord()">⏺ REC</button>
      <button class="drive-toggle" id="drvAuto" onclick="toggleAuto()">🤖 AUTO</button>
    </div>
  </div>
</div>

<script>
// ── WebSocket ─────────────────────────────────────────────────────────────────
let ws;
let currentMode = 'idle';
let carMarker = null;
let waypointMarkers = [];
let waypoints = [];
let addingWaypoints = false;

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);

  ws.onopen = () => {
    document.getElementById('connDot').classList.add('connected');
    document.getElementById('connText').textContent = 'Connected';
  };

  ws.onclose = () => {
    document.getElementById('connDot').classList.remove('connected');
    document.getElementById('connText').textContent = 'Reconnecting...';
    setTimeout(connect, 2000);
  };

  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);

    if (msg.type === 'init') {
      setMode(msg.mode);
      updateMetrics(msg.metrics);
      setRos2(msg.ros2);
      setWaypoints(msg.waypoints || []);
      msg.logs.forEach(line => appendLog(line));
    }
    else if (msg.type === 'log') {
      appendLog(msg.line);
    }
    else if (msg.type === 'mode') {
      setMode(msg.mode);
    }
    else if (msg.type === 'metrics') {
      updateMetrics(msg.data);
      if (msg.ros2 !== undefined) setRos2(msg.ros2);
      if (msg.mode) setMode(msg.mode);
    }
    else if (msg.type === 'waypoints') {
      setWaypoints(msg.waypoints);
    }
  };
}

// ── Controls ──────────────────────────────────────────────────────────────────
function send(data) {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify(data));
}

function startMode(mode) {
  send({ action: 'start', mode });
}

function stopMode() {
  send({ action: 'stop' });
}

function estop() {
  if (confirm('Trigger Emergency Stop?')) {
    send({ action: 'estop' });
  }
}

// ── Mode UI ───────────────────────────────────────────────────────────────────
function setMode(mode) {
  currentMode = mode;
  var pill = document.getElementById('statusPill');
  pill.className = 'status-pill status-' + mode;
  pill.textContent = mode.replace('_', ' ').toUpperCase();

  // Highlight active button
  document.querySelectorAll('.btn').forEach(function(b) { b.classList.remove('active'); });
  if (mode !== 'idle') {
    document.querySelectorAll('.btn').forEach(function(b) {
      if (b.textContent.toLowerCase().indexOf(mode.replace('_','')) >= 0) {
        b.classList.add('active');
      }
    });
  }

  // Show/hide phone drive overlay
  var overlay = document.getElementById('driveOverlay');
  if (mode !== 'idle') {
    overlay.classList.add('active');
    document.body.style.paddingBottom = '210px';
  } else {
    overlay.classList.remove('active');
    document.body.style.paddingBottom = '0';
  }
}

// ── ROS2 Status ──────────────────────────────────────────────────────────────
function setRos2(connected) {
  const dot = document.getElementById('ros2Dot');
  if (connected) {
    dot.classList.add('connected');
  } else {
    dot.classList.remove('connected');
  }
}

// ── Metrics ───────────────────────────────────────────────────────────────────
function updateMetrics(m) {
  document.getElementById('val-nir').textContent =
    m.nir !== undefined ? m.nir.toFixed(3) : '—';
  document.getElementById('val-interventions').textContent = m.interventions || 0;
  document.getElementById('val-meters').textContent =
    m.autonomous_meters ? Math.round(m.autonomous_meters) : 0;
  document.getElementById('val-speed').textContent =
    m.speed !== undefined ? m.speed.toFixed(1) : '0.0';
  document.getElementById('val-heading').textContent =
    m.heading !== undefined ? Math.round(m.heading) + '°' : '—';
  document.getElementById('val-steering').textContent =
    m.steering !== undefined ? m.steering.toFixed(2) : '0.0';

  // Update car position on map + show GPS coords when map unavailable
  if (m.gps_lat && m.gps_lon) {
    updateCarPosition(m.gps_lat, m.gps_lon);
    if (!map) {
      var el = document.getElementById('gpsCoords');
      el.style.display = 'block';
      document.getElementById('gpsText').textContent =
        '📍 ' + m.gps_lat.toFixed(6) + ', ' + m.gps_lon.toFixed(6) +
        '  |  🧭 ' + Math.round(m.heading || 0) + '°' +
        '  |  🏎️ ' + (m.speed || 0).toFixed(1) + ' m/s';
    }
  }

  // Color NIR
  var nirEl = document.getElementById('m-nir');
  var nir = m.nir || 0;
  nirEl.className = 'metric ' + (nir < 1 ? 'good' : nir < 5 ? 'warn' : '');

  // Update drive overlay readouts
  var ds = document.getElementById('driveSpeed');
  if (ds) ds.textContent = (m.speed || 0).toFixed(1) + ' m/s';

  // Update REC/AUTO button states
  var recBtn = document.getElementById('drvRec');
  var autoBtn = document.getElementById('drvAuto');
  if (recBtn) {
    if (m.recording) { recBtn.classList.add('on'); } else { recBtn.classList.remove('on'); }
  }
  if (autoBtn) {
    if (m.autonomous) { autoBtn.classList.add('on'); } else { autoBtn.classList.remove('on'); }
  }
}

// ── Terminal ──────────────────────────────────────────────────────────────────
function appendLog(line) {
  const term = document.getElementById('terminal');
  const div = document.createElement('div');

  let cls = '';
  if (line.includes('[WARN]') || line.includes('warn'))  cls = 'log-warn';
  else if (line.includes('[ERROR]') || line.includes('error')) cls = 'log-error';
  else if (line.includes('[DASHBOARD]')) cls = 'log-dash';
  else if (line.includes('[INFO]'))  cls = 'log-info';

  if (cls) div.className = cls;
  div.textContent = line;
  term.appendChild(div);

  // Keep last 200 lines
  while (term.children.length > 200) term.removeChild(term.firstChild);
  term.scrollTop = term.scrollHeight;
}

// ── Map ───────────────────────────────────────────────────────────────────────
let map = null;
let carIcon = null;

function initMap() {
  if (typeof L === "undefined" || L === null) {
    document.getElementById("map").innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#555;font-size:13px">Map unavailable offline — GPS in metrics</div>';
    return;
  }
  map = L.map("map").setView([30.5083, -97.6789], 17);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "OpenStreetMap",
    maxZoom: 19,
  }).addTo(map);
  carIcon = L.divIcon({
    html: '<div style="font-size:24px;transform:translate(-50%,-50%)">🚗</div>',
    iconSize: [0,0],
    className: '',
  });
  map.on('click', (e) => {
    const { lat, lng } = e.latlng;
    send({ action: 'add_waypoint', lat, lon: lng });
  });
}

// Init map after page loads (Leaflet loaded synchronously on home WiFi, skipped on hotspot)
window.addEventListener('load', initMap);

function updateCarPosition(lat, lon) {
  if (!map || !carIcon) return;
  if (!carMarker) {
    carMarker = L.marker([lat, lon], { icon: carIcon }).addTo(map);
  } else {
    carMarker.setLatLng([lat, lon]);
  }
}

function setWaypoints(wps) {
  waypoints = wps;

  // Remove old markers
  waypointMarkers.forEach(m => { if (map) map.removeLayer(m); });
  waypointMarkers = [];

  wps.forEach((wp, i) => {
    if (!map) return;
    const icon = L.divIcon({
      html: `<div style="background:#76c7ff;color:#000;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:11px;transform:translate(-50%,-50%)">${i+1}</div>`,
      iconSize: [0,0],
      className: '',
    });
    const marker = L.marker([wp.lat, wp.lon], { icon });
    marker.addTo(map);
    waypointMarkers.push(marker);
  });

  // Update list
  const list = document.getElementById('wpList');
  list.innerHTML = '';
  wps.forEach((wp, i) => {
    const div = document.createElement('div');
    div.className = 'waypoint-item';
    div.innerHTML = `
      <span>#${i+1} — ${wp.lat.toFixed(6)}, ${wp.lon.toFixed(6)}</span>
      <button onclick="removeWaypoint(${i})">×</button>
    `;
    list.appendChild(div);
  });

  document.getElementById('wpCount').textContent = `${wps.length} waypoint${wps.length !== 1 ? 's' : ''}`;
}

function clearWaypoints() {
  if (confirm('Clear all waypoints?')) send({ action: 'clear_waypoints' });
}

function removeWaypoint(idx) {
  send({ action: 'remove_waypoint', idx });
}

function centerOnCar() {
  if (carMarker && map) map.setView(carMarker.getLatLng(), 18);
}

// ── Phone Drive Joystick ─────────────────────────────────────────────────────
var joyCanvas, joyCtx;
var joyX = 0, joyY = 0, joyActive = false;
var JR = 60, TR = 22;  // joystick radius, thumb radius

function initJoystick() {
  joyCanvas = document.getElementById('joystick');
  if (!joyCanvas) return;
  joyCtx = joyCanvas.getContext('2d');

  // Touch events
  joyCanvas.addEventListener('touchstart', function(e) {
    e.preventDefault(); joyActive = true; joyFromTouch(e.touches[0]);
  }, {passive: false});
  joyCanvas.addEventListener('touchmove', function(e) {
    e.preventDefault(); joyFromTouch(e.touches[0]);
  }, {passive: false});
  joyCanvas.addEventListener('touchend', function(e) {
    e.preventDefault(); joyActive = false; joyX = 0; joyY = 0; drawJoy();
  }, {passive: false});

  // Mouse fallback (desktop testing)
  var mouseDown = false;
  joyCanvas.addEventListener('mousedown', function(e) { mouseDown = true; joyActive = true; joyFromMouse(e); });
  window.addEventListener('mousemove', function(e) { if (mouseDown) joyFromMouse(e); });
  window.addEventListener('mouseup', function() {
    if (mouseDown) { mouseDown = false; joyActive = false; joyX = 0; joyY = 0; drawJoy(); }
  });

  // E-STOP button — immediate, no confirm
  var estopBtn = document.getElementById('driveEstop');
  estopBtn.addEventListener('touchstart', function(e) { e.preventDefault(); estopNow(); }, {passive: false});
  estopBtn.addEventListener('click', function(e) { estopNow(); });

  drawJoy();

  // Send phone commands at 20Hz when joystick active
  setInterval(function() {
    if (joyX !== 0 || joyY !== 0) {
      send({action: 'phone_control', steering: Math.round(joyX * 100) / 100, throttle: Math.round(joyY * 100) / 100});
      var stEl = document.getElementById('driveSteer');
      if (stEl) stEl.textContent = 'STR ' + joyX.toFixed(2) + '  THR ' + joyY.toFixed(2);
    }
  }, 50);
}

function joyFromTouch(touch) {
  var rect = joyCanvas.getBoundingClientRect();
  var cx = rect.width / 2, cy = rect.height / 2;
  var dx = (touch.clientX - rect.left - cx) / JR;
  var dy = -(touch.clientY - rect.top - cy) / JR;
  applyJoy(dx, dy);
}

function joyFromMouse(e) {
  var rect = joyCanvas.getBoundingClientRect();
  var cx = rect.width / 2, cy = rect.height / 2;
  var dx = (e.clientX - rect.left - cx) / JR;
  var dy = -(e.clientY - rect.top - cy) / JR;
  applyJoy(dx, dy);
}

function applyJoy(dx, dy) {
  var dist = Math.sqrt(dx * dx + dy * dy);
  if (dist > 1) { dx /= dist; dy /= dist; }
  if (Math.abs(dx) < 0.12) dx = 0;
  if (Math.abs(dy) < 0.12) dy = 0;
  joyX = dx; joyY = dy;
  drawJoy();
}

function drawJoy() {
  var w = joyCanvas.width, h = joyCanvas.height;
  var cx = w / 2, cy = h / 2;
  joyCtx.clearRect(0, 0, w, h);

  // Outer ring
  joyCtx.beginPath();
  joyCtx.arc(cx, cy, JR, 0, Math.PI * 2);
  joyCtx.strokeStyle = '#333';
  joyCtx.lineWidth = 2;
  joyCtx.stroke();

  // Crosshair
  joyCtx.beginPath();
  joyCtx.moveTo(cx - JR, cy); joyCtx.lineTo(cx + JR, cy);
  joyCtx.moveTo(cx, cy - JR); joyCtx.lineTo(cx, cy + JR);
  joyCtx.strokeStyle = '#1a1a2e';
  joyCtx.lineWidth = 1;
  joyCtx.stroke();

  // Labels
  joyCtx.font = '9px sans-serif';
  joyCtx.fillStyle = '#333';
  joyCtx.textAlign = 'center';
  joyCtx.fillText('FWD', cx, cy - JR - 4);
  joyCtx.fillText('REV', cx, cy + JR + 12);
  joyCtx.fillText('L', cx - JR - 8, cy + 3);
  joyCtx.fillText('R', cx + JR + 8, cy + 3);

  // Thumb
  var tx = cx + joyX * JR;
  var ty = cy - joyY * JR;
  joyCtx.beginPath();
  joyCtx.arc(tx, ty, TR, 0, Math.PI * 2);
  joyCtx.fillStyle = joyActive ? '#76c7ff' : '#444';
  joyCtx.fill();
  joyCtx.strokeStyle = joyActive ? '#5ab0e8' : '#333';
  joyCtx.lineWidth = 2;
  joyCtx.stroke();
}

function estopNow() {
  send({action: 'estop'});
  joyActive = false; joyX = 0; joyY = 0;
  if (joyCtx) drawJoy();
}

function toggleRecord() { send({action: 'phone_record'}); }
function toggleAuto() { send({action: 'phone_auto'}); }

window.addEventListener('load', initJoystick);

// ── Init ──────────────────────────────────────────────────────────────────────
document.getElementById('cameraImg').src = 'http://' + location.host + '/stream';
connect();
</script>
</body>
</html>
"""

# ── App Setup ──────────────────────────────────────────────────────────────────

loop = None

async def create_app():
    app = web.Application()
    app.router.add_get('/', handle_index)
    app.router.add_get('/stream', handle_stream)
    app.router.add_get('/ws', handle_ws)
    return app

async def main():
    global loop
    loop = asyncio.get_event_loop()

    load_waypoints()

    # Start ROS2 topic bridge (subscribes to GPS, speed, steering, etc.)
    if HAS_RCLPY:
        threading.Thread(target=start_ros2_bridge, daemon=True).start()
        print('  ROS2 bridge: enabled (rclpy found)')
    else:
        print('  ROS2 bridge: disabled (rclpy not found, run on Jetson)')

    # Start metrics poller in background (fallback for interventions file)
    t = threading.Thread(target=poll_metrics, daemon=True)
    t.start()

    app = await create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()

    print(f'')
    print(f'  CREStE-Nano Dashboard running!')
    print(f'  Open on your phone: http://192.168.1.125:{PORT}')
    print(f'')

    # Auto-launch data_collection mode so GPS + camera + teleop are live immediately
    threading.Thread(target=start_mode, args=('data_collection',), daemon=True).start()
    print('  Auto-starting data_collection mode...')

    await asyncio.Event().wait()

if __name__ == '__main__':
    asyncio.run(main())
