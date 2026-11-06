import os
import sys
import math
import subprocess
import threading
import ctypes
import numpy as np
from datetime import datetime

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    print("[ERROR] Install pybullet:  pip install pybullet")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.animation import FuncAnimation
    import matplotlib.gridspec as gridspec
except ImportError:
    print("[ERROR] Install matplotlib:  pip install matplotlib")
    sys.exit(1)

try:
    import imageio_ffmpeg
    HAS_IMAGEIO_FFMPEG = True
except ImportError:
    HAS_IMAGEIO_FFMPEG = False
    print("[ERROR] Install imageio-ffmpeg:  pip install imageio-ffmpeg")
    sys.exit(1)


TURTLEBOT_URDF_PATH = "C:/Users/Devan/pybullet_robots-master/pybullet_robots-master/data/turtlebot.urdf"
URDF_FALLBACK        = "r2d2.urdf"

ROBOT_COLOR     = [0.05, 0.05, 0.05, 1.0]
WHEEL_COLOR     = [0.02, 0.02, 0.02, 1.0]

K_ATTR         = 1.5
K_REP          = 12.0
D0             = 2.5
GOAL_TOLERANCE = 0.25

GOAL      = np.array([6.0, 6.0, 0.0])
START_POS = np.array([-5.0, -5.0, 0.05])
START_THETA = math.pi / 4

OBSTACLES = [
    {"pos": np.array([0.0,  0.0, 0.0]), "radius": 1.0, "height": 2.0},
    {"pos": np.array([3.0,  3.0, 0.0]), "radius": 1.2, "height": 2.5},
]

WHEEL_RADIUS    = 0.05
WHEEL_BASE      = 0.30
MAX_LINEAR_VEL  = 1.5
MAX_ANGULAR_VEL = 1.5
LIN_VEL_GAIN    = 1.0
ANG_VEL_GAIN    = 1.5
MOTOR_FORCE     = 10.0

SIM_DT                 = 1.0 / 120.0
CONTROL_DT             = 1.0 / 30.0
MAX_SIM_STEPS          = 8000
PHYSICS_STEPS_PER_FRAME = 12
ANIMATION_INTERVAL     = 1

VIDEO_FPS = 20
GIF_FPS = 15
GIF_FRAME_STRIDE = 2

CAM_WIDTH  = 480
CAM_HEIGHT = 360
CAM_FOV    = 90.0

OUTPUT_DIR = "apf_recordings"
os.makedirs(OUTPUT_DIR, exist_ok=True)
_timestamp            = datetime.now().strftime("%Y%m%d_%H%M%S")
RECORD_3D_PATH        = os.path.join(OUTPUT_DIR, f"3d_simulation_{_timestamp}.mp4")
RECORD_ODO_PATH       = os.path.join(OUTPUT_DIR, f"odometry_{_timestamp}.mp4")
RECORD_POV_PATH       = os.path.join(OUTPUT_DIR, f"robot_pov_{_timestamp}.mp4")


def attractive_potential(robot_pos, goal):
    diff = robot_pos - goal
    return 0.5 * K_ATTR * float(np.dot(diff, diff))


def attractive_gradient(robot_pos, goal):
    return K_ATTR * (robot_pos - goal)


def distance_to_cylinder_3d(robot_pos, obs):
    cx, cy, cz_bottom = obs["pos"]
    r = obs["radius"]
    h = obs["height"]
    z_top = cz_bottom + h
    closest_z = max(cz_bottom, min(robot_pos[2], z_top))
    dx = robot_pos[0] - cx
    dy = robot_pos[1] - cy
    dz = robot_pos[2] - closest_z
    horizontal = math.sqrt(dx * dx + dy * dy)
    radial = max(0.0, horizontal - r)
    return math.sqrt(radial * radial + dz * dz)


def repulsive_potential_3d(robot_pos, obs):
    d = distance_to_cylinder_3d(robot_pos, obs)
    if d <= 1e-9:
        return float("inf")
    if d > D0:
        return 0.0
    return 0.5 * K_REP * (1.0 / d - 1.0 / D0) ** 2


def repulsive_gradient_3d(robot_pos, obs):
    cx, cy, cz_bottom = obs["pos"]
    r = obs["radius"]
    h = obs["height"]
    z_top = cz_bottom + h
    closest_z = max(cz_bottom, min(robot_pos[2], z_top))
    dx = robot_pos[0] - cx
    dy = robot_pos[1] - cy
    dz = robot_pos[2] - closest_z
    horizontal = math.sqrt(dx * dx + dy * dy)
    radial = max(0.0, horizontal - r)
    d = math.sqrt(radial * radial + dz * dz)

    if horizontal < 1e-6:
        if abs(dz) < 1e-6:
            grad_dir = np.array([1.0, 0.0, 0.0])
        else:
            grad_dir = np.array([0.0, 0.0, np.sign(dz)])
    else:
        radial_dir_xy = np.array([dx / horizontal, dy / horizontal, 0.0])
        if abs(dz) < 1e-6:
            grad_dir = radial_dir_xy
        else:
            z_dir = np.array([0.0, 0.0, np.sign(dz)])
            grad_dir = (radial_dir_xy * radial + z_dir * abs(dz)) / (d + 1e-9)

    if d <= 1e-9:
        return -K_REP * 1000.0 * grad_dir
    if d > D0:
        return np.zeros(3)

    magnitude = K_REP * (1.0 / d - 1.0 / D0) * (1.0 / (d * d))
    return -magnitude * grad_dir


def compute_apf_force(robot_pos, goal, obstacles):
    F_attr = -attractive_gradient(robot_pos, goal)
    F_rep = np.zeros(3)
    for obs in obstacles:
        F_rep += -repulsive_gradient_3d(robot_pos, obs)
    F_total = F_attr + F_rep
    return F_total, F_attr, F_rep


def apf_to_ddmr_control(robot_pos, robot_theta, goal, obstacles):
    F, _, _ = compute_apf_force(robot_pos, goal, obstacles)
    F_xy = F[:2]
    F_mag = float(np.linalg.norm(F_xy))

    if F_mag < 1e-6:
        return 0.0, 0.0, F

    desired_theta = math.atan2(F_xy[1], F_xy[0])
    theta_error = desired_theta - robot_theta
    theta_error = math.atan2(math.sin(theta_error), math.cos(theta_error))

    omega = ANG_VEL_GAIN * theta_error
    omega = max(-MAX_ANGULAR_VEL, min(MAX_ANGULAR_VEL, omega))

    alignment = math.cos(theta_error)
    v = LIN_VEL_GAIN * F_mag * max(0.0, alignment)
    v = max(0.0, min(MAX_LINEAR_VEL, v))

    if abs(theta_error) > math.pi / 2:
        v = 0.0

    return v, omega, F


def dd_to_wheel_velocities(v, omega):
    v_left_lin  = v - omega * WHEEL_BASE / 2.0
    v_right_lin = v + omega * WHEEL_BASE / 2.0
    return v_left_lin / WHEEL_RADIUS, v_right_lin / WHEEL_RADIUS


def setup_pybullet():
    p.connect(p.GUI)

    # Hide PyBullet GUI elements: bottom parameter panel, camera RGB/Depth/Segmentation views
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(SIM_DT)
    p.loadURDF("plane.urdf")
    robot_id = load_robot()
    make_robot_black(robot_id)
    obstacle_ids = create_obstacles()
    goal_id = create_goal_marker(GOAL)
    p.resetDebugVisualizerCamera(
        cameraDistance=18.0,
        cameraYaw=35,
        cameraPitch=-65,
        cameraTargetPosition=[0.5, 0.5, 0.5],
    )
    return robot_id, obstacle_ids, goal_id


def load_robot():
    candidates = [TURTLEBOT_URDF_PATH, URDF_FALLBACK]
    for path in candidates:
        try:
            if path == URDF_FALLBACK:
                orn = p.getQuaternionFromEuler([0, 0, START_THETA])
                rid = p.loadURDF(path, START_POS.tolist(), orn)
            else:
                if not os.path.exists(path):
                    continue
                orn = p.getQuaternionFromEuler([0, 0, START_THETA])
                rid = p.loadURDF(path, START_POS.tolist(), orn)
            print(f"[INFO] Loaded robot URDF: {path}")
            return rid
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
    raise RuntimeError("No URDF could be loaded.")


def make_robot_black(robot_id):
    n = p.getNumJoints(robot_id)
    print(f"[INFO] Coloring robot black ({n + 1} parts)...")
    try:
        p.changeVisualShape(robot_id, -1, rgbaColor=ROBOT_COLOR,
                            specularColor=[0.3, 0.3, 0.3])
    except Exception as e:
        print(f"[WARN] Base color change failed: {e}")
    for i in range(n):
        try:
            info = p.getJointInfo(robot_id, i)
            name = info[1].decode("utf-8").lower()
            color = WHEEL_COLOR if "wheel" in name else ROBOT_COLOR
            p.changeVisualShape(robot_id, i, rgbaColor=color,
                                specularColor=[0.3, 0.3, 0.3])
        except Exception as e:
            print(f"[WARN] Link {i} color change failed: {e}")
    print("[INFO] Robot colored black.")


def find_wheel_joints(robot_id):
    n = p.getNumJoints(robot_id)
    print(f"[INFO] Robot has {n} joints:")
    print("-" * 60)

    revolute_joints = []
    wheel_joints = []

    for i in range(n):
        info = p.getJointInfo(robot_id, i)
        name = info[1].decode("utf-8")
        jtype = info[2]
        type_map = {
            p.JOINT_REVOLUTE:   "REVOLUTE",
            p.JOINT_PRISMATIC:  "PRISMATIC",
            p.JOINT_SPHERICAL:  "SPHERICAL",
            p.JOINT_PLANAR:     "PLANAR",
            p.JOINT_FIXED:      "FIXED",
        }
        type_str = type_map.get(jtype, f"UNKNOWN({jtype})")
        print(f"  Joint {i:2d}: '{name}'  type={type_str}")

        name_lower = name.lower()
        if jtype == p.JOINT_REVOLUTE:
            revolute_joints.append((i, name_lower))
            if "wheel" in name_lower:
                wheel_joints.append((i, name_lower))

    print("-" * 60)
    print(f"[INFO] Revolute joints: {revolute_joints}")
    print(f"[INFO] Wheel joints   : {wheel_joints}")

    left = right = None
    for idx, name in wheel_joints:
        if "left" in name:
            left = idx
        elif "right" in name:
            right = idx

    if left is None or right is None:
        for idx, name in revolute_joints:
            if "left" in name and left is None:
                left = idx
            elif "right" in name and right is None:
                right = idx

    if (left is None or right is None) and len(wheel_joints) >= 2:
        left, right = wheel_joints[0][0], wheel_joints[1][0]

    if (left is None or right is None) and len(revolute_joints) >= 2:
        left, right = revolute_joints[0][0], revolute_joints[1][0]

    if left is None or right is None:
        raise RuntimeError(
            "Could not detect left/right wheel joints. Check the joint list above."
        )

    print(f"[INFO] Left wheel joint = {left}")
    print(f"[INFO] Right wheel joint = {right}")
    return left, right


def create_obstacles():
    ids = []
    for i, obs in enumerate(OBSTACLES):
        col = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=obs["radius"], height=obs["height"]
        )
        vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=obs["radius"],
            length=obs["height"],
            rgbaColor=[0.85, 0.20, 0.20, 0.85],
        )
        pos = [obs["pos"][0], obs["pos"][1], obs["pos"][2] + obs["height"] / 2.0]
        oid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
        )
        ids.append(oid)
    return ids


def create_goal_marker(goal_pos):
    vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.25, rgbaColor=[0.10, 0.90, 0.10, 0.90]
    )
    gid = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=vis,
        basePosition=[goal_pos[0], goal_pos[1], 0.25],
    )
    beam = p.createVisualShape(
        p.GEOM_CYLINDER, radius=0.03, length=2.0, rgbaColor=[0.10, 0.90, 0.10, 0.6]
    )
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=beam,
                      basePosition=[goal_pos[0], goal_pos[1], 1.0])
    return gid


def get_robot_state(robot_id):
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    euler = p.getEulerFromQuaternion(orn)
    return np.array([pos[0], pos[1], pos[2]]), euler[2]


def get_robot_pov_image(robot_id, width=CAM_WIDTH, height=CAM_HEIGHT, fov=CAM_FOV):
    """
    Synthetic camera mounted on the robot, looking forward at 90 deg FOV.
    Returns RGB numpy array of shape (height, width, 3).
    """
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    euler = p.getEulerFromQuaternion(orn)
    yaw = euler[2]

    cam_height_offset = 0.35
    eye = [pos[0], pos[1], pos[2] + cam_height_offset]

    look_dist = 3.0
    target = [
        pos[0] + look_dist * math.cos(yaw),
        pos[1] + look_dist * math.sin(yaw),
        pos[2] + cam_height_offset,
    ]

    up = [0, 0, 1]

    view_matrix = p.computeViewMatrix(eye, target, up)
    aspect = width / height
    proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, 0.01, 100)

    try:
        _, _, px, _, _ = p.getCameraImage(
            width, height, view_matrix, proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=p.ER_NO_SEGMENTATION_MASK,
        )
    except Exception:
        try:
            _, _, px, _, _ = p.getCameraImage(
                width, height, view_matrix, proj_matrix,
                renderer=p.ER_TINY_RENDERER,
                flags=p.ER_NO_SEGMENTATION_MASK,
            )
        except Exception:
            return None

    img = np.array(px, dtype=np.uint8).reshape(height, width, 4)
    return img[:, :, :3]


def write_mp4_ffmpeg(frames, path, fps=VIDEO_FPS):
    """
    Write MP4 using ffmpeg directly via subprocess.
    Uses H264 codec + yuv420p pixel format for maximum compatibility
    (VS Code, Windows Media Player, QuickTime, browsers, etc.)
    """
    if not frames:
        print("[WARN] No frames to save.")
        return False

    h, w = frames[0].shape[:2]
    print(f"[INFO] Writing {len(frames)} frames ({w}x{h}) to {path} via ffmpeg...")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_exe, '-y',
        '-loglevel', 'error',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-crf', '23',
        '-movflags', '+faststart',
        path,
    ]

    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Drain stderr in a background thread to prevent pipe deadlock.
        # Without this, ffmpeg's stderr can fill the OS pipe buffer (~64KB),
        # causing ffmpeg to block on stderr write -> stop reading stdin ->
        # Python blocks on stdin write -> DEADLOCK (TimeoutExpired).
        stderr_chunks = []
        def drain_stderr():
            try:
                while True:
                    chunk = proc.stderr.read(4096)
                    if not chunk:
                        break
                    stderr_chunks.append(chunk)
            except Exception:
                pass
        stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
        stderr_thread.start()

        # Write frames to stdin
        try:
            for f in frames:
                proc.stdin.write(f.tobytes())
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

        # Wait for ffmpeg to finish (no timeout -> cannot TimeoutExpired)
        stderr_thread.join()
        proc.wait()

        if os.path.exists(path) and os.path.getsize(path) > 1000:
            size_kb = os.path.getsize(path) // 1024
            print(f"[INFO] MP4 saved (ffmpeg H264): {path} ({size_kb} KB)")
            return True
        else:
            stderr = b''.join(stderr_chunks).decode(errors='replace')[-500:]
            print(f"[ERROR] MP4 file too small or missing.")
            print(f"[ffmpeg stderr]: {stderr}")
            return False
    except Exception as e:
        print(f"[ERROR] ffmpeg subprocess failed: {e}")
        return False


# ============================================================================
# WINDOW ARRANGEMENT (side-by-side for OBS recording)
# ============================================================================

def get_screen_resolution():
    """Get primary screen resolution (Windows)."""
    try:
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


def find_window_by_title(substring):
    """Find a window whose title contains the substring (Windows)."""
    if not sys.platform.startswith("win"):
        return None
    try:
        user32 = ctypes.windll.user32
        found = []

        EnumWindowsProc = ctypes.WINFUNCTYPE(
            ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p
        )

        def callback(hwnd, _):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                if substring.lower() in buf.value.lower():
                    found.append(hwnd)
            return True

        user32.EnumWindows(EnumWindowsProc(callback), 0)
        return found[0] if found else None
    except Exception:
        return None


def move_window(hwnd, x, y, w, h, remove_border=True):
    """Move and resize a window by HWND (Windows). Borderless for clean fullscreen."""
    try:
        user32 = ctypes.windll.user32
        # SWP_NOZORDER=0x0004 | SWP_SHOWWINDOW=0x0040 | SWP_FRAMECHANGED=0x0020
        flags = 0x0004 | 0x0040 | 0x0020

        if remove_border:
            # Remove title bar and borders: WS_POPUP=0x80000000, remove WS_CAPTION(0xC00000) & WS_THICKFRAME(0x40000)
            GWL_STYLE = -16
            WS_POPUP = 0x80000000
            WS_VISIBLE = 0x10000000
            current_style = user32.GetWindowLongW(hwnd, GWL_STYLE)
            new_style = (current_style & ~0xC00000 & ~0x40000) | WS_POPUP | WS_VISIBLE
            user32.SetWindowLongW(hwnd, GWL_STYLE, new_style)

        user32.SetWindowPos(hwnd, 0, x, y, w, h, flags)
        return True
    except Exception as e:
        print(f"[WARN] move_window failed: {e}")
        return False


def arrange_windows_side_by_side(odo_gui):
    """
    Arrange PyBullet (left half) and Matplotlib (right half) side by side,
    both borderless and fullscreen. Perfect for OBS capture - no overlap.
    """
    sw, sh = get_screen_resolution()
    half_w = sw // 2
    print(f"[INFO] Screen resolution: {sw}x{sh}")
    print(f"[INFO] Arranging windows side-by-side (each {half_w}x{sh}), borderless...")

    # --- 1. Move PyBullet window to LEFT half (borderless fullscreen) ---
    pybullet_hwnd = None
    for title in ["PyBullet", "Bullet", "OpenGL", "ExampleBrowser"]:
        pybullet_hwnd = find_window_by_title(title)
        if pybullet_hwnd:
            break

    if pybullet_hwnd:
        if move_window(pybullet_hwnd, 0, 0, half_w, sh, remove_border=False):
            print(f"[INFO] PyBullet window -> left half (0,0 {half_w}x{sh})")
        else:
            print("[WARN] Failed to move PyBullet window")
    else:
        print("[WARN] PyBullet window not found - arrange manually if needed")

    # --- 2. Move Matplotlib window to RIGHT half (borderless fullscreen) ---
    try:
        mgr = odo_gui.fig.canvas.manager
        if hasattr(mgr, 'window'):
            win = mgr.window
            # TkAgg backend
            if hasattr(win, 'wm_geometry'):
                win.wm_geometry(f"{half_w}x{sh}+{half_w}+0")
                # Remove decorations on Tk
                try:
                    win.overrideredirect(True)  # borderless
                    win.geometry(f"{half_w}x{sh}+{half_w}+0")
                except Exception:
                    pass
                print(f"[INFO] Matplotlib window -> right half ({half_w},0 {half_w}x{sh})")
            # Qt backend
            elif hasattr(win, 'setGeometry'):
                win.setGeometry(half_w, 0, half_w, sh)
                # Frameless on Qt
                try:
                    from PyQt5.QtCore import Qt
                    win.setWindowFlags(Qt.FramelessWindowHint)
                    win.show()
                except Exception:
                    pass
                print(f"[INFO] Matplotlib window -> right half ({half_w},0 {half_w}x{sh})")
            else:
                print("[WARN] Unknown matplotlib backend - arrange manually")
        else:
            print("[WARN] Matplotlib window manager not available")
    except Exception as e:
        print(f"[WARN] Could not move matplotlib window: {e}")


class OdometryGUI:
    def __init__(self, goal, obstacles, start_pos):
        self.goal = goal
        self.obstacles = obstacles
        self.start_pos = start_pos

        self.traj_x = []
        self.traj_y = []
        self.robot_arrow = None
        self.force_arrow = None
        self.frames = []
        self.reached_goal = False

        self.fig = plt.figure(figsize=(10, 6), dpi=80)
        gs = gridspec.GridSpec(
            2, 2, height_ratios=[1, 1.3],
            hspace=0.35, wspace=0.25,
        )
        self.ax_attr = self.fig.add_subplot(gs[0, 0])
        self.ax_rep  = self.fig.add_subplot(gs[0, 1])
        self.ax_main = self.fig.add_subplot(gs[1, :])

        self.fig.suptitle(
            "GUI 2 - Odometry DDMR (APF)  |  Attractive + Repulsive Field  |  Auto-Recording",
            fontsize=13, fontweight="bold",
        )

        self._precompute_fields()
        self._setup_attractive_plot()
        self._setup_repulsive_plot()
        self._setup_main_plot()

    def _precompute_fields(self):
        xs = np.arange(-7, 8, 0.25)
        ys = np.arange(-7, 8, 0.25)
        self.X, self.Y = np.meshgrid(xs, ys)

        self.Z_attr  = np.zeros_like(self.X)
        self.Z_rep   = np.zeros_like(self.X)
        self.Z_total = np.zeros_like(self.X)

        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                q = np.array([self.X[i, j], self.Y[i, j], 0.0])
                u_a = attractive_potential(q, self.goal)
                u_r = 0.0
                for obs in self.obstacles:
                    u_r += repulsive_potential_3d(q, obs)
                self.Z_attr[i, j]  = u_a
                self.Z_rep[i, j]   = min(u_r, 80.0)
                self.Z_total[i, j] = min(u_a + u_r, 80.0)

    def _setup_attractive_plot(self):
        ax = self.ax_attr
        ax.set_xlim(-7, 8)
        ax.set_ylim(-7, 8)
        ax.set_aspect("equal")
        ax.set_title("(a) Attractive Potential Field (GOAL)",
                     fontsize=11, fontweight="bold", color="navy")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.2)

        cf = ax.contourf(self.X, self.Y, self.Z_attr, levels=30, cmap="Blues")
        cs = ax.contour(self.X, self.Y, self.Z_attr, levels=10,
                        colors="navy", linewidths=0.4, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.1f")

        ax.add_patch(Circle((self.goal[0], self.goal[1]), 0.25,
                            color="lime", alpha=0.95))
        ax.plot(self.goal[0], self.goal[1], "*", color="lime",
                markersize=14, markeredgecolor="black")
        ax.text(self.goal[0] + 0.35, self.goal[1] + 0.35, "GOAL",
                color="lime", fontsize=10, fontweight="bold")

        ax.add_patch(Circle((self.start_pos[0], self.start_pos[1]), 0.18,
                            color="cyan", alpha=0.9))
        ax.text(self.start_pos[0] + 0.3, self.start_pos[1] - 0.4, "START",
                color="cyan", fontsize=9, fontweight="bold")

        for obs in self.obstacles:
            ax.add_patch(Circle((obs["pos"][0], obs["pos"][1]), obs["radius"],
                                color="red", alpha=0.4, fill=False,
                                linestyle="--", linewidth=1.2))

        self.fig.colorbar(cf, ax=ax, shrink=0.7, label="U_attr (J)")
        self.traj_attr, = ax.plot([], [], "cyan", linewidth=1.5, alpha=0.9)

    def _setup_repulsive_plot(self):
        ax = self.ax_rep
        ax.set_xlim(-7, 8)
        ax.set_ylim(-7, 8)
        ax.set_aspect("equal")
        ax.set_title("(b) Repulsive Potential Field (OBSTACLES)",
                     fontsize=11, fontweight="bold", color="darkred")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.2)

        cf = ax.contourf(self.X, self.Y, self.Z_rep, levels=30, cmap="Reds")
        cs = ax.contour(self.X, self.Y, self.Z_rep, levels=8,
                        colors="maroon", linewidths=0.4, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.1f")

        for k, obs in enumerate(self.obstacles):
            ax.add_patch(Circle((obs["pos"][0], obs["pos"][1]), obs["radius"],
                                color="red", alpha=0.85,
                                label=f"Obs {k+1} (r={obs['radius']}, h={obs['height']}m)"))
            ax.add_patch(Circle((obs["pos"][0], obs["pos"][1]),
                                obs["radius"] + D0,
                                color="red", alpha=0.2, linestyle="--",
                                fill=False, linewidth=1.0))
            ax.text(obs["pos"][0], obs["pos"][1], f"O{k+1}",
                    ha="center", va="center", color="white",
                    fontsize=9, fontweight="bold")

        ax.add_patch(Circle((self.goal[0], self.goal[1]), 0.18,
                            color="lime", alpha=0.7))
        ax.add_patch(Circle((self.start_pos[0], self.start_pos[1]), 0.18,
                            color="cyan", alpha=0.7))

        ax.legend(loc="lower right", fontsize=7)
        self.fig.colorbar(cf, ax=ax, shrink=0.7, label="U_rep (J)")
        self.traj_rep, = ax.plot([], [], "cyan", linewidth=1.5, alpha=0.9)

    def _setup_main_plot(self):
        ax = self.ax_main
        ax.set_xlim(-7, 8)
        ax.set_ylim(-7, 8)
        ax.set_aspect("equal")
        ax.set_title("(c) Combined Potential Field + Trajectory (Odometry)",
                     fontsize=11, fontweight="bold", color="black")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.3)

        cf = ax.contourf(self.X, self.Y, self.Z_total, levels=30,
                         cmap="viridis", alpha=0.45)
        cs = ax.contour(self.X, self.Y, self.Z_total, levels=10,
                        colors="gray", linewidths=0.4, alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f")

        for k, obs in enumerate(self.obstacles):
            ax.add_patch(Circle((obs["pos"][0], obs["pos"][1]), obs["radius"],
                                color="red", alpha=0.75,
                                label="Obstacle" if k == 0 else None))
            ax.add_patch(Circle((obs["pos"][0], obs["pos"][1]),
                                obs["radius"] + D0,
                                color="red", alpha=0.15, linestyle="--",
                                fill=False, linewidth=1.0))

        ax.add_patch(Circle((self.goal[0], self.goal[1]), 0.25,
                            color="green", alpha=0.9, label="Goal"))
        ax.plot(self.goal[0], self.goal[1], "*", color="lime",
                markersize=14, markeredgecolor="black")
        ax.add_patch(Circle((self.start_pos[0], self.start_pos[1]), 0.18,
                            color="blue", alpha=0.9, label="Start"))

        self.traj_line, = ax.plot([], [], "cyan", linewidth=2.2,
                                  label="Trajectory")

        self.info_text = ax.text(
            0.01, 0.98, "", transform=ax.transAxes,
            verticalalignment="top", fontsize=8, family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
        )
        ax.legend(loc="lower right", fontsize=8)

    def update(self, robot_pos, robot_theta, force, v, omega, step):
        self.traj_x.append(robot_pos[0])
        self.traj_y.append(robot_pos[1])

        dist_goal = float(np.linalg.norm(robot_pos[:2] - self.goal[:2]))
        if dist_goal < GOAL_TOLERANCE:
            self.reached_goal = True

        self.traj_line.set_data(self.traj_x, self.traj_y)
        self.traj_attr.set_data(self.traj_x, self.traj_y)
        self.traj_rep.set_data(self.traj_x, self.traj_y)

        ax = self.ax_main
        if self.robot_arrow is not None:
            self.robot_arrow.remove()
        al = 0.6
        self.robot_arrow = ax.arrow(
            robot_pos[0], robot_pos[1],
            al * math.cos(robot_theta), al * math.sin(robot_theta),
            head_width=0.18, head_length=0.10, fc="blue", ec="blue",
        )

        if self.force_arrow is not None:
            self.force_arrow.remove()
        Fxy = force[:2]
        Fm = float(np.linalg.norm(Fxy))
        if Fm > 0.01:
            scale = 0.5 / max(Fm, 0.1)
            self.force_arrow = ax.arrow(
                robot_pos[0], robot_pos[1],
                Fxy[0] * scale, Fxy[1] * scale,
                head_width=0.12, head_length=0.06, fc="orange", ec="orange",
            )

        self.info_text.set_text(
            f"Step     : {step}\n"
            f"Pos      : ({robot_pos[0]:+.2f}, {robot_pos[1]:+.2f})\n"
            f"Theta    : {math.degrees(robot_theta):+7.1f} deg\n"
            f"v        : {v:.3f} m/s\n"
            f"omega    : {omega:+.3f} rad/s\n"
            f"|F_apf|  : {Fm:.3f}\n"
            f"Goal dist: {dist_goal:.2f} m\n"
            f"Status   : {'REACHED' if self.reached_goal else 'NAVIGATING'}"
        )

    def capture_frame(self):
        try:
            self.fig.canvas.draw()
            w, h = self.fig.canvas.get_width_height()
            buf = self.fig.canvas.buffer_rgba()
            img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
            self.frames.append(img[:, :, :3].copy())
        except Exception:
            pass

    def save_recording(self, path):
        """Save odometry as MP4 (with GIF fallback if MP4 fails)."""
        if not self.frames:
            print("[WARN] No frames to save.")
            return False

        print(f"[INFO] Saving {len(self.frames)} odometry frames as MP4...")
        ok = write_mp4_ffmpeg(self.frames, path, fps=VIDEO_FPS)

        if not ok:
            print("[WARN] MP4 failed, falling back to GIF...")
            gif_path = path.replace(".mp4", ".gif")
            frames_to_save = self.frames[::GIF_FRAME_STRIDE]
            try:
                imageio.mimsave(gif_path, frames_to_save, fps=GIF_FPS,
                                subrectangles=True, palettesize=128)
                size_kb = os.path.getsize(gif_path) // 1024
                print(f"[INFO] GIF fallback saved: {gif_path} ({size_kb} KB)")
                return True
            except Exception as e:
                print(f"[ERROR] GIF fallback also failed: {e}")
                return False
        return ok


def main():
    print("=" * 72)
    print(" APF Simulation - DDMR (Differential Drive Mobile Robot)")
    print(" - 2 obstacles 3D with 3D Repulsive Potential Field")
    print(" - Goal with Attractive Potential Field")
    print(" - 3 separate recordings:")
    print("     1. PyBullet 3D view  -> 3d_simulation_*.mp4")
    print("     2. Matplotlib odometry -> odometry_*.mp4")
    print("     3. Robot POV camera (90 deg FOV) -> robot_pov_*.mp4")
    print(" - Auto-record & auto-save when goal is reached")
    print(" - Robot (TurtleBot2) colored BLACK")
    print("=" * 72)
    print(f"[INFO] ffmpeg available: {HAS_IMAGEIO_FFMPEG}")
    print(f"[INFO] Output 3D  : {RECORD_3D_PATH}")
    print(f"[INFO] Output Odo: {RECORD_ODO_PATH}")
    print(f"[INFO] Output POV: {RECORD_POV_PATH}")

    robot_id, obstacle_ids, goal_id = setup_pybullet()
    left_wheel, right_wheel = find_wheel_joints(robot_id)

    recording_3d = False
    try:
        p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, RECORD_3D_PATH)
        recording_3d = True
        print("[INFO] 3D recording started.")
    except Exception as e:
        print(f"[WARN] 3D recording failed: {e}")

    odo_gui = OdometryGUI(GOAL, OBSTACLES, START_POS)

    state = {
        "step": 0, "last_ctrl": 0, "done": False, "saved": False,
        "v": 0.0, "omega": 0.0, "force": np.zeros(3),
        "goal_reached": False,
        "pov_frames": [],
    }
    control_interval = max(1, int(CONTROL_DT / SIM_DT))

    def simulation_step():
        if state["done"]:
            return
        p.stepSimulation()
        state["step"] += 1

        robot_pos, robot_theta = get_robot_state(robot_id)

        if state["step"] - state["last_ctrl"] >= control_interval:
            v, omega, force = apf_to_ddmr_control(
                robot_pos, robot_theta, GOAL, OBSTACLES
            )
            state["v"] = v
            state["omega"] = omega
            state["force"] = force
            state["last_ctrl"] = state["step"]

        vL, vR = dd_to_wheel_velocities(state["v"], state["omega"])
        p.setJointMotorControl2(robot_id, left_wheel, p.VELOCITY_CONTROL,
                                targetVelocity=vL, force=MOTOR_FORCE)
        p.setJointMotorControl2(robot_id, right_wheel, p.VELOCITY_CONTROL,
                                targetVelocity=vR, force=MOTOR_FORCE)

        odo_gui.update(robot_pos, robot_theta, state["force"],
                       state["v"], state["omega"], state["step"])

        pov_img = get_robot_pov_image(robot_id)
        if pov_img is not None:
            state["pov_frames"].append(pov_img)

        dist = float(np.linalg.norm(robot_pos[:2] - GOAL[:2]))
        if dist < GOAL_TOLERANCE:
            print(f"[INFO] Goal reached at step {state['step']}!")
            state["goal_reached"] = True
            state["done"] = True
        elif state["step"] >= MAX_SIM_STEPS:
            print(f"[INFO] Maximum steps reached.")
            state["done"] = True

    def save_all_and_cleanup():
        if state["saved"]:
            return
        state["saved"] = True
        print("[INFO] Saving recordings...")
        if recording_3d:
            try:
                p.stopStateLogging()
                if os.path.exists(RECORD_3D_PATH):
                    print(f"[INFO] 3D MP4 saved : {RECORD_3D_PATH} "
                          f"({os.path.getsize(RECORD_3D_PATH)//1024} KB)")
                else:
                    print(f"[WARN] 3D MP4 not found")
            except Exception as e:
                print(f"[WARN] stopStateLogging failed: {e}")

        odo_gui.capture_frame()
        odo_gui.save_recording(RECORD_ODO_PATH)

        if state["pov_frames"]:
            write_mp4_ffmpeg(state["pov_frames"], RECORD_POV_PATH, fps=VIDEO_FPS)
        else:
            print("[WARN] No POV frames captured.")

        try:
            p.disconnect()
        except Exception:
            pass
        print("[INFO] Done. Thank you.")

    def update_frame(_frame):
        if state["done"]:
            if not state["saved"]:
                save_all_and_cleanup()
                plt.close(odo_gui.fig)
            return []
        for _ in range(PHYSICS_STEPS_PER_FRAME):
            if state["done"]:
                break
            simulation_step()
        odo_gui.capture_frame()
        return []

    ani = FuncAnimation(
        odo_gui.fig, update_frame,
        interval=ANIMATION_INTERVAL, blit=False, cache_frame_data=False,
    )

    # Show matplotlib window (non-blocking) so its HWND exists,
    # then arrange both windows side-by-side for OBS recording.
    plt.show(block=False)
    plt.pause(0.3)
    arrange_windows_side_by_side(odo_gui)

    # Re-arrange after a short delay (windows often need retries to settle,
    # especially PyBullet's OpenGL window which initializes slowly)
    def rearrange_later():
        import time
        time.sleep(1.0)
        arrange_windows_side_by_side(odo_gui)
        time.sleep(0.5)
        # Final pass
        arrange_windows_side_by_side(odo_gui)

    t = threading.Thread(target=rearrange_later, daemon=True)
    t.start()

    # Re-enter blocking main loop
    plt.show()

    save_all_and_cleanup()


if __name__ == "__main__":
    main()
