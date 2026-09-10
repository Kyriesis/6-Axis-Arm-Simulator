"""web_server.py — 虚拟六轴机械臂网页控制后端（仅标准库）

本地 HTTP 服务：提供 JSON API 与静态网页（web/index.html），
Python 运动学模型（arm_model / ik_solver / trajectory）为唯一真源。

API:
    GET  /api/limits                     -> 6 轴限位
    GET  /api/fk?j=q1,q2,q3,q4,q5,q6     -> 各关节坐标 + 末端位姿
    POST /api/ik    {position:[x,y,z], orientation?:[r,p,y], initial?:[6]}
                                           -> 关节角（失败返回 success:false）
    POST /api/trajectory {from:[6], to:[6], profile?, dt?, with_fk?}
                                           -> 采样角度序列（可选每帧 FK 坐标）
    POST /api/demo/circle {center?, radius?, orientation?, points?, with_fk?}
                                           -> 圆形末端轨迹（离线 IK + 平滑路径）

运行:  python web_server.py [端口]
启动后自动打开浏览器；端口被占用时自动向后探测（8000-8010）。
"""
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arm_model as am
import ik_solver
import trajectory as traj

# ---------------- 仿真参数（与 simulation.py 一致） ----------------
V_MAX = np.array([120.0, 90.0, 90.0, 180.0, 180.0, 360.0])    # 度/秒
A_MAX = np.array([300.0, 240.0, 240.0, 540.0, 540.0, 900.0])  # 度/秒^2
DT = 0.02                                                      # 默认采样周期（秒）

# 圆形演示默认参数（均已验证可达）
CIRCLE_CENTER = [0.15, 0.0, 0.12]   # 圆心（m）
CIRCLE_RADIUS = 0.06                # 半径（m）
CIRCLE_RPY = [180.0, 0.0, 0.0]      # 工具姿态（度，ZYX 欧拉角）
CIRCLE_POINTS = 61                  # 离线 IK 路标数

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


# ---------------- 业务逻辑 ----------------
def fk_payload(q_deg):
    """FK 结果字典：各关节坐标 + 末端位姿。"""
    q_deg = am.clamp_to_limits(np.asarray(q_deg, dtype=float))
    transforms = am.forward_kinematics(q_deg)
    joints = [T[:3, 3].tolist() for T in transforms]
    pose = am.pose_to_xyz_rpy(transforms[6])
    return {
        "q": q_deg.tolist(),
        "joints": joints,                       # 7 个点（基座 + 6 关节/末端）
        "pose": {"position": pose[:3].tolist(),
                 "rpy_deg": pose[3:].tolist()},
    }


def ik_payload(position, orientation=None, initial=None):
    """IK 求解。orientation 为 ZYX 欧拉角（度），缺省保持当前姿态。"""
    position = np.asarray(position, dtype=float)
    if orientation is not None:
        T_target = am.make_pose(*position, *orientation)
    else:
        # 只给位置: 姿态取初值（或零位）的当前姿态
        q_init = (np.asarray(initial, dtype=float) if initial is not None
                  else np.mean(am.JOINT_LIMITS, axis=1))
        T_cur = am.end_pose(am.clamp_to_limits(q_init))
        T_target = T_cur.copy()
        T_target[:3, 3] = position
    q0 = (np.asarray(initial, dtype=float) if initial is not None else None)
    q_sol, info = ik_solver.inverse_kinematics(T_target, q0_deg=q0)
    return {
        "success": bool(info["success"]),
        "q": q_sol.tolist(),
        "pos_err": info["pos_err"],
        "ori_err": info["ori_err"],
        "in_limits": bool(info["in_limits"]),
        "message": "" if info["success"] else "IK 未收敛（目标可能不可达）",
    }


def trajectory_payload(q_from, q_to, profile="trapezoid", dt=DT, with_fk=False):
    """点到点轨迹。"""
    q0 = am.clamp_to_limits(np.asarray(q_from, dtype=float))
    q1 = am.clamp_to_limits(np.asarray(q_to, dtype=float))
    if profile not in ("trapezoid", "scurve"):
        profile = "trapezoid"
    dt = float(np.clip(dt, 0.005, 0.1))
    data = traj.plan_trajectory(q0, q1, V_MAX, A_MAX, dt=dt, profile=profile)
    payload = {
        "dt": dt,
        "profile": profile,
        "duration": float(data["T"]),
        "n_frames": int(len(data["t"])),
        "q": np.round(data["q"], 4).tolist(),
    }
    if with_fk:
        payload["frames"] = [am.joint_positions(qq).tolist() for qq in data["q"]]
    return payload


def circle_payload(center=None, radius=None, orientation=None, points=None,
                   dt=DT, with_fk=True, initial=None, plane="xy"):
    """圆形末端轨迹：离线 IK 连续解链 + Catmull-Rom 平滑路径。

    plane: "xy" 水平圆（默认）或 "xz" 竖直圆。
    """
    center = np.asarray(center if center is not None else CIRCLE_CENTER,
                        dtype=float)
    radius = float(radius if radius is not None else CIRCLE_RADIUS)
    rpy = np.asarray(orientation if orientation is not None else CIRCLE_RPY,
                     dtype=float)
    n_pts = int(np.clip(points if points is not None else CIRCLE_POINTS, 9, 361))
    q0 = (np.asarray(initial, dtype=float) if initial is not None
          else np.array([0.0, 0.0, 90.0, 0.0, 0.0, 0.0]))

    poses = []
    for k in range(n_pts):
        th = 2.0 * np.pi * k / (n_pts - 1)
        if plane == "xz":
            poses.append(am.make_pose(center[0] + radius * np.cos(th),
                                      center[1], center[2] + radius * np.sin(th),
                                      *rpy))
        else:
            poses.append(am.make_pose(center[0] + radius * np.cos(th),
                                      center[1] + radius * np.sin(th),
                                      center[2], *rpy))
    waypoints = ik_solver.ik_continuous(poses, q0)
    data = traj.plan_path(waypoints, V_MAX, A_MAX, dt=dt)
    # 计算轨迹实际圆度误差（每 3 帧抽测 FK）
    max_err = 0.0
    for k in range(0, len(data["q"]), 3):
        p = am.end_pose(data["q"][k])[:3, 3] - center
        if plane == "xz":
            err = max(abs(np.hypot(p[0], p[2]) - radius), abs(p[1]))
        else:
            err = max(abs(np.hypot(p[0], p[1]) - radius), abs(p[2]))
        max_err = max(max_err, err)
    return {
        "dt": dt,
        "duration": float(data["T"]),
        "n_frames": int(len(data["t"])),
        "center": center.tolist(),
        "radius": radius,
        "plane": plane,
        "max_err_m": float(max_err),
        "q": np.round(data["q"], 4).tolist(),
        "frames": [am.joint_positions(qq).tolist() for qq in data["q"]],
    }


def random_target_payload(current=None, seed=None):
    """生成一个"合理的"随机目标关节角：限位内（留 5° 边距）、
    末端距基座在工作空间中段（0.15~0.35 m）、雅可比条件数良好
    （远离奇异），且与当前位置变化足够大（>15°）。
    """
    rng = np.random.default_rng(seed)
    q_cur = np.asarray(current, dtype=float) if current is not None else None
    lo = am.JOINT_LIMITS[:, 0] + 5.0
    hi = am.JOINT_LIMITS[:, 1] - 5.0
    for _ in range(500):
        q = rng.uniform(lo, hi)
        T = am.end_pose(q)
        p = T[:3, 3]
        dist = float(np.linalg.norm(p))
        if not (0.15 <= dist <= 0.35):
            continue
        if p[2] < 0.03:   # 末端不低于安装平面，避免"穿桌"姿态
            continue
        cond = float(np.linalg.cond(am.geometric_jacobian(q)))
        if cond > 100.0:
            continue
        if q_cur is not None and np.max(np.abs(q - q_cur)) < 15.0:
            continue
        return {"q": q.tolist(), "position": p.tolist(),
                "cond": cond, "dist": dist}
    return {"error": "未找到满足条件的随机目标"}


# ---------------- HTTP 服务 ----------------
class Handler(BaseHTTPRequestHandler):
    server_version = "DummyArmWeb/1.0"

    # ---- 工具 ----
    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type):
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def log_message(self, fmt, *args):  # 精简日志
        sys.stdout.write("[http] %s\n" % (fmt % args))

    # ---- 路由 ----
    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            route = parsed.path
            if route == "/" or route == "/index.html":
                return self._send_file(os.path.join(WEB_DIR, "index.html"),
                                       "text/html; charset=utf-8")
            if route == "/api/limits":
                return self._send_json({
                    "limits": am.JOINT_LIMITS.tolist(),
                    "dims": am.ARM_DIMS,
                })
            if route == "/api/fk":
                qs = parse_qs(parsed.query)
                q = [float(x) for x in qs.get("j", ["0,0,90,0,0,0"])[0].split(",")]
                if len(q) != 6:
                    return self._send_json({"error": "需要 6 个关节角"}, 400)
                return self._send_json(fk_payload(q))
            if route == "/api/demo/circle":
                qs = parse_qs(parsed.query)
                def _f(key, default):
                    return (float(qs[key][0]) if key in qs else default)
                center = ([_f("cx", CIRCLE_CENTER[0]), _f("cy", CIRCLE_CENTER[1]),
                           _f("cz", CIRCLE_CENTER[2])] if "cx" in qs else None)
                payload = circle_payload(
                    center=center,
                    radius=_f("r", CIRCLE_RADIUS) if "r" in qs else None,
                    points=int(_f("n", CIRCLE_POINTS)) if "n" in qs else None,
                    plane=qs.get("plane", ["xy"])[0],
                    with_fk=True)
                return self._send_json(payload)
            return self._send_json({"error": "Not Found"}, 404)
        except (ValueError, KeyError) as e:
            return self._send_json({"error": f"参数错误: {e}"}, 400)
        except Exception as e:  # noqa: BLE001
            return self._send_json({"error": f"服务器错误: {e}"}, 500)

    def do_POST(self):
        try:
            parsed = urlparse(self.path)
            data = self._read_json()
            if parsed.path == "/api/ik":
                if "position" not in data:
                    return self._send_json({"error": "缺少 position"}, 400)
                return self._send_json(ik_payload(
                    data["position"], data.get("orientation"),
                    data.get("initial")))
            if parsed.path == "/api/trajectory":
                if "from" not in data or "to" not in data:
                    return self._send_json({"error": "缺少 from/to"}, 400)
                return self._send_json(trajectory_payload(
                    data["from"], data["to"],
                    profile=data.get("profile", "trapezoid"),
                    dt=data.get("dt", DT),
                    with_fk=bool(data.get("with_fk", False))))
            if parsed.path == "/api/demo/circle":
                return self._send_json(circle_payload(
                    center=data.get("center"),
                    radius=data.get("radius"),
                    orientation=data.get("orientation"),
                    points=data.get("points"),
                    dt=data.get("dt", DT),
                    with_fk=bool(data.get("with_fk", True)),
                    initial=data.get("initial"),
                    plane=data.get("plane", "xy")))
            if parsed.path == "/api/random_target":
                return self._send_json(random_target_payload(
                    current=data.get("current"), seed=data.get("seed")))
            return self._send_json({"error": "Not Found"}, 404)
        except (ValueError, KeyError) as e:
            return self._send_json({"error": f"参数错误: {e}"}, 400)
        except Exception as e:  # noqa: BLE001
            return self._send_json({"error": f"服务器错误: {e}"}, 500)


def find_port(start=8000, end=8010):
    import socket
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"端口 {start}-{end} 均被占用")


def main():
    # 控制台可能为 cp1252：统一改为 UTF-8 且永不抛编码异常，防止日志打印崩溃
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    port = int(sys.argv[1]) if len(sys.argv) > 1 else find_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"[web_server] 服务已启动: {url}  (Ctrl+C 退出)")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web_server] 已退出")


if __name__ == "__main__":
    main()
