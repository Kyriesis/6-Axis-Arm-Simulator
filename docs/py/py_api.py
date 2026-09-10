"""静态站点 API：与 web_server.py 的 payload 函数一一对应。

在浏览器中由 Pyodide（WASM Python）加载，所有函数返回 JSON 字符串，
供前端 JS 直接 JSON.parse。业务逻辑与 web_server.py 保持同步。
"""
import json

import numpy as np

import arm_model as am
import ik_solver
import trajectory as traj

# ---------------- 仿真参数（与 web_server.py / simulation.py 一致） ----------------
V_MAX = np.array([120.0, 90.0, 90.0, 180.0, 180.0, 360.0])    # 度/秒
A_MAX = np.array([300.0, 240.0, 240.0, 540.0, 540.0, 900.0])  # 度/秒^2
DT = 0.02                                                      # 默认采样周期（秒）

CIRCLE_CENTER = [0.15, 0.0, 0.12]   # 圆心（m）
CIRCLE_RADIUS = 0.06                # 半径（m）
CIRCLE_RPY = [180.0, 0.0, 0.0]      # 工具姿态（度，ZYX 欧拉角）
CIRCLE_POINTS = 61                  # 离线 IK 路标数


def _jsonify(payload):
    return json.dumps(payload, ensure_ascii=False)


# ---------------- 业务逻辑（与 web_server.py 相同） ----------------
def fk_payload(q_deg):
    """FK 结果字典：各关节坐标 + 末端位姿。"""
    q_deg = am.clamp_to_limits(np.asarray(q_deg, dtype=float))
    transforms = am.forward_kinematics(q_deg)
    joints = [T[:3, 3].tolist() for T in transforms]
    pose = am.pose_to_xyz_rpy(transforms[6])
    return {
        "q": q_deg.tolist(),
        "joints": joints,
        "pose": {"position": pose[:3].tolist(),
                 "rpy_deg": pose[3:].tolist()},
    }


def ik_payload(position, orientation=None, initial=None):
    """IK 求解。orientation 为 ZYX 欧拉角（度），缺省保持当前姿态。"""
    position = np.asarray(position, dtype=float)
    if orientation is not None:
        T_target = am.make_pose(*position, *orientation)
    else:
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
        "pos_err": float(info["pos_err"]),
        "ori_err": float(info["ori_err"]),
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
    """圆形末端轨迹：离线 IK 连续解链 + Catmull-Rom 平滑路径。"""
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


# ---------------- JS 调用入口（参数/返回值均为 JSON 字符串） ----------------
def limits_json(_=""):
    return _jsonify({"limits": am.JOINT_LIMITS.tolist(), "dims": am.ARM_DIMS})


def fk_json(q_csv):
    q = [float(x) for x in str(q_csv).split(",")]
    return _jsonify(fk_payload(q))


def ik_json(body):
    b = json.loads(body)
    return _jsonify(ik_payload(b["position"], b.get("orientation"),
                               b.get("initial")))


def trajectory_json(body):
    b = json.loads(body)
    return _jsonify(trajectory_payload(
        b["from"], b["to"], b.get("profile", "trapezoid"),
        b.get("dt", DT), b.get("with_fk", False)))


def circle_json(body):
    b = json.loads(body)
    return _jsonify(circle_payload(
        b.get("center"), b.get("radius"), b.get("orientation"),
        b.get("points"), b.get("dt", DT), b.get("with_fk", True),
        b.get("initial"), b.get("plane", "xy")))


def random_json(body):
    b = json.loads(body)
    return _jsonify(random_target_payload(b.get("current")))
