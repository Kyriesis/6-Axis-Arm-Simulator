"""trajectory.py — 关节空间轨迹规划（点到点 + 多轴时间同步）

核心思想与 Dummy-Robot 固件一致（见仓库 README"控制算法"一节）：
    收到目标关节角后，计算 6 个轴的差分角度，找出"最难走的轴"
    （按比例换算后耗时最长者），以它的运动时间作为整段轨迹的基准时间，
    其余各轴按同一时刻缩放自己的速度/加速度，从而保证 6 轴同时到达、
    轨迹同步。这正是后续同时驱动 6 个步进电机（STEP/DIR 脉冲）时
    所需要的脉冲时序逻辑。

每个单轴速度曲线支持两种：
    - "trapezoid": 梯形速度曲线（匀加速-匀速-匀减速）
    - "scurve":     S 曲线（加速度按余弦渐变，速度曲线呈 S 形，冲击更小）

输出为固定采样周期 dt 的 (N, 6) 关节角序列，以及速度/加速度序列。
"""
import numpy as np


def _profile_time(delta, v_max, a_max, profile="trapezoid"):
    """计算单轴走完 delta（度）所需的最短时间及峰值速度。

    返回 (T, v_peak)。若达不到 v_max 则退化为纯加减速三角形曲线。
    S 曲线的 a_max 指加速度峰值（平均加速度为其一半），加速时间加倍。
    """
    delta = abs(delta)
    if delta < 1e-12:
        return 0.0, 0.0

    if profile == "scurve":
        t_acc_full = 2.0 * v_max / a_max        # 加速到 v_max 的时间
        d_acc = 0.5 * v_max * t_acc_full        # 单段加速位移（平均加速度 a_max/2）
    else:
        t_acc_full = v_max / a_max
        d_acc = 0.5 * a_max * t_acc_full ** 2

    if delta >= 2 * d_acc:  # 能达到最高速
        T = 2 * t_acc_full + (delta - 2 * d_acc) / v_max
        return T, v_max
    else:  # 三角形曲线
        if profile == "scurve":
            t_acc = np.sqrt(2.0 * delta / a_max)
            v_peak = 0.5 * a_max * t_acc
        else:
            t_acc = np.sqrt(delta / a_max)
            v_peak = a_max * t_acc
        T = 2 * t_acc
        return T, v_peak


def _synchronize(q0, q1, v_max, a_max, profile):
    """多轴时间同步：以各轴最短运动时间中的最大者作为公共时间 T，
    反算各轴在该时间下的峰值速度（按比例缩放），保证 6 轴同时到达。

    返回 (T, v_peak[6])，v_peak 带符号。
    """
    dq = q1 - q0
    T_axis = np.zeros(6)
    v_pk = np.zeros(6)
    for i in range(6):
        T_axis[i], _ = _profile_time(dq[i], v_max[i], a_max[i], profile)
    T = np.max(T_axis)
    if T < 1e-9:
        return 0.0, np.zeros(6)
    # 各轴在公共时间 T 内按比例缩放峰值速度（时间比 = 速度比的倒数）
    for i in range(6):
        if abs(dq[i]) < 1e-12:
            v_pk[i] = 0.0
        else:
            _, v_best = _profile_time(dq[i], v_max[i], a_max[i], profile)
            v_pk[i] = v_best * (T_axis[i] / T)
    return T, v_pk


def _trapezoid_velocity(t, T, v_peak, delta):
    """梯形速度曲线在某时刻的速度值（带符号），同时返回加速度。"""
    if T < 1e-12 or abs(delta) < 1e-12:
        return 0.0, 0.0
    s = np.sign(delta)
    D = abs(delta)
    # 位移约束 D = v_peak*(T - t_acc) -> t_acc = T - D/v_peak
    # （梯形: D = 2*0.5*a*t_acc^2 + v*(T-2*t_acc), v = a*t_acc，可化简得上式）
    t_acc = max(T - D / v_peak, 0.0)
    a = v_peak / t_acc if t_acc > 1e-12 else np.inf
    if t < t_acc:
        return s * a * t, s * a
    elif t < T - t_acc:
        return s * v_peak, 0.0
    else:
        return s * a * (T - t), s * (-a)


def _scurve_velocity(t, T, v_peak, delta):
    """S 曲线（加速度余弦渐变）速度与加速度。

    加速段: a(t) = a_pk * (1 - cos(2*pi*t/Ta)) / 2,  Ta 为加(减)速时间。
    速度为 a 的积分（正弦速度段），匀速段速度 = v_peak。
    """
    if T < 1e-12 or abs(delta) < 1e-12:
        return 0.0, 0.0
    s = np.sign(delta)
    D = abs(delta)
    t_acc = max(T - D / v_peak, 1e-9)
    a_pk = v_peak / (t_acc / 2.0)  # 平均加速度 = v_peak/t_acc = a_pk/2
    if t < t_acc:
        v = a_pk * (t / 2.0 - t_acc * np.sin(2 * np.pi * t / t_acc) / (4 * np.pi))
        a = a_pk * (1 - np.cos(2 * np.pi * t / t_acc)) / 2.0
        return s * v, s * a
    elif t < T - t_acc:
        return s * v_peak, 0.0
    else:
        tau = t - (T - t_acc)
        v_drop = a_pk * (tau / 2.0 - t_acc * np.sin(2 * np.pi * tau / t_acc) / (4 * np.pi))
        v = v_peak - v_drop
        a = -a_pk * (1 - np.cos(2 * np.pi * tau / t_acc)) / 2.0
        return s * v, s * a


def plan_trajectory(q0_deg, q1_deg, v_max_deg_s, a_max_deg_s2,
                    dt=0.01, profile="trapezoid"):
    """关节空间点到点轨迹规划（多轴时间同步 + 固定周期采样）。

    参数:
        q0_deg, q1_deg: 起点/终点关节角（度），长度 6
        v_max_deg_s:    各轴最大速度（度/秒），长度 6
        a_max_deg_s2:   各轴最大加速度（度/秒^2），长度 6
        dt:             采样周期（秒）
        profile:        "trapezoid"（梯形）或 "scurve"（S 曲线）

    返回:
        result: dict，包含
            t:      (N,) 时间序列
            q:      (N, 6) 关节角序列（度）
            qd:     (N, 6) 关节角速度（度/秒）
            qdd:    (N, 6) 关节角加速度（度/秒^2）
            T:      轨迹总时长（秒）
            v_peak: (6,) 各轴实际峰值速度（度/秒，同步后）
    """
    q0 = np.asarray(q0_deg, dtype=float)
    q1 = np.asarray(q1_deg, dtype=float)
    v_max = np.asarray(v_max_deg_s, dtype=float)
    a_max = np.asarray(a_max_deg_s2, dtype=float)
    assert q0.shape == q1.shape == v_max.shape == a_max.shape == (6,)

    T, v_peak = _synchronize(q0, q1, v_max, a_max, profile)
    if T < dt:  # 原地不动或极短
        T = dt
    n = int(np.ceil(T / dt)) + 1
    t = np.linspace(0.0, T, n)

    vel_fn = _trapezoid_velocity if profile == "trapezoid" else _scurve_velocity

    q = np.zeros((n, 6))
    qd = np.zeros((n, 6))
    qdd = np.zeros((n, 6))
    dq = q1 - q0
    for i in range(6):
        for k, tk in enumerate(t):
            v, a = vel_fn(tk, T, abs(v_peak[i]), dq[i])
            qd[k, i] = v
            qdd[k, i] = a
        # 角度由速度积分（累积梯形法），保证数值一致
        q[:, i] = q0[i] + np.concatenate([[0.0], np.cumsum(0.5 * (qd[:-1, i] + qd[1:, i]) * np.diff(t))])
    q[-1, :] = q1  # 末端精确对齐

    return {"t": t, "q": q, "qd": qd, "qdd": qdd, "T": T, "v_peak": v_peak}


def plan_waypoints(waypoints_deg, v_max_deg_s, a_max_deg_s2, dt=0.01,
                   profile="trapezoid"):
    """多点轨迹：依次经过各路标点，段间速度归零（MoveJ 风格）。

    参数:
        waypoints_deg: (M, 6) 路标关节角序列，M >= 2
    返回:
        与 plan_trajectory 相同的 dict（各段拼接，时间轴连续）。
    """
    wp = np.atleast_2d(np.asarray(waypoints_deg, dtype=float))
    assert wp.shape[1] == 6 and wp.shape[0] >= 2

    segments = []
    for k in range(wp.shape[0] - 1):
        seg = plan_trajectory(wp[k], wp[k + 1], v_max_deg_s, a_max_deg_s2,
                              dt=dt, profile=profile)
        segments.append(seg)

    t_all, q_all, qd_all, qdd_all = [], [], [], []
    t_off = 0.0
    for seg in segments:
        t_all.append(seg["t"] + t_off)
        q_all.append(seg["q"])
        qd_all.append(seg["qd"])
        qdd_all.append(seg["qdd"])
        t_off = t_all[-1][-1] + dt  # 段间留一个采样周期的停顿
    return {
        "t": np.concatenate(t_all),
        "q": np.vstack(q_all),
        "qd": np.vstack(qd_all),
        "qdd": np.vstack(qdd_all),
        "T": t_all[-1][-1],
        "v_peak": segments[-1]["v_peak"],
    }


def _catmull_rom(p0, p1, p2, p3, u):
    """非端点 Clamped Catmull-Rom 样条: 过 p1、p2，参数 u∈[0,1]。"""
    u2, u3 = u * u, u * u * u
    return (0.5 * ((2.0 * p1)
                   + (-p0 + p2) * u
                   + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
                   + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3))


def plan_path(waypoints_deg, v_max_deg_s, a_max_deg_s2, dt=0.02):
    """密集路标的平滑关节空间路径（Catmull-Rom 插值 + 分段速度限制）。

    适用于"末端沿连续路径运动"的场景（如圆形轨迹）：路标由离线 IK 生成，
    相邻点角度差小。与 plan_waypoints 的"每点减速到 0"不同，本函数用
    样条穿过所有路标，速度连续，不中途停顿。

    每段分配时间 T_seg = 该段按梯形曲线同步后的最短运动时间，
    全局按此时间轴均匀采样。

    参数:
        waypoints_deg: (M, 6) 关节角路标序列，M >= 2
        v_max_deg_s / a_max_deg_s2: 各轴速度/加速度上限（长度 6）
        dt: 采样周期（秒）
    返回:
        与 plan_trajectory 相同的 dict（qd/qdd 为数值微分）。
    """
    wp = np.atleast_2d(np.asarray(waypoints_deg, dtype=float))
    assert wp.shape[1] == 6 and wp.shape[0] >= 2
    v_max = np.asarray(v_max_deg_s, dtype=float)
    a_max = np.asarray(a_max_deg_s2, dtype=float)

    # 每段按梯形曲线同步得到的最短时间
    T_seg = np.zeros(wp.shape[0] - 1)
    for k in range(len(T_seg)):
        T_seg[k], _ = _synchronize(wp[k], wp[k + 1], v_max, a_max, "trapezoid")
    T_knots = np.concatenate([[0.0], np.cumsum(T_seg)])
    T_total = T_knots[-1]
    if T_total < 1e-9:
        T_total = dt
    n = int(np.ceil(T_total / dt)) + 1
    t = np.linspace(0.0, T_total, n)

    q = np.zeros((n, 6))
    for k in range(n):
        tk = t[k]
        seg = min(np.searchsorted(T_knots, tk, side="right") - 1, len(T_seg) - 1)
        seg = max(seg, 0)
        u = (tk - T_knots[seg]) / T_seg[seg] if T_seg[seg] > 1e-12 else 0.0
        u = min(max(u, 0.0), 1.0)
        i1, i2 = seg, seg + 1
        i0 = max(i1 - 1, 0)
        i3 = min(i2 + 1, wp.shape[0] - 1)
        for j in range(6):
            q[k, j] = _catmull_rom(wp[i0, j], wp[i1, j], wp[i2, j], wp[i3, j], u)

    # 数值微分求速度/加速度（供检查有界性）
    qd = np.zeros_like(q)
    qd[1:-1] = (q[2:] - q[:-2]) / (2 * dt)
    qd[0] = (q[1] - q[0]) / dt
    qd[-1] = (q[-1] - q[-2]) / dt
    qdd = np.zeros_like(q)
    qdd[1:-1] = (q[2:] - 2 * q[1:-1] + q[:-2]) / dt ** 2

    return {"t": t, "q": q, "qd": qd, "qdd": qdd, "T": T_total,
            "v_peak": np.abs(qd).max(axis=0)}
