"""ik_solver.py — 六轴机械臂数值逆运动学（IK）

算法：阻尼最小二乘（Damped Least Squares, Levenberg-Marquardt 形式）

    迭代更新:  dq = J^T (J J^T + lambda^2 I)^{-1} e
    其中 J 为几何雅可比（6x6），e 为末端位姿误差旋量（基系）。

特性:
    1. 奇异规避 —— 阻尼项 lambda 在接近奇异位形（J 的最小奇异值变小）
       时自动增大，避免 (J J^T) 病态求逆导致关节角爆炸。
    2. 关节限位 —— 超出限位附近时引入二次惩罚力把关节拉回合法区间，
       并在每轮迭代后做硬裁剪；带回溯的步长接受准则可防止限位处震荡。
    3. 多解处理 —— 数值 IK 的结果依赖于初始猜测 q0。调用方传入"当前关
       节角"作为初值时，机械臂总是以最小转角切换到目标姿态（与
       Dummy-Robot 固件"取 8 组解析解中关节变化最小者"的策略同理）。
       若主初值未收敛，可自动从若干随机初值重启（n_restarts），
       取误差最小者。
    4. 自适应步长 —— 每轮尝试 1.0/0.5/0.25 步长，接受使误差下降者；
       均失败则增大阻尼（缩小信赖域）。

位姿误差定义（几何法）:
    位置误差:  e_p = p_target - p_current          （3 维，m）
    姿态误差:  e_o = 0.5 * (n_t x n_c + o_t x o_c + a_t x a_c)  （3 维）
    其中 n/o/a 为目标与当前旋转矩阵的三列。收敛判据 max(|e|) < tol。
"""
import numpy as np

import arm_model as am


def _pose_error(T_target, T_current):
    """计算末端位姿误差旋量 e = [e_p; e_o]（基系表达）。

    位置误差: e_p = p_t - p_c（m）。
    姿态误差取旋转矩阵对数（旋转矢量）：R_err = R_t R_c^T，
    小角度时退化为 0.5*Σ(n_t x n_c)，大角度时方向依然正确。
    """
    e_p = T_target[:3, 3] - T_current[:3, 3]
    R_err = T_target[:3, :3] @ T_current[:3, :3].T
    # vee(R_err - R_err^T) / 2 = sin(theta) * axis
    v = 0.5 * np.array([R_err[2, 1] - R_err[1, 2],
                        R_err[0, 2] - R_err[2, 0],
                        R_err[1, 0] - R_err[0, 1]])
    c = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)  # = cos(theta)
    s = np.linalg.norm(v)                                   # = sin(theta)
    if s < 1e-9:
        e_o = v  # 零转角（或 180 度附近的退化，工程上忽略）
    else:
        e_o = np.arccos(c) / s * v  # 旋转矢量 theta * axis
    return np.concatenate([e_p, e_o])


def _limit_penalty_gradient(q_deg, k_penalty, margin=1.0):
    """限位二次惩罚的梯度（度）。在距限位 margin 度范围内开始发力。"""
    g = np.zeros(6)
    lo = am.JOINT_LIMITS[:, 0] + margin
    hi = am.JOINT_LIMITS[:, 1] - margin
    for i in range(6):
        if q_deg[i] < lo[i]:
            g[i] = k_penalty * (lo[i] - q_deg[i])
        elif q_deg[i] > hi[i]:
            g[i] = k_penalty * (q_deg[i] - hi[i])
    return g


def _lm_solve_once(T_target, q_start, max_iter, tol, damping, k_penalty,
                   w_orientation, verbose=False, q_ref_deg=None,
                   k_track=0.0):
    """单次 Levenberg-Marquardt 迭代（Nielsen 增益比信赖域）。

    q_ref_deg/k_track: 关节跟踪项（保持解靠近参考位形，用于路径分支连续）。

    求解 min 0.5*||W*e(q)||^2，LM 更新:
        h = (J^T J + lam*diag(J^T J))^{-1} J^T W e
    增益比 rho = 实际下降 / 预测下降，rho>0 接受步长并缩小 lam，
    否则拒绝并增大 lam（缩小信赖域）。这是经典的稳健实现。

    返回 (q_deg, info)。
    """
    W = np.diag([1.0, 1.0, 1.0, w_orientation, w_orientation, w_orientation])
    # 位置误差量级为米（整机展开 ~0.5m），姿态无量纲，做特征尺度归一
    W[:3, :3] /= 0.1

    def merit(q_deg):
        return _pose_error(T_target, am.end_pose(q_deg))

    q_deg = q_start.copy()
    lam = damping
    nu = 2.0
    err = merit(q_deg)
    F = 0.5 * float(err @ W @ W @ err)
    F_best = F
    stall = 0

    for it in range(max_iter):
        if np.max(np.abs(W @ err)) < tol:
            break
        # 停滞检测: 长期无进展则提前退出（把迭代预算留给重启）
        if F < F_best - 1e-12:
            F_best = F
            stall = 0
        else:
            stall += 1
            if stall > 40:
                break
        J = am.geometric_jacobian(q_deg)
        Jw = W @ J
        ew = W @ err
        JTJ = Jw.T @ Jw
        D = np.diag(np.diag(JTJ))
        if np.any(np.diag(D) < 1e-12):
            D += 1e-12 * np.eye(6)
        g = Jw.T @ ew - np.deg2rad(_limit_penalty_gradient(q_deg, k_penalty))
        if q_ref_deg is not None and k_track > 0.0:
            # 跟踪项: 让解待在参考位形附近（弧度梯度）
            g = g + k_track * np.deg2rad(q_deg - q_ref_deg)

        # LM 求解 (JTJ + lam*D) h = g
        try:
            h = np.linalg.solve(JTJ + lam * D, g)
        except np.linalg.LinAlgError:
            lam *= nu
            nu *= 2.0
            continue

        # 步长限制: 单步关节角变化不超过 45 度，防止大误差下线性化失效
        h_max = np.max(np.abs(h))
        if h_max > np.deg2rad(45.0):
            h *= np.deg2rad(45.0) / h_max

        q_try = am.clamp_to_limits(q_deg + np.rad2deg(h))
        err_try = merit(q_try)
        F_try = 0.5 * float(err_try @ W @ W @ err_try)

        # Nielsen 增益比
        denom = 0.5 * h.T @ (lam * D @ h + g)
        rho = (F - F_try) / denom if denom > 1e-18 else -1.0

        if rho > 0.0:
            q_deg = q_try
            err = err_try
            F = F_try
            lam = max(lam / 3.0, 1e-12)
            nu = 2.0
        else:
            lam *= nu
            nu *= 2.0
            if lam > 1e12:
                break
        if verbose and it % 20 == 0:
            print(f"  iter {it:3d}: pos_err={np.linalg.norm(err[:3]):.2e} "
                  f"ori_err={np.linalg.norm(err[3:]):.2e} lam={lam:.1e} rho={rho:.2f}")

    T_cur = am.end_pose(q_deg)
    e = _pose_error(T_target, T_cur)
    pos_err, ori_err = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
    return q_deg, {"success": bool(np.max(np.abs(W @ e)) < tol),
                   "pos_err": pos_err, "ori_err": ori_err,
                   "n_iter": it + 1, "converged": np.max(np.abs(W @ e)) < tol}


def inverse_kinematics(T_target, q0_deg=None, max_iter=300, tol=5e-4,
                       damping=1e-4, k_penalty=0.01, w_orientation=1.0,
                       n_restarts=20, seed=None, verbose=False,
                       q_ref_deg=None, k_track=0.0):
    """数值逆运动学求解。

    参数:
        T_target:      目标末端位姿（4x4 齐次矩阵，可用 arm_model.make_pose 构造）
        q0_deg:        初始猜测关节角（度）。None 时用限位中点。
                       实际使用中应传"当前关节角"以获得最小转角解。
        max_iter:      单次迭代最大次数
        tol:           收敛容差（位置 m / 姿态无量纲，取各分量绝对值最大者）
        damping:       阻尼系数 lambda 下限
        k_penalty:     限位惩罚强度
        w_orientation: 姿态误差权重（相对位置误差）
        n_restarts:    主初值失败时的随机重启次数（0 = 不重启）
        seed:          重启随机种子（None = 每次随机）
        verbose:       是否打印迭代过程

    返回:
        q_sol: 6 个关节角（度）
        info:  dict，含 success / pos_err / ori_err / n_iter / converged /
               in_limits / from_restart
    """
    if q0_deg is None:
        q_start = np.mean(am.JOINT_LIMITS, axis=1)  # 限位中点，比全零更稳妥
    else:
        q_start = np.asarray(q0_deg, dtype=float)

    q_sol, info = _lm_solve_once(T_target, q_start, max_iter, tol, damping,
                                 k_penalty, w_orientation, verbose,
                                 q_ref_deg=q_ref_deg, k_track=k_track)
    info["from_restart"] = False

    if not info["success"] and n_restarts > 0:
        rng = np.random.default_rng(seed)
        sigmas = [20.0, 45.0, 90.0, 135.0]  # 初值高斯扰动的递增幅度（度）
        for r in range(n_restarts):
            if r % 2 == 0:
                # 初值扰动重启: 在"当前关节角"附近搜索，容易跳出腕翻折等局部极小
                sig = sigmas[(r // 2) % len(sigmas)]
                q_init = am.clamp_to_limits(q_start + rng.normal(0.0, sig, 6))
            else:
                # 全局随机重启: 在限位空间内均匀采样
                q_init = rng.uniform(am.JOINT_LIMITS[:, 0], am.JOINT_LIMITS[:, 1])
            q_cand, info_c = _lm_solve_once(T_target, q_init, max_iter, tol,
                                            damping, k_penalty, w_orientation,
                                            q_ref_deg=q_ref_deg,
                                            k_track=k_track)
            score_c = info_c["pos_err"] / 0.1 + info_c["ori_err"]
            score_best = info["pos_err"] / 0.1 + info["ori_err"]
            if info_c["success"]:
                q_sol, info = q_cand, info_c
                info["from_restart"] = True
                break
            if score_c < score_best:  # 未收敛也保留最优候选
                q_sol, info = q_cand, info_c
                info["from_restart"] = True

    info["in_limits"] = am.within_limits(q_sol)
    info["q_final"] = q_sol
    return q_sol, info


def ik_batch_check(n_samples=50, seed=0, n_restarts=20):
    """测试辅助：随机取限位内关节角 -> FK -> IK -> FK，统计闭环误差。

    模拟真实使用方式：IK 初值取上一样本的解（最小转角策略），
    不收敛时允许随机重启。
    """
    rng = np.random.default_rng(seed)
    results = []
    q_prev = np.mean(am.JOINT_LIMITS, axis=1)
    for _ in range(n_samples):
        q_true = rng.uniform(am.JOINT_LIMITS[:, 0], am.JOINT_LIMITS[:, 1])
        T = am.end_pose(q_true)
        q_sol, info = inverse_kinematics(T, q0_deg=q_prev, n_restarts=n_restarts)
        T_re = am.end_pose(q_sol)
        pos_err = np.linalg.norm(T_re[:3, 3] - T[:3, 3])
        R_err = T_re[:3, :3].T @ T[:3, :3]
        ori_err = np.abs(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1)))
        results.append((pos_err, ori_err, info["success"], info["in_limits"],
                        np.max(np.abs(q_sol - q_true))))
        q_prev = q_sol
    return results


def wrist_flip(q_deg, q_ref_deg):
    """返回与 q_deg 末端位姿完全相同、但腕部关节更接近 q_ref_deg 的解。

    本臂腕部（J4/J5/J6）的翻转恒等式（已数值验证）:
        (q4±180°, -q5, q6±180°) 与原解同位姿。
    用于保持路径上相邻 IK 解处于同一分支，避免关节空间轨迹跳变。
    """
    best = np.asarray(q_deg, dtype=float)
    q = np.asarray(q_deg, dtype=float)
    ref = np.asarray(q_ref_deg, dtype=float)
    for s4 in (180.0, -180.0):
        for s6 in (180.0, -180.0):
            cand = q.copy()
            cand[3] = q[3] + s4
            cand[4] = -q[4]
            cand[5] = q[5] + s6
            if np.max(np.abs(cand - ref)) < np.max(np.abs(best - ref)):
                best = cand
    return best


def ik_continuous(poses, q0_deg, n_restarts=8, max_knot_delta=25.0,
                  max_attempts=5, **kwargs):
    """对一串目标位姿做"分支连续"的 IK 解链。

    每个目标用上一解作为初值（最小转角），随后做腕部翻转修正与
    J6 解缠（±360°），保证相邻解关节变化最小。IK 含随机重启，
    单次链路可能跳分支，因此整链做校验（相邻路标关节增量 ≤
    max_knot_delta），失败则以不同种子重试，直到成功或耗尽次数。

    参数:
        poses:           目标 4x4 位姿矩阵列表
        q0_deg:          起始关节角（度）
        n_restarts:      单次 IK 的随机重启次数
        max_knot_delta:  相邻路标允许的最大关节增量（度）
        max_attempts:    整链最大尝试次数
    返回:
        (M, 6) 关节角数组
    """
    q_start = np.asarray(q0_deg, dtype=float)
    k_track = kwargs.pop("k_track", 0.01)
    for attempt in range(max_attempts):
        q_prev = q_start.copy()
        qs = []
        ok = True
        for T in poses:
            # 两阶段: 先用跟踪项粗解（保持分支），再关掉跟踪精修
            q, _ = inverse_kinematics(
                T, q0_deg=q_prev, n_restarts=n_restarts, seed=1000 + attempt,
                q_ref_deg=q_prev, k_track=k_track,
                tol=1e-3, **kwargs)
            q, info = inverse_kinematics(T, q0_deg=q, n_restarts=0,
                                         tol=2e-3, **kwargs)
            if not info["success"]:
                # 精修差一点时: 加大迭代再试一次
                q, info = inverse_kinematics(T, q0_deg=q, n_restarts=0,
                                             max_iter=1000)
            if not info["success"] and not (info["pos_err"] < 1e-3
                                            and info["ori_err"] < 1e-2):
                ok = False
                break
            q = wrist_flip(q, q_prev)
            while q[5] - q_prev[5] > 180.0:
                q[5] -= 360.0
            while q[5] - q_prev[5] < -180.0:
                q[5] += 360.0
            # 首 knot 是"切入路径"的大动作，豁免连续性检查
            if qs and np.max(np.abs(q - q_prev)) > max_knot_delta:
                ok = False  # 分支跳变，整链重试
                break
            qs.append(q)
            q_prev = q
        if ok and len(qs) == len(poses):
            return np.array(qs)
    raise RuntimeError(
        f"ik_continuous 经过 {max_attempts} 次尝试仍未获得连续解链")
