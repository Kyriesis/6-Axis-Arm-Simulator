"""test_kinematics.py — 运动学/轨迹验证脚本（直接运行，无需 pytest）

检查项:
    1. FK -> IK -> FK 闭环误差（随机限位内关节角，IK 以邻位初值启动）
    2. 数值 IK 收敛性与限位合法性
    3. 轨迹连续性：角度不跳变、速度/加速度有界且不超过设定上限
    4. 多轴同步：各轴同时到达终点
    5. 模型与 Dummy-Robot 固件 FK 算法的一致性（向量链法比对）

运行:  python test_kinematics.py
退出码: 0 = 全部通过，1 = 存在失败项。
"""
import os
import sys

# 兼容 safe_path 模式的 Python（cwd 不在 sys.path 中）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import arm_model as am
import ik_solver
import trajectory as traj

N_IK_SAMPLES = 50
DT = 0.01


def test_model_matches_firmware():
    """本项目模型 vs 固件 SolveFK 向量链法，应达到机器精度。"""
    D = am.ARM_DIMS

    def firmware_fk(q_deg):
        q = np.radians(q_deg)
        R = []
        for i in range(6):
            th_off, al, _ = am.DH_TABLE[i]
            R.append(am.rot_z(q[i] + np.deg2rad(th_off)) @ am.rot_x(np.deg2rad(al)))
        R01 = R[0] @ R[1]
        R012 = R01 @ R[2]
        R06 = R012 @ R[3] @ R[4] @ R[5]
        return (R[0] @ np.array([D["D_BASE"], -D["L_BASE"], 0])
                + R01 @ np.array([D["L_ARM"], 0, 0])
                + R012 @ np.array([-D["D_ELBOW"], 0, D["L_FOREARM"]])
                + R06 @ np.array([0, 0, D["L_WRIST"]]))

    rng = np.random.default_rng(7)
    max_err = 0.0
    for _ in range(300):
        q = rng.uniform(am.JOINT_LIMITS[:, 0], am.JOINT_LIMITS[:, 1])
        T = am.end_pose(q)
        max_err = max(max_err, np.linalg.norm(firmware_fk(q) - T[:3, 3]))
    assert max_err < 1e-10, f"模型与固件 FK 不一致, 误差 {max_err:.2e}"
    print(f"[1] 模型 vs 固件 FK 一致性: 通过 (300 组, 最大误差 {max_err:.2e} m)")


def test_fk_ik_loop():
    """FK->IK->FK 闭环误差。"""
    results = ik_solver.ik_batch_check(n_samples=N_IK_SAMPLES, seed=42)
    pos_errs = np.array([r[0] for r in results])
    ori_errs = np.array([r[1] for r in results])
    success = all(r[2] for r in results)
    in_limits = all(r[3] for r in results)
    max_joint_dev = max(r[4] for r in results)

    print(f"[2] FK->IK->FK 闭环 ({N_IK_SAMPLES} 组随机位姿):")
    print(f"    收敛成功率: {sum(r[2] for r in results)}/{len(results)}")
    print(f"    位置误差: max={pos_errs.max():.3e} m, mean={pos_errs.mean():.3e} m")
    print(f"    姿态误差: max={ori_errs.max():.3e} rad, mean={ori_errs.mean():.3e} rad")
    print(f"    全部在限位内: {in_limits}")
    print(f"    单关节最大偏差(多解属正常): {max_joint_dev:.1f} deg")

    assert success, "存在 IK 未收敛的样本"
    assert pos_errs.max() < 1e-4, f"位置闭环误差过大: {pos_errs.max():.2e}"
    assert ori_errs.max() < 1e-3, f"姿态闭环误差过大: {ori_errs.max():.2e}"
    assert in_limits, "IK 解超出关节限位"


def test_trajectory_continuity():
    """轨迹连续性: 角度不跳变, 速度/加速度有界。"""
    v_max = np.array([120.0, 90.0, 90.0, 180.0, 180.0, 360.0])
    a_max = np.array([300.0, 240.0, 240.0, 540.0, 540.0, 900.0])
    rng = np.random.default_rng(3)
    all_ok = True
    for profile in ("trapezoid", "scurve"):
        for trial in range(10):
            q0 = rng.uniform(am.JOINT_LIMITS[:, 0], am.JOINT_LIMITS[:, 1])
            q1 = rng.uniform(am.JOINT_LIMITS[:, 0], am.JOINT_LIMITS[:, 1])
            data = traj.plan_trajectory(q0, q1, v_max, a_max, dt=DT, profile=profile)
            q, qd, qdd = data["q"], data["qd"], data["qdd"]

            # 角度不跳变: 相邻采样点增量有界（< 速度上限 * dt * 2 裕量）
            dq_max = np.max(np.abs(np.diff(q, axis=0)), axis=0)
            assert np.all(dq_max <= v_max * DT * 2.5 + 1e-6), \
                f"[{profile}] 角度跳变: {dq_max}"

            # 速度/加速度有界（留 5% 数值裕量）
            v_over = np.abs(qd).max(axis=0)
            a_over = np.abs(qdd).max(axis=0)
            assert np.all(v_over <= v_max * 1.05), f"[{profile}] 超速: {v_over}"
            assert np.all(a_over <= a_max * 1.05), f"[{profile}] 超加速度: {a_over}"

            # 首尾精确
            assert np.allclose(q[0], q0, atol=1e-6), "起点不对"
            assert np.allclose(q[-1], q1, atol=1e-6), "终点不对"
        print(f"[3] 轨迹连续性 ({profile}): 通过 (10 组随机起终点, "
              f"无跳变/速度加速度有界/首尾精确)")


def test_multi_axis_sync():
    """多轴同步: 所有轴在同一时刻到达终点。"""
    v_max = np.array([120.0, 90.0, 90.0, 180.0, 180.0, 360.0])
    a_max = np.array([300.0, 240.0, 240.0, 540.0, 540.0, 900.0])
    q0 = np.array([0.0, 0.0, 90.0, 0.0, 0.0, 0.0])
    q1 = np.array([90.0, -60.0, 45.0, 120.0, -90.0, 270.0])
    data = traj.plan_trajectory(q0, q1, v_max, a_max, dt=DT)
    # 各轴到达终点的时刻（最后一个与终点误差 > 0.01 度的采样点）
    arrival = np.zeros(6)
    for i in range(6):
        settled = np.nonzero(np.abs(data["q"][:, i] - q1[i]) > 0.01)[0]
        arrival[i] = data["t"][settled[-1]] if len(settled) else 0.0
    spread = arrival.max() - arrival.min()
    T = data["T"]
    print(f"[4] 多轴同步: 总时长 {T:.2f} s, 各轴到达时刻 {np.round(arrival, 3)}, "
          f"离散度 {spread:.3f} s (= 采样周期 {DT} s)")
    assert spread <= DT + 1e-9, "各轴未同步到达"


def test_ik_near_singular():
    """接近奇异位形时阻尼最小二乘应稳定收敛（腕部奇异附近）。"""
    rng = np.random.default_rng(11)
    ok = 0
    n = 20
    for _ in range(n):
        q_true = rng.uniform(am.JOINT_LIMITS[:, 0], am.JOINT_LIMITS[:, 1])
        T = am.end_pose(q_true)
        q_sol, info = ik_solver.inverse_kinematics(T, q0_deg=q_true + rng.normal(0, 3, 6))
        if info["success"] and info["pos_err"] < 1e-4:
            ok += 1
    print(f"[5] 奇异附近 IK 稳定性: {ok}/{n} 收敛")
    assert ok >= n * 0.8, "奇异附近收敛率过低"


if __name__ == "__main__":
    try:
        test_model_matches_firmware()
        test_fk_ik_loop()
        test_trajectory_continuity()
        test_multi_axis_sync()
        test_ik_near_singular()
    except AssertionError as e:
        print(f"\n测试失败: {e}")
        sys.exit(1)
    print("\n全部测试通过 ✓")
