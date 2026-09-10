"""simulation.py — 虚拟六轴机械臂 3D 动画仿真（主程序）

功能:
    1. 规划一条多路标关节空间轨迹（模拟 MoveJ 指令序列）；
    2. matplotlib 3D 动画展示机械臂运动；
    3. 下方 6 条曲线实时显示 6 个"虚拟电机"的角位置，
       相当于 6 路 STEP/DIR 脉冲计数器（度 -> 脉冲的映射仅差一个比例），
       直观展示多轴时间同步轨迹的同步性。

运行:
    python simulation.py            # 弹出动画窗口，跑完 MAX_FRAMES 帧后自动退出
    python simulation.py --save     # 无窗口（Agg 后端），渲染若干帧保存 PNG 截图

截图保存在本目录 snapshot.png。
"""
import os
import sys

# 兼容 safe_path 模式的 Python（cwd 不在 sys.path 中）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import arm_model as am
import trajectory as traj

# ---------------- 仿真参数 ----------------
DT = 0.02                 # 轨迹采样周期（秒）
MAX_FRAMES = 200          # 动画帧数上限（自动退出用）
PULSES_PER_DEG = 100.0    # 虚拟电机细分：每度脉冲数（仅用于"脉冲计数"显示）

# 各轴速度/加速度限制（度/秒，度/秒^2）——虚拟值，可按需调整
V_MAX = np.array([120.0, 90.0, 90.0, 180.0, 180.0, 360.0])
A_MAX = np.array([300.0, 240.0, 240.0, 540.0, 540.0, 900.0])

# 6 个路标点（度），覆盖大臂俯仰、肘部、腕部各种动作
WAYPOINTS = np.array([
    [0.0, 0.0, 90.0, 0.0, 0.0, 0.0],
    [45.0, -30.0, 60.0, 30.0, 45.0, 90.0],
    [45.0, 20.0, 120.0, -60.0, -30.0, 180.0],
    [-60.0, 10.0, 70.0, 60.0, 60.0, -90.0],
    [-60.0, -40.0, 45.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 90.0, 0.0, 0.0, 0.0],
])


def build_trajectory():
    """规划演示轨迹（S 曲线加减速 + 多轴同步）。"""
    # 限位保护
    wp = np.clip(WAYPOINTS, am.JOINT_LIMITS[:, 0], am.JOINT_LIMITS[:, 1])
    return traj.plan_waypoints(wp, V_MAX, A_MAX, dt=DT, profile="scurve")


def simulate_motor_pulses(q_deg):
    """把关节角序列换算成 6 路虚拟电机的脉冲计数（模拟 STEP 脉冲累加）。"""
    return q_deg * PULSES_PER_DEG


def run(save_snapshot=False):
    import matplotlib
    # 中文字体（Windows 自带的微软雅黑/黑体），并修复负号显示
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                              "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    if save_snapshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    data = build_trajectory()
    if save_snapshot:
        snapshot_path = render_snapshot(data)
        print(f"[simulation] 已保存截图: {snapshot_path}")
        n_frames = min(len(data["t"]), MAX_FRAMES)
        print(f"[simulation] 轨迹时长 {data['T']:.2f} s, 共 {n_frames} 帧, "
              f"采样周期 {DT*1000:.0f} ms")
        return snapshot_path

    t, q = data["t"], data["q"]
    pulses = simulate_motor_pulses(q)
    n_frames = min(len(t), MAX_FRAMES)

    # ---------- 画布布局 ----------
    fig = plt.figure(figsize=(12, 8), dpi=110)
    fig.suptitle("Dummy-Robot 虚拟六轴机械臂仿真（稚辉君开源项目参数）")
    ax3d = fig.add_subplot(2, 1, 1, projection="3d")
    ax_j = [fig.add_subplot(2, 6, 7 + i) for i in range(6)]

    # ---------- 3D 机械臂 ----------
    reach = (am.ARM_DIMS["L_BASE"] + am.ARM_DIMS["D_BASE"]
             + am.ARM_DIMS["L_ARM"] + am.ARM_DIMS["L_FOREARM"]
             + am.ARM_DIMS["L_WRIST"] + am.ARM_DIMS["D_ELBOW"])
    ax3d.set_xlim(-reach, reach)
    ax3d.set_ylim(-reach, reach)
    ax3d.set_zlim(0.0, reach + 0.05)
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    arm_line, = ax3d.plot([], [], [], "o-", color="#1f77b4", lw=3, ms=6)
    tip_trace, = ax3d.plot([], [], [], "-", color="#d62728", lw=1, alpha=0.6)
    trace_pts = []

    # ---------- 6 路虚拟电机曲线 ----------
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    motor_lines = []
    for i in range(6):
        ax = ax_j[i]
        ax.set_xlim(0, max(t[-1], 1.0))
        ax.set_ylim(np.min(pulses[:, i]) * 1.1 - 1, np.max(pulses[:, i]) * 1.1 + 1)
        ax.set_title(f"J{i+1} 电机脉冲", fontsize=9)
        ax.grid(True, alpha=0.3)
        ln, = ax.plot([], [], color=colors[i], lw=1.2)
        motor_lines.append(ln)

    def update(frame):
        k = min(frame, len(t) - 1)
        # 3D 机械臂
        pts = am.joint_positions(q[k])
        arm_line.set_data(pts[:, 0], pts[:, 1])
        arm_line.set_3d_properties(pts[:, 2])
        trace_pts.append(pts[-1])
        tr = np.array(trace_pts)
        tip_trace.set_data(tr[:, 0], tr[:, 1])
        tip_trace.set_3d_properties(tr[:, 2])
        # 电机曲线
        for i in range(6):
            motor_lines[i].set_data(t[:k + 1], pulses[:k + 1, i])
        ax3d.set_title(f"t = {t[k]:.2f} s   帧 {k}/{n_frames-1}", fontsize=10)
        return [arm_line, tip_trace, *motor_lines]

    ani = FuncAnimation(fig, update, frames=n_frames, interval=DT * 1000,
                        blit=False, repeat=False)

    if save_snapshot:
        # 截图已在上方提前返回，此处不会执行
        return None
    else:
        def on_close(_):
            plt.close(fig)
        print(f"[simulation] 动画共 {n_frames} 帧, 播完自动退出。")
        plt.show()


def render_snapshot(data, path="snapshot.png"):
    """用 Agg 后端渲染一张静态截图: 4 个关键相位的 3D 姿态 + 6 路电机曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                              "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    t, q = data["t"], data["q"]
    pulses = simulate_motor_pulses(q)
    reach = (am.ARM_DIMS["L_BASE"] + am.ARM_DIMS["D_BASE"]
             + am.ARM_DIMS["L_ARM"] + am.ARM_DIMS["L_FOREARM"]
             + am.ARM_DIMS["L_WRIST"] + am.ARM_DIMS["D_ELBOW"])

    fig = plt.figure(figsize=(14, 9), dpi=110)
    fig.suptitle("Dummy-Robot 虚拟六轴机械臂仿真（稚辉君开源项目参数）")
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 2])

    # 上排: 4 个关键相位的 3D 姿态 + 到该时刻的末端轨迹
    phases = [0, len(t) // 3, 2 * len(t) // 3, len(t) - 1]
    for col, k in enumerate(phases):
        ax = fig.add_subplot(gs[0, col], projection="3d")
        pts = am.joint_positions(q[k])
        trace = np.array([am.joint_positions(q[j])[-1] for j in range(0, k + 1, 2)])
        ax.plot(trace[:, 0], trace[:, 1], trace[:, 2],
                "-", color="#d62728", lw=1, alpha=0.7)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "o-", color="#1f77b4",
                lw=3, ms=6)
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_zlim(0.0, reach + 0.05)
        ax.set_title(f"t = {t[k]:.2f} s", fontsize=10)
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        if col == 0:
            ax.set_zlabel("Z", fontsize=8)
        ax.view_init(elev=25, azim=35)

    # 下排: 6 路虚拟电机脉冲曲线
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    gs2 = gs[1, :].subgridspec(1, 6)
    for i in range(6):
        ax = fig.add_subplot(gs2[0, i])
        ax.plot(t, pulses[:, i], color=colors[i], lw=1.2)
        ax.set_title(f"J{i+1} 电机脉冲", fontsize=9)
        ax.set_xlabel("t (s)", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


if __name__ == "__main__":
    save = "--save" in sys.argv
    run(save_snapshot=save)
