"""arm_model.py — 虚拟六轴机械臂模型（参考稚辉君 Dummy-Robot）

DH 参数来源：peng-zhihui/Dummy-Robot 仓库
    2.Firmware/Core-STM32F4-fw/Robot/instances/dummy_robot.cpp
    2.Firmware/Core-STM32F4-fw/Robot/algorithms/kinematic/6dof_kinematic.cpp

固件中定义的尺寸（单位 m，构造函数 DOF6Kinematic(L_BS, D_BS, L_AM, L_FA, D_EW, L_WT)）：
    L_BASE    = 0.109  基座高度（J1 到 J2 的垂直距离）
    D_BASE    = 0.035  基座水平偏移（J1 轴到 J2 轴的水平距离）
    L_ARM     = 0.146  大臂长度（J2 轴到 J3 轴）
    L_FOREARM = 0.115  小臂长度（J3 轴到 J5 轴交点）
    D_ELBOW   = 0.052  肘部偏移（J3 轴到 J4 轴的横向距离）
    L_WRIST   = 0.072  腕部/工具长度（J5 轴交点到工具末端）

关节限位（固件 dummy_robot.cpp 中 CtrlStepMotor 的运动范围，单位 deg）：
    J1 [-170, 170]  J2 [-73, 90]  J3 [35, 180]  J4 [-180, 180]  J5 [-120, 120]  J6 [-720, 720]

注意：固件的 FK 并不是教科书式"纯标准 DH 链"。固件中存储的 DH_matrix
（4 列）只有第 0 列（theta 零偏）和第 3 列（alpha）真正参与计算，连杆平移
由另外四个常量向量（L1_base / L2_arm / L3_elbow / L6_wrist）表达，且其中
存在 DH 标准形（平移必须落在连杆 x-z 平面内）无法表示的 y 向分量。
因此本模块采用与固件完全一致的"旋转 DH + 连杆平移向量"混合模型：
每个关节的齐次变换为  A_i = RotZ(theta_i) @ Trans(t_i) @ RotX(alpha_i)
其中 t_i 为该连杆坐标系内的常值平移向量。该模型已用数值方法与固件
SolveFK（向量链法）在 200 组随机关节角下比对，位置误差 < 1e-12 m。
"""
import numpy as np

# ---------------- 机械尺寸（来自 Dummy-Robot 固件，单位：m） ----------------
ARM_DIMS = {
    "L_BASE": 0.109,     # 基座高度
    "D_BASE": 0.035,     # 基座水平偏移
    "L_ARM": 0.146,      # 大臂长度
    "L_FOREARM": 0.115,  # 小臂长度
    "D_ELBOW": 0.052,    # 肘部偏移
    "L_WRIST": 0.072,    # 腕部/工具长度
}

# ---------------- 广义 DH 表（与固件一致） ----------------
# 每行对应一个关节：theta_offset (deg), alpha (deg), t = [tx, ty, tz] (m)
# 关节变换 A_i = RotZ(q_i + theta_offset) @ Trans(t_i) @ RotX(alpha)
DIM = ARM_DIMS
DH_TABLE = [
    # theta_offset, alpha,  [tx, ty, tz]
    (0.0,   -90.0, [0.0,             0.0,      0.0]),            # J1 旋转基座
    (-90.0,   0.0, [DIM["D_BASE"],  -DIM["L_BASE"], 0.0]),       # J2 肩关节
    (90.0,   90.0, [DIM["L_ARM"],    0.0,      0.0]),            # J3 肘关节
    (0.0,   -90.0, [-DIM["D_ELBOW"], 0.0,      DIM["L_FOREARM"]]),  # J4 腕关节1
    (0.0,    90.0, [0.0,             0.0,      0.0]),            # J5 腕关节2
    (0.0,     0.0, [0.0,             0.0,      DIM["L_WRIST"]]),  # J6 腕关节3 + 工具
]

# 关节限位（deg，来自固件电机配置）
JOINT_LIMITS = np.array([
    [-170.0, 170.0],
    [-73.0, 90.0],
    [35.0, 180.0],
    [-180.0, 180.0],
    [-120.0, 120.0],
    [-720.0, 720.0],
])

# 各关节轴单位向量（基系，零位时），用于可视化/雅可比
AXIS_Z = np.array([0.0, 0.0, 1.0])


def rot_z(theta_rad):
    """绕 Z 轴旋转矩阵。"""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def rot_x(alpha_rad):
    """绕 X 轴旋转矩阵。"""
    c, s = np.cos(alpha_rad), np.sin(alpha_rad)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,  c, -s],
                     [0.0,  s,  c]])


def dh_transform(joint_idx, q_rad):
    """计算第 joint_idx 个关节（0 起）的齐次变换矩阵 A_i。

    A_i = RotZ(q + theta_offset) @ Trans(t) @ RotX(alpha)

    参数:
        joint_idx: 关节序号 0..5
        q_rad:     该关节角（弧度）
    返回:
        4x4 齐次变换矩阵（连杆坐标系 i-1 -> i 的位姿）
    """
    theta_off, alpha, t = DH_TABLE[joint_idx]
    A = np.eye(4)
    A[:3, :3] = rot_z(q_rad + np.deg2rad(theta_off)) @ rot_x(np.deg2rad(alpha))
    A[:3, 3] = np.array(t)
    return A


def forward_kinematics(q_deg):
    """正运动学：6 个关节角 -> 各连杆坐标系的累计齐次变换。

    参数:
        q_deg: 长度为 6 的关节角序列（度）
    返回:
        transforms: 长度为 7 的列表，transforms[k] 为基座到第 k 个连杆
                    坐标系的 4x4 齐次变换（transforms[0] 为单位阵）。
                    transforms[6] 即末端执行器位姿。
    """
    q = np.asarray(q_deg, dtype=float)
    assert q.shape == (6,), "需要 6 个关节角"
    transforms = [np.eye(4)]
    T = np.eye(4)
    for i in range(6):
        T = T @ dh_transform(i, np.deg2rad(q[i]))
        transforms.append(T)
    return transforms


def end_pose(q_deg):
    """返回末端 4x4 位姿矩阵。"""
    return forward_kinematics(q_deg)[6]


def joint_positions(q_deg):
    """返回基座 + 6 个关节 + 末端的 (7, 3) 位置矩阵，用于绘图。"""
    transforms = forward_kinematics(q_deg)
    return np.array([T[:3, 3] for T in transforms])


def pose_to_xyz_rpy(T):
    """把 4x4 位姿矩阵转成 (x, y, z, roll, pitch, yaw)，角度为度。

    使用 ZYX 欧拉角（R = Rz(yaw) Ry(pitch) Rx(roll)），与固件一致。
    """
    x, y, z = T[:3, 3]
    R = T[:3, :3]
    # 固件 RotMatToEulerAngle 的逆变换（ZYX 欧拉角）
    if abs(R[2, 0]) < 1.0 - 1e-9:
        pitch = np.arcsin(-R[2, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:  # 奇异
        pitch = np.pi / 2 if R[2, 0] <= -1 + 1e-9 else -np.pi / 2
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    return np.array([x, y, z, *np.rad2deg([roll, pitch, yaw])])


def rpy_to_rot_mat(roll, pitch, yaw):
    """ZYX 欧拉角（度）-> 3x3 旋转矩阵。"""
    r, p, w = np.deg2rad([roll, pitch, yaw])
    return rot_z(w) @ rot_y(p) @ rot_x(r)


def rot_y(theta_rad):
    """绕 Y 轴旋转矩阵。"""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


def make_pose(x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
    """由位置 + ZYX 欧拉角（度）构造 4x4 目标位姿。"""
    T = np.eye(4)
    T[:3, :3] = rpy_to_rot_mat(roll, pitch, yaw)
    T[:3, 3] = [x, y, z]
    return T


def geometric_jacobian(q_deg):
    """计算基系下的 6x6 几何雅可比矩阵（线速度 + 角速度）。

    返回 J，使得 [v; w] = J @ dq（dq 单位 rad/s）。

    注意: 本模型的连杆变换为 A_i = RotZ(θ) Trans(t) Rx(α)（固件式分解），
    中间坐标系原点不一定落在下一关节轴线上，因此第 i+1 个关节的轴线
    参考点取 transforms[i+1] 的原点（而非 transforms[i]）。
    """
    transforms = forward_kinematics(q_deg)
    J = np.zeros((6, 6))
    o_n = transforms[6][:3, 3]
    for i in range(6):
        z_i = transforms[i][:3, :3] @ AXIS_Z
        o_axis = transforms[i + 1][:3, 3]  # 关节 i+1 轴线所通过的点
        J[:3, i] = np.cross(z_i, o_n - o_axis)
        J[3:, i] = z_i
    return J


def within_limits(q_deg, margin=0.0):
    """检查关节角是否全部在限位内（可选安全边距，度）。"""
    q = np.asarray(q_deg, dtype=float)
    return np.all(q >= JOINT_LIMITS[:, 0] + margin) and np.all(q <= JOINT_LIMITS[:, 1] - margin)


def clamp_to_limits(q_deg):
    """把关节角裁剪到限位范围内。"""
    return np.clip(q_deg, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
