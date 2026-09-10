# 虚拟六轴机械臂仿真（参考稚辉君 Dummy-Robot）

纯软件六轴机械臂运动学仿真项目，以稚辉君（彭志辉）开源的
[Dummy-Robot](https://github.com/peng-zhihui/Dummy-Robot) 为参数基准，
用于验证运动学（FK/IK）与多轴同步轨迹规划逻辑，无需实体硬件。
后续可直接对接 6 路 STEP/DIR 步进电机控制。

## 文件结构

| 文件 | 说明 |
| --- | --- |
| `arm_model.py` | 机械臂模型：DH 参数表、正运动学（FK）、关节限位、几何雅可比 |
| `ik_solver.py` | 数值逆运动学（阻尼最小二乘 LM + 限位惩罚 + 随机重启 + 连续解链） |
| `trajectory.py` | 关节空间轨迹规划：梯形/S 曲线加减速、多轴时间同步、平滑路径 |
| `simulation.py` | 桌面仿真主程序：matplotlib 3D 动画 + 6 路虚拟电机脉冲曲线 |
| `web_server.py` | 网页控制后端：本地 HTTP 服务（纯标准库），JSON API + 静态页面 |
| `web/index.html` | 网页控制前端：three.js 3D 场景、滑块/点动、IK、轨迹回放 |
| `test_kinematics.py` | 验证脚本：FK→IK→FK 闭环误差、轨迹连续性、多轴同步等 |
| `reference/` | Dummy-Robot 固件源码参考（运动学核心实现） |
| `snapshot.png` | 仿真静态截图（运行 `python simulation.py --save` 生成） |

## DH 参数来源

取自 Dummy-Robot 固件（`2.Firmware/Core-STM32F4-fw/`）：

- 尺寸：`Robot/instances/dummy_robot.cpp` 中
  `DOF6Kinematic(0.109, 0.035, 0.146, 0.115, 0.052, 0.072)`（单位 m）
- 关节限位：同文件电机配置
  J1 ±170°、J2 -73~90°、J3 35~180°、J4 ±180°、J5 ±120°、J6 ±720°

注意：固件的 FK 并非教科书式"纯标准 DH 链"——其存储的 DH 矩阵只有
theta 零偏和 alpha 参与计算，连杆平移由四个常量向量表达，且含 DH 标准形
无法表示的 y 向分量。本项目采用与固件逐位一致的
`A_i = RotZ(θ) Trans(t) RotX(α)` 混合模型，已用数值方法与固件
`SolveFK` 在数百组随机角下比对，位置误差 < 1e-12 m。

## 运行方法

依赖：Python 3 + numpy + matplotlib（`pip install numpy matplotlib`）。

```bash
cd "d:/VSCode Sandbox/Projects/6 axle arm"

# 1. 运行验证（FK/IK 闭环误差、轨迹连续性、多轴同步）
python test_kinematics.py

# 2. 动画仿真（3D 机械臂 + 6 路电机脉冲曲线，播完自动退出）
python simulation.py

# 3. 无窗口模式，直接保存静态截图 snapshot.png
python simulation.py --save

# 4. 网页控制界面（自动打开浏览器；端口被占用时自动换 8001-8010）
python web_server.py
```

## 网页控制

`python web_server.py` 启动本地服务（默认 127.0.0.1:8000），自动打开浏览器。
three.js 通过 CDN 加载，需联网。Python 运动学模型是唯一真源，前端只做显示与交互。

## 纯前端部署（GitHub Pages）

`docs/` 目录是完整可独立运行的静态站点：前端优先连接本地后端，
检测不到时自动切换 **Pyodide**（浏览器内 WASM 运行的 CPython + numpy），
直接加载 `docs/py/` 下的原版运动学模块，功能与本地版完全一致，
零移植、零误差。首次加载需下载 Pyodide 运行时（约 15 MB，之后浏览器缓存）。

部署步骤：仓库 Settings → Pages → Source 选 "Deploy from a branch" →
分支 `main`、目录 `/docs`，保存后约 1 分钟可通过
`https://kyriesis.github.io/6-Axis-Arm-Simulator/` 访问。

注意：`docs/py/py_api.py` 与 `web_server.py` 的 payload 函数是同一套逻辑的
两份拷贝，修改业务逻辑时需同步两处。

功能：6 轴滑块拖动（实时 FK 刷新 3D）、每轴 ± 点动（按住连续步进 0.5°）、
末端位姿实时显示、IK 目标位姿求解并执行、点到点梯形/S 曲线轨迹、
圆形末端轨迹演示（水平/竖直，离线 IK 连续解链 + Catmull-Rom 平滑，
末端圆度误差 < 0.2 mm）、随机点动（后端生成限位内/远离奇异/工作空间中
段的随机目标，S 曲线到位后停 1 秒循环，可随时关闭）、停止/急停、
速度倍率（10%~100%，作用于所有轨迹回放）、末端轨迹拖尾（渐隐曲线，
回零/重播时清空）。状态栏统一显示当前模式（手动/轨迹回放/随机点动）与提示。

渲染说明：模型 z 轴朝上，前端统一经 `toScene` 映射到 three.js 的 y 朝上
坐标；底座高 50 mm，J1 原点位于底座上表面中心（机械臂整体抬高 BASE_H）。
主屏下方为 6 路虚拟步进电机 STEP/DIR 波形（逻辑分析仪样式，100 脉冲/°，
含 DIR 电平与瞬时脉冲频率），随任意关节运动实时滚动。

JSON API（可用 curl 测试）：

```
GET  /api/limits                      # 6 轴限位与尺寸
GET  /api/fk?j=q1,q2,q3,q4,q5,q6      # 各关节坐标 + 末端位姿
POST /api/ik      {position:[x,y,z], orientation?:[r,p,y], initial?:[6]}
POST /api/trajectory {from:[6], to:[6], profile?:trapezoid|scurve, dt?, with_fk?}
POST /api/demo/circle {center?, radius?, orientation?, points?, plane?:xy|xz, with_fk?}
POST /api/random_target {current?:[6]}   # 随机目标：限位内/远离奇异/变化>15°
```

轨迹与圆形接口返回按固定 dt 采样的 6 轴角度序列（with_fk=true 时附每帧
各关节坐标），前端按 dt 定时回放并同步滑块。

注意：本机 Python 位于 `D:\Python` 且启用了 safe_path（存在
`python._pth`），脚本内已做 `sys.path` 兼容处理，直接运行即可。
Windows 控制台建议设置 `set PYTHONUTF8=1` 以避免中文输出乱码。

## 轨迹规划思想（与 Dummy-Robot 固件一致）

MoveJ 指令到达后，计算 6 个轴的差分角度，取"最难走的轴"（换算后耗时
最长者）的运动时间作为公共时间，其余各轴按同一时刻缩放速度/加速度，
保证 6 轴同时到达、严格同步——这正是同时驱动 6 个步进电机时所需要的
脉冲时序逻辑。每个单轴支持梯形或 S 曲线（余弦加速）速度规划。
