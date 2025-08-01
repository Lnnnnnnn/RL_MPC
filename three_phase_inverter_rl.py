"""
三相两电平逆变器（带 L 滤波器）的强化学习示例
===================================================

本示例展示了如何使用 Gym 创建一个模拟三相两电平电压源
逆变器（VSI）的自定义环境，并利用 PyTorch 实现 DQN（Deep
Q‑Network）控制算法。逆变器连接到三个相同的 L 滤波器，
控制目标是在每个相线上生成与参考正弦波相同的电流。环境
具有 8 个离散动作，每个动作对应于逆变器三相开关状态的
一种组合。基于欧姆定律 V=IR【303936183651214†L54-L60】和电感两端电压 V = L·di/dt【663002723754139†L131-L137】的关系，环境在
每个时间步利用欧拉近似计算电流的变化。

环境说明
------------

**状态变量**：当前电感电流和对应的参考电流（共 6 个实
数）。

**动作空间**：``Discrete(8)``，动作编号 0–7 分别代表三相
开关 (S\_a, S\_b, S\_c) 的二进制组合。其中 ``0`` 表示某
相下桥臂导通，``1`` 表示上桥臂导通。由于同相上下桥臂
互补导通，故每相只有两种状态。根据三相开关组合，可得
到八个离散的相电压向量。为保持三相平衡，需从各相的
直流母线电压去除平均值，得到相对中性点的线电压。

**奖励函数**：负的平方误差，即 \(\sum_{p\in{a,b,c}}
(i_p - i_{\text{ref},p})^2\)。控制任务是最小化该误差，使相电
流跟踪正弦参考。

本文件既包含环境 ``ThreePhaseInverterEnv`` 的定义，也包含
DQN 智能体的简易实现和训练主函数 ``main()``。程序入口
在文末，如果直接运行此脚本，将自动训练一小段时间并打
印训练过程中的奖励平均值。

依赖：需要安装 gym 或 gymnasium、numpy、torch、matplotlib。
"""

import math
import random
from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces
import matplotlib.pyplot as plt
import os
import importlib.util  # 用于检查是否安装了 onnx 库


class ThreePhaseInverterEnv(gym.Env):
    """自定义 Gym 环境，用于模拟三相两电平逆变器。

    状态变量包括三相电感电流和目标参考电流，动作是三相
    开关状态组合。环境使用欧拉法离散化电感电流的微分方程，
    根据开关状态计算相电压，然后根据 V = L·di/dt 和 V = IR
    更新电流。
    """

    def __init__(
        self,
        v_dc: float = 400.0,
        inductance: float = 2e-3,
        resistance: float = 1.0,
        sample_time: float = 5e-5,
        ref_frequency: float = 50.0,
        ref_current: float = 10.0,
        episode_length: int = 2000,
        seed: int = 0,
        switch_penalty: float = 0.0,
        harmonic_penalty: float = 0.0,
        harmonic_window: int = 20,
    ):
        super().__init__()
        self.v_dc = v_dc
        self.L = inductance
        self.R = resistance
        self.dt = sample_time
        self.ref_freq = ref_frequency
        self.ref_amp = ref_current
        self.episode_length = episode_length
        self.time = 0.0
        # 状态：[i_a, i_b, i_c, i_ref_a, i_ref_b, i_ref_c]
        high = np.array([np.finfo(np.float32).max] * 6, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(8)
        self.state = np.zeros(6, dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self.step_count = 0
        # 上一次动作，用于计算开关变化惩罚
        self.prev_action: int | None = None
        # 历史电流，用于估计谐波含量
        self.current_history: list[list[float]] = []
        self.harmonic_window = harmonic_window
        self.switch_penalty = switch_penalty
        self.harmonic_penalty = harmonic_penalty

    def reset(self) -> np.ndarray:
        """重置环境，返回初始观察。"""
        self.time = 0.0
        self.step_count = 0
        # 初始电流设为零，初始角度随机化
        self.state = np.zeros(6, dtype=np.float32)
        # 重置开关惩罚和谐波窗口
        self.prev_action = None
        self.current_history = []
        return self.state.copy()

    def _switch_to_voltage(self, action: int) -> Tuple[float, float, float]:
        """根据动作编号计算三相相电压。

        每个动作编码为 3 位二进制，依次对应相 A、B、C 的上桥臂
        导通情况。上桥臂导通代表相对直流母线输出 +Vdc/2，下桥臂
        导通代表 -Vdc/2。为了获得相对中性点电压，需要从三相电压
        中减去平均值。

        返回值：三个相的线电压 (v_a, v_b, v_c)。
        """
        # 解析动作为三相开关状态
        s_a = (action >> 2) & 0x1
        s_b = (action >> 1) & 0x1
        s_c = action & 0x1
        # 每相桥臂输出：+Vdc/2 或 -Vdc/2
        v_leg_a = (2 * s_a - 1) * self.v_dc / 2.0
        v_leg_b = (2 * s_b - 1) * self.v_dc / 2.0
        v_leg_c = (2 * s_c - 1) * self.v_dc / 2.0
        # 相电压 = leg 电压 - 平均值，保证三相和为 0
        mean_leg = (v_leg_a + v_leg_b + v_leg_c) / 3.0
        v_a = v_leg_a - mean_leg
        v_b = v_leg_b - mean_leg
        v_c = v_leg_c - mean_leg
        return v_a, v_b, v_c

    def _reference_currents(self, t: float) -> Tuple[float, float, float]:
        """计算在时间 t 时刻三相正弦参考电流。

        相位差设定为 120°。由于 sin 函数周期为 2π，因而
        phi_b = -2π/3, phi_c = -4π/3。振幅为 ref_amp。
        """
        omega = 2.0 * math.pi * self.ref_freq
        i_ref_a = self.ref_amp * math.sin(omega * t)
        i_ref_b = self.ref_amp * math.sin(omega * t - 2.0 * math.pi / 3.0)
        i_ref_c = self.ref_amp * math.sin(omega * t - 4.0 * math.pi / 3.0)
        return i_ref_a, i_ref_b, i_ref_c

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """执行一步仿真并返回结果。

        步骤包括：
          1. 依据动作计算三相开关状态，并转换为三相输出电压。开关
             状态由 3 位二进制表示，每位代表某相的上/下桥臂。两
             个桥臂互补，使输出仅为 +Vdc/2 或 -Vdc/2，并减去平均值
             获得相对中性点电压。
          2. 根据欧姆定律【303936183651214†L54-L60】计算电阻产生的电压降，并利用电感两端的
             基本关系 V=L·di/dt【663002723754139†L131-L137】更新电流。这儿使用欧拉法进行
             时间离散化。
          3. 更新时间、计数器以及计算新的参考电流。
          4. 奖励采用负的归一化均方误差，使得奖励值较为平滑，
             有利于训练稳定。
        返回值包括新的状态向量、标量奖励、布尔结束标志以及
        调试信息（此处为空字典）。
        """
        assert self.action_space.contains(action), "非法动作"
        # 当前电流
        i_a, i_b, i_c = self.state[0:3]
        # 计算相电压
        v_a, v_b, v_c = self._switch_to_voltage(action)
        # 使用欧姆定律和电感方程计算 di/dt
        # di/dt = (v - R * i) / L
        di_a = (v_a - self.R * i_a) / self.L
        di_b = (v_b - self.R * i_b) / self.L
        di_c = (v_c - self.R * i_c) / self.L
        # 用欧拉法更新电流
        i_a_next = i_a + di_a * self.dt
        i_b_next = i_b + di_b * self.dt
        i_c_next = i_c + di_c * self.dt
        # 更新时间
        self.time += self.dt
        self.step_count += 1
        # 计算参考电流
        i_ref_a, i_ref_b, i_ref_c = self._reference_currents(self.time)
        # 更新状态向量
        self.state = np.array([i_a_next, i_b_next, i_c_next,
                               i_ref_a, i_ref_b, i_ref_c], dtype=np.float32)
        # 基础奖励为负的归一化均方误差。归一化可以缓解奖励量级过大导致的
        # 数值问题，让总奖励更易于直观比较。
        error = (
            (i_a_next - i_ref_a) ** 2
            + (i_b_next - i_ref_b) ** 2
            + (i_c_next - i_ref_c) ** 2
        )
        reward = -error / (self.ref_amp ** 2)
        # 开关变化惩罚：若连续两步动作不同，则添加惩罚
        if self.prev_action is not None and action != self.prev_action:
            reward -= self.switch_penalty
        self.prev_action = action
        # 谐波惩罚：维护固定长度的电流历史，计算当前电流与均值的差异
        if self.harmonic_penalty > 0:
            # 将新的电流样本加入历史
            self.current_history.append([i_a_next, i_b_next, i_c_next])
            if len(self.current_history) > self.harmonic_window:
                self.current_history.pop(0)
            # 计算窗口内电流均值
            hist = np.array(self.current_history)
            mean_currents = hist.mean(axis=0)
            # 误差表示快速变化部分，惩罚高频谐波
            harmonic_error = (
                (i_a_next - mean_currents[0]) ** 2
                + (i_b_next - mean_currents[1]) ** 2
                + (i_c_next - mean_currents[2]) ** 2
            )
            reward -= self.harmonic_penalty * harmonic_error
        # 是否结束
        done = self.step_count >= self.episode_length
        return self.state.copy(), float(reward), done, {}


class ReplayBuffer:
    """简单的经验回放缓冲区。"""

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (np.stack(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.stack(next_states), np.array(dones, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.buffer)


class DQN(nn.Module):
    """全连接前馈神经网络，用于逼近 Q 值函数。

    架构由隐藏层列表 `hidden_sizes` 决定，允许用户灵活调整
    网络深度和每层节点数以提升逼近能力。
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int] | None = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 128]
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_model(model: nn.Module, filepath: str) -> None:
    """保存 PyTorch 模型的状态字典到指定路径。"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"模型已保存到 {filepath}")


def load_model(model: nn.Module, filepath: str) -> None:
    """从指定路径加载模型权重到给定模型实例。"""
    state_dict = torch.load(filepath, map_location="cpu")
    model.load_state_dict(state_dict)
    print(f"已从 {filepath} 加载模型权重")


def export_to_onnx(model: nn.Module, filepath: str, input_dim: int) -> None:
    """将 PyTorch 模型导出为 ONNX 格式。

    导出前使用一个假输入张量推断模型的计算图。导出的
    ONNX 文件可在其他框架（如 Simulink）中进行验证。
    """
    # 如果系统中未安装 onnx 库，则跳过导出并提示用户
    if importlib.util.find_spec("onnx") is None:
        print("ONNX 库未安装，跳过导出 ONNX 文件。")
        return
    dummy_input = torch.zeros(1, input_dim)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # 调用 torch.onnx.export 进行导出。该函数会自动使用已安装的 onnx 库
    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        input_names=["state"],
        output_names=["q_values"],
        opset_version=11,
    )
    print(f"模型已导出为 ONNX 格式：{filepath}")

def train_dqn(
    env: ThreePhaseInverterEnv,
    num_episodes: int = 200,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-3,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    target_update_freq: int = 10,
    save_path: str | None = None,
    onnx_path: str | None = None,
    load_path: str | None = None,
    hidden_sizes: List[int] | None = None,
    plot_live: bool = False,
    export_onnx_flag: bool = True,
) -> nn.Module:
    """训练 DQN 智能体，对环境进行若干回合训练。

    核心流程：
      * 初始化行为网络 `policy_net` 和目标网络 `target_net`。目标
        网络是行为网络的一个延迟拷贝，用于计算稳定的目标 Q 值。
      * 在每一步，采用 ε‑贪心策略选择动作。以概率 ε 随机探索，
        否则选择当前行为网络 Q 值最大的动作。
      * 与环境交互，收集 (state, action, reward, next_state, done) 五元组
        存入经验回放缓冲区。经验回放可以打乱样本相关性，提高
        学习效率。
      * 当缓冲区存储足够样本后，按批次随机抽取经验，计算 TD
        目标值，并用均方误差损失对行为网络进行梯度更新。
      * 每隔若干回合将目标网络的参数替换为行为网络，以便跟随
        最新策略。
      * 训练结束后，将所有回合的奖励绘制为曲线，方便查看收敛
        趋势。

    参数说明：
      - env: 要训练的环境实例。
      - num_episodes: 总回合数。较大的回合数有助于更充分的训练。
      - batch_size: 每次梯度更新所需的样本数量。
      - gamma: 折扣因子，权衡立即奖励和未来奖励的重要性。
      - lr: 优化器学习率。
      - epsilon_start/end/decay: ε‑贪心探索策略的起始值、结束值和
        每步衰减因子，用于平衡探索与利用。
      - target_update_freq: 更新目标网络参数的间隔（以回合为单位）。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 打印所使用的设备，方便确认是否在 GPU 上训练
    print(f"使用的设备: {device}")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    # 创建策略网络和目标网络，允许自定义隐藏层结构
    # 根据传入的 hidden_sizes 创建策略网络和目标网络
    policy_net = DQN(obs_dim, act_dim, hidden_sizes=hidden_sizes).to(device)
    target_net = DQN(obs_dim, act_dim, hidden_sizes=hidden_sizes).to(device)
    # 如果提供了预训练模型路径，则在继续训练前加载其参数
    if load_path is not None and os.path.exists(load_path):
        load_model(policy_net, load_path)
        print(f"继续训练已加载的模型: {load_path}")
    # 初始时目标网络复制策略网络参数
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayBuffer(100000)
    epsilon = epsilon_start
    all_rewards: List[float] = []
    # 用于存储每回合的 epsilon 以便绘制 epsilon 曲线
    all_epsilons: List[float] = []
    # 如果启用实时绘图，则初始化交互式图形
    if plot_live:
        plt.ion()
        fig, ax1 = plt.subplots(figsize=(8, 4))
        # 奖励曲线使用 ax1
        line_reward, = ax1.plot([], [], label="episode reward")
        window = 10
        line_avg, = ax1.plot([], [], label=f"moving average ({window})")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total reward")
        ax1.set_title("Training progress")
        # 第二 y 轴用于绘制 epsilon
        ax2 = ax1.twinx()
        line_epsilon, = ax2.plot([], [], color="tab:red", label="epsilon")
        ax2.set_ylabel("Epsilon", color="tab:red")
        ax2.tick_params(axis='y', labelcolor="tab:red")
        # 合并图例
        lines = [line_reward, line_avg, line_epsilon]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right")
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            # ε‑greedy 选择动作
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                    q_values = policy_net(state_t)
                    action = int(torch.argmax(q_values, dim=1).item())
            next_state, reward, done, _ = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            # 当经验足够时开始训练
            if len(memory) >= batch_size:
                states, actions, rewards, next_states, dones = memory.sample(batch_size)
                states = torch.from_numpy(states).float().to(device)
                actions = torch.from_numpy(actions).long().to(device)
                rewards = torch.from_numpy(rewards).float().to(device)
                next_states = torch.from_numpy(next_states).float().to(device)
                dones = torch.from_numpy(dones).float().to(device)
                # 计算 Q(s,a)
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                # 计算 target Q 值
                with torch.no_grad():
                    max_next_q = target_net(next_states).max(1)[0]
                    target = rewards + gamma * (1.0 - dones) * max_next_q
                loss = nn.functional.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # 更新 epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        all_rewards.append(total_reward)
        all_epsilons.append(epsilon)
        # 定期更新目标网络
        if (episode + 1) % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        # 在每个回合结束后输出当前回合的训练信息
        # 在每个回合结束后输出当前回合的训练信息。
        # 打印时保留更多小数位，便于观察收敛趋势。
        print(
            f"Episode {episode + 1}/{num_episodes} | "
            f"total reward: {total_reward:.4f} | "
            f"epsilon: {epsilon:.3f}"
        )
        # 每 10 个回合计算一次最近 10 个回合的平均奖励，方便观察训练趋势
        if (episode + 1) % 10 == 0 and len(all_rewards) >= 10:
            avg_r = sum(all_rewards[-10:]) / 10
            print(
                f" -- Average reward (last 10 episodes): {avg_r:.4f}"
            )
        # 如果启用了实时绘图，则更新图像数据
        if plot_live:
            x = list(range(1, len(all_rewards) + 1))
            # 更新奖励曲线
            line_reward.set_data(x, all_rewards)
            # 更新移动平均
            if len(all_rewards) >= window:
                moving_avg = [
                    sum(all_rewards[i - window : i]) / window
                    for i in range(window, len(all_rewards) + 1)
                ]
                line_avg.set_data(list(range(window, len(all_rewards) + 1)), moving_avg)
            # 更新 epsilon 曲线
            line_epsilon.set_data(x, all_epsilons)
            # 自适应调整坐标轴范围
            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
            plt.pause(0.001)

    # 训练结束后绘制奖励曲线。
    # 如果未启用实时绘图，则在训练结束时生成静态图像；若启用
    # 实时绘图，则在训练结束后关闭交互模式并保存当前图。
    if all_rewards:
        if not plot_live:
            plt.figure(figsize=(8, 4))
            plt.plot(all_rewards, label="episode reward")
            window = 10
            if len(all_rewards) >= window:
                moving_avg = [
                    sum(all_rewards[i - window : i]) / window
                    for i in range(window, len(all_rewards) + 1)
                ]
                plt.plot(
                    range(window, len(all_rewards) + 1),
                    moving_avg,
                    label=f"moving average ({window})",
                )
            plt.xlabel("Episode")
            plt.ylabel("Total reward")
            plt.title("Training reward over episodes")
            plt.legend()
            plt.tight_layout()
            plt.savefig("training_curve.png")
            plt.close()
        else:
            # 关闭交互模式，保存实时绘制的图
            plt.ioff()
            plt.savefig("training_curve.png")
            plt.close()

    # 保存模型权重
    if save_path is not None:
        save_model(policy_net.cpu(), save_path)
    # 导出 ONNX
    if onnx_path is not None and export_onnx_flag:
        export_to_onnx(policy_net.cpu(), onnx_path, obs_dim)

    # 返回训练好的模型实例
    return policy_net


def main() -> None:
    """主函数：创建环境并训练 DQN 智能体。

    该函数负责实例化自定义环境 `ThreePhaseInverterEnv` 并
    调用 `train_dqn` 进行训练。可在此处调整参数来模拟不
    同的电力电子系统。训练完成后，会在当前目录生成
    `training_curve.png`，展示回合奖励随训练进度的变化。
    """
    # 创建环境。参数可根据实际逆变器硬件进行调整：
    #  - v_dc: 直流母线电压
    #  - inductance: L 滤波器的电感 (H)
    #  - resistance: 电阻模型，用于模拟负载或线路损耗 (Ω)
    #  - sample_time: 控制间隔 (s)，需足够小以稳定模拟
    #  - ref_frequency: 参考电流的频率 (Hz)
    #  - ref_current: 参考电流的幅值 (A)
    #  - episode_length: 每个回合的步数
    env = ThreePhaseInverterEnv(
        v_dc=400.0,
        inductance=2e-3,
        resistance=1.0,
        sample_time=5e-5,
        ref_frequency=50.0,
        ref_current=10.0,
        episode_length=2000,
        seed=0,
    )
    # 训练智能体并保存模型。我们同时将训练好的策略网络导出为
    # PyTorch 的 .pth 文件和 ONNX 文件，方便后续在 Simulink
    # 等其他平台上验证。
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    model_pth = os.path.join(save_dir, "policy_net.pth")
    model_onnx = os.path.join(save_dir, "policy_net.onnx")
    trained_model = train_dqn(
        env,
        num_episodes=200,
        batch_size=64,
        save_path=model_pth,
        onnx_path=model_onnx,
        load_path=None,
    )

    # 训练结束后可以调用 validate 函数评估模型效果。
    # 这里不自动运行，以免延长训练时间。用户可自行在需要时调用。


def validate(env: ThreePhaseInverterEnv, model_path: str, episodes: int = 10) -> None:
    """使用保存的模型在环境中做推理验证。

    加载存储的权重，并以贪心策略运行若干回合，输出每个回合
    的总奖励以及平均奖励。该函数可用于评估训练效果。

    参数：
      - env: 需要评估的环境实例。
      - model_path: 模型权重 .pth 文件路径。
      - episodes: 验证回合数。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    model = DQN(obs_dim, act_dim).to(device)
    load_model(model, model_path)
    model.eval()
    returns = []
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            with torch.no_grad():
                state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                q_values = model(state_t)
                action = int(torch.argmax(q_values, dim=1).item())
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
        returns.append(total_reward)
        print(f"验证回合 {ep+1}/{episodes} 总奖励: {total_reward:.4f}")
    if returns:
        avg_ret = sum(returns) / len(returns)
        print(f"平均验证奖励: {avg_ret:.4f}")


if __name__ == "__main__":
    main()