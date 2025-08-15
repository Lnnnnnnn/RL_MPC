"""
基于文献中的并联混合逆变器 αβγ 状态空间模型的 RL 示例
========================================================

本文件演示如何根据提供的状态空间方程在 αβγ 坐标系下构建并
联混合逆变器 (PHC) 的强化学习环境。文献中给出了系统的状态
向量 \(x_{\alpha\beta\gamma}=[I_{G\alpha},I_{HF\alpha},I_{G\beta},I_{HF\beta},I_{HF\gamma}]^T\)，
控制输入向量 \(u=[V_{LF\alpha},V_{LF\beta},V_{LF\gamma},V_{HF\alpha},V_{HF\beta},V_{HF\gamma}]^T\)，
以及网侧电压向量 \(v=[V_{G\alpha},V_{G\beta}]^T\)。系统动态满足

\[\dot{x} = A x + B u + E v,\]

其中矩阵 A、B、E 的元素由感性耦合网络的电感和电阻决定。由
于高频和低频支路均使用两电平逆变器，理论上动作空间包含
6 个电压控制输入的离散组合，导致 64 个动作。为简化示例，
本模型仅控制高频支路的三相电压 \(V_{HF\alpha}, V_{HF\beta}, V_{HF\gamma}\)，
而将低频支路电压 \(V_{LF\alpha}, V_{LF\beta}, V_{LF\gamma}\) 设为零，从而动
作空间为 8 种组合。网侧电压 v 采用理想正弦波给定。

由于缺乏所有元件参数的具体数值，代码中使用表~\ref{tab:Design
Parameters and Selected Components} 中的电感值及假定的电阻值构建
矩阵 A、B、E。用户应根据文献给出的参数调整电阻、电感及
电源电压，以获得更准确的模型。

参考文献：
 见表中所示的实验参数及状态空间方程【291549923093159†L31-L47】。
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
import importlib.util


"""
并联混合逆变器 αβγ 状态空间强化学习环境（仅保留 64 种动作）。

本模块整合了并联混合逆变器 (PHC) 的 64 状态环境、经验回放、
DQN 模型以及训练与验证函数。与原版不同，此文件只包含一个
环境类 `PHCAlphaBeta64Env`，不再提供 8 状态的 `PHCAlphaBetaEnv`。

系统模型采用文献中给出的状态空间形式：
\[\dot{x} = A x + B u + E v,\]
其中状态向量
    x = [I_{G\alpha}, I_{HF\alpha}, I_{G\beta}, I_{HF\beta}, I_{HF\gamma}]^T，
控制输入向量
    u = [V_{LF\alpha}, V_{LF\beta}, V_{LF\gamma}, V_{HF\alpha}, V_{HF\beta}, V_{HF\gamma}]^T，
网侧电压向量
    v = [V_{G\alpha}, V_{G\beta}]^T。
动作空间共有 64 种组合，每个动作的 6 位二进制编码分别表示 LF 和 HF
支路的三相开关状态。奖励函数综合考虑：

 1. LF 支路电流 (I_G\alpha, I_G\beta) 跟踪参考值；
 2. HF 支路电流 (I_{HF\alpha}, I_{HF\beta}, I_{HF\gamma}) 逼近 0；
 3. 开关惩罚：LF 支路开关次数的惩罚权重大于 HF 支路；
 4. 可选谐波惩罚：基于滑动窗口的电流方差抑制高频谐波。

用户可通过 `switch_penalty_lf`、`switch_penalty_hf`、`harmonic_penalty`、
`harmonic_window` 等参数调整这些惩罚项。训练与验证接口与
`three_phase_inverter_rl.py` 中类似，可直接用于 DQN 训练。
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


class PHCAlphaBeta64Env(gym.Env):
    """
    并联混合逆变器 αβγ 状态空间环境，动作空间为 64。

    状态向量 x = [I_{G\alpha}, I_{HF\alpha}, I_{G\beta}, I_{HF\beta}, I_{HF\gamma}]。
    动作向量 6 位二进制编码，前三位为 LF 支路三相开关状态 (a,b,c)，
    后三位为 HF 支路三相开关状态 (a,b,c)。每位 1 表示上桥臂导通
    (+Vdc/2)，0 表示下桥臂导通 (-Vdc/2)。

    奖励由四部分组成：
      - LF 电流跟踪误差 (网侧电流与参考电流之差)；
      - HF 电流偏离零的大小；
      - LF 和 HF 支路的开关次数惩罚；
      - 可选谐波惩罚 (基于滑动窗口电流方差)。

    使用参数 `switch_penalty_lf`、`switch_penalty_hf`、`harmonic_penalty`、
    `harmonic_window` 进行调节。
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        v_dc: float = 300.0,
        sample_time: float = 1e-5,
        ref_frequency: float = 50.0,
        ref_current: float = 10.0,
        episode_length: int = 2000,
        seed: int = 0,
        switch_penalty_lf: float = 0.0,
        switch_penalty_hf: float = 0.0,
        harmonic_penalty: float = 0.0,
        harmonic_window: int = 20,
    ) -> None:
        super().__init__()
        self.v_dc = v_dc
        self.dt = sample_time
        self.ref_freq = ref_frequency
        self.ref_amp = ref_current
        self.episode_length = episode_length
        self.time = 0.0
        self.step_count = 0
        # 状态空间：五维实数向量
        high = np.array([np.finfo(np.float32).max] * 5, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        # 动作空间：64 种组合
        self.action_space = spaces.Discrete(64)
        self.state = np.zeros(5, dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        # 构建 A、B、E 矩阵
        self.A, self.B, self.E = self._compute_state_matrices()
        # 上一步动作，用于计算开关次数
        self.prev_action: int | None = None
        # 谐波惩罚窗口
        self._current_window: Deque[np.ndarray] = deque(maxlen=harmonic_window)
        # 开关及谐波惩罚系数
        self.switch_penalty_lf = switch_penalty_lf
        self.switch_penalty_hf = switch_penalty_hf
        self.harmonic_penalty = harmonic_penalty
        self.harmonic_window = harmonic_window

    # ----- 模型参数矩阵 -----
    def _compute_state_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        根据文献表中的电感与假设的电阻值计算 A、B、E 矩阵。

        用户应根据实际实验参数调整电阻值 R 和电感值 L。此函数
        返回连续时间状态空间模型的矩阵，用于后续离散化。
        """
        # 电感值 (H)
        LLFa = 420e-6
        LHF_a = 157.12e-6
        LG_a = 420e-6
        LLFb = 420e-6
        LHF_b = 157.12e-6
        LG_b = 420e-6
        LHF_g = 1485e-6
        # 假设的电阻值 (Ω)
        RLF_a = 0.1
        RHF_a = 0.1
        RG_a = 0.1
        RLF_b = 0.1
        RHF_b = 0.1
        RG_b = 0.1
        RHF_g = 0.1
        Lcm = LHF_g  # 公共模感应
        A = np.zeros((5, 5), dtype=np.float64)
        # # α 通道
        # denom_a = LLFa * LHF_a + (LLFa + LHF_a) * LG_a
        # A[0, 0] = - (LHF_a * RLF_a + (LLFa + LHF_a) * RG_a) / denom_a
        # A[0, 2] = - (LLFa * RHF_a - LHF_a * RLF_a) / denom_a
        # A[2, 0] = - (LLFa * RG_a - LG_a * RLF_a) / denom_a
        # A[2, 2] = - (LLFa * RHF_a + (RLF_a + RHF_a) * LG_a) / denom_a
        # # β 通道
        # denom_b = LLFb * LHF_b + (LLFb + LHF_b) * LG_b
        # A[1, 1] = - (LHF_b * RLF_b + (LLFb + LHF_b) * RG_b) / denom_b
        # A[1, 3] = - (LLFb * RHF_b - LHF_b * RLF_b) / denom_b
        # A[3, 1] = - (LLFb * RG_b - LG_b * RLF_b) / denom_b
        # A[3, 3] = - (LLFb * RHF_b + (RLF_b + RHF_b) * LG_b) / denom_b
        # # γ 通道
        # A[4, 4] = - RHF_g / Lcm

        A[0, 0] = 1
        A[1, 1] = 1
        A[2, 2] = 1
        A[3, 3] = 1
        A[4, 4] = 1

        # 输入矩阵 B
        B = np.zeros((5, 6), dtype=np.float64)

        # # α 通道
        # B[0, 0] = LHF_a / denom_a
        # B[0, 3] = - LLFa / denom_a
        # B[2, 0] = - LG_a / denom_a
        # B[2, 3] = (LLFa + LG_a) / denom_a
        # # β 通道
        # B[1, 1] = LHF_b / denom_b
        # B[1, 4] = - LLFb / denom_b
        # B[3, 1] = - LG_b / denom_b
        # B[3, 4] = (LLFb + LG_b) / denom_b
        # # γ 通道
        # B[4, 2] = -1.0 / Lcm
        # B[4, 5] = 1.0 / Lcm

        B[0, 0] = 0.0057
        B[0, 1] = 0.0123
        B[1, 0] = -0.0123
        B[1, 1] = 0.0246

        B[2, 2] = 0.0057
        B[2, 3] = 0.0123
        B[3, 2] = -0.0123
        B[3, 3] = 0.0246

        B[4, 4] = -0.0042
        B[4, 5] = 0.0042

        # 网侧电压矩阵 E
        E = np.zeros((5, 2), dtype=np.float64)
        # E[0, 0] = - (LLFa + LHF_a) / denom_a
        # E[2, 0] = - LLFa / denom_a
        # E[1, 1] = - (LLFb + LHF_b) / denom_b
        # E[3, 1] = - LLFb / denom_b

        E[0, 0] = -0.0181
        E[1, 0] = -0.0123
        E[2, 1] = -0.0181
        E[3, 1] = -0.0123
        # γ 通道不受网侧电压影响
        return A, B, E

    # ----- 环境交互 -----
    def reset(self) -> np.ndarray:
        """重置环境到初始状态。"""
        self.time = 0.0
        self.step_count = 0
        self.state = np.zeros(5, dtype=np.float32)
        self.prev_action = None
        self._current_window.clear()
        return self.state.copy()

    def _grid_voltage_ab(self, t: float) -> Tuple[float, float]:
        """
        计算 αβ 坐标下的网侧电压 V_Gα、V_Gβ。
        假设网侧三相电压为平衡的正弦波，线电压 RMS 为 172.5 V。
        """
        V_line_rms = 172.5
        V_phase_peak = V_line_rms * math.sqrt(2/3)
        omega = 2.0 * math.pi * self.ref_freq
        Va = V_phase_peak * math.sin(omega * t)
        Vb = V_phase_peak * math.sin(omega * t - 2.0 * math.pi / 3.0)
        Vc = V_phase_peak * math.sin(omega * t - 4.0 * math.pi / 3.0)
        V_alpha = math.sqrt(2.0 / 3.0) * (Va - 0.5 * (Vb + Vc))
        V_beta = math.sqrt(2.0 / 3.0) * (math.sqrt(3) / 2.0) * (Vb - Vc)
        return V_alpha, V_beta

    def _reference_current_ab(self, t: float) -> Tuple[float, float]:
        """
        计算 αβ 坐标下的参考电流 I_ref_α、I_ref_β。
        假定三相电流幅值为 ref_amp 的正弦波。
        """
        I_phase_peak = self.ref_amp
        omega = 2.0 * math.pi * self.ref_freq
        ia = I_phase_peak * math.sin(omega * t)
        ib = I_phase_peak * math.sin(omega * t - 2.0 * math.pi / 3.0)
        ic = I_phase_peak * math.sin(omega * t - 4.0 * math.pi / 3.0)
        I_alpha = math.sqrt(2.0/3.0) * (ia - 0.5 * (ib + ic))
        I_beta = math.sqrt(2.0/3.0) * (math.sqrt(3)/2.0) * (ib - ic)
        return I_alpha, I_beta

    def _decode_two_branches(self, action: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        将动作解码为两支路的三相电压 (abc)，并去除零序。

        返回 (v_LF_a, v_LF_b, v_LF_c), (v_HF_a, v_HF_b, v_HF_c)。
        """
        # LF 位 5-3，HF 位 2-0
        s_lf_a = (action >> 5) & 0x1
        s_lf_b = (action >> 4) & 0x1
        s_lf_c = (action >> 3) & 0x1
        s_hf_a = (action >> 2) & 0x1
        s_hf_b = (action >> 1) & 0x1
        s_hf_c = action & 0x1
        # 计算桥臂电压
        lf_leg_a = (2 * s_lf_a - 1) * self.v_dc / 2.0
        lf_leg_b = (2 * s_lf_b - 1) * self.v_dc / 2.0
        lf_leg_c = (2 * s_lf_c - 1) * self.v_dc / 2.0
        hf_leg_a = (2 * s_hf_a - 1) * self.v_dc / 2.0
        hf_leg_b = (2 * s_hf_b - 1) * self.v_dc / 2.0
        hf_leg_c = (2 * s_hf_c - 1) * self.v_dc / 2.0
        # # 去除零序分量
        # mean_lf = (lf_leg_a + lf_leg_b + lf_leg_c) / 3.0
        # v_lf_a = lf_leg_a - mean_lf
        # v_lf_b = lf_leg_b - mean_lf
        # v_lf_c = lf_leg_c - mean_lf
        # mean_hf = (hf_leg_a + hf_leg_b + hf_leg_c) / 3.0
        # v_hf_a = hf_leg_a - mean_hf
        # v_hf_b = hf_leg_b - mean_hf
        # v_hf_c = hf_leg_c - mean_hf
        return (lf_leg_a, lf_leg_b, lf_leg_c), (hf_leg_a, hf_leg_b, hf_leg_c)

    def _abc_to_abg(self, v_a: float, v_b: float, v_c: float) -> Tuple[float, float, float]:
        """
        将三相电压 (a,b,c) 转换到 αβγ 坐标。
        """
        v_alpha = math.sqrt(2.0 / 3.0) * (v_a - 0.5 * (v_b + v_c))
        v_beta = math.sqrt(2.0 / 3.0) * (math.sqrt(3) / 2.0) * (v_b - v_c)
        v_gamma = math.sqrt(1.0 / 3.0) * (v_a + v_b + v_c)
        return v_alpha, v_beta, v_gamma

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行一步仿真，返回下一个状态、奖励、终止标志和信息。

        奖励包括 LF 电流跟踪误差、HF 电流误差、开关惩罚和谐波惩罚。
        """
        assert self.action_space.contains(action), "非法动作"
        x = self.state.astype(np.float64)
        # 解码动作得到两支路电压
        (v_lf_a, v_lf_b, v_lf_c), (v_hf_a, v_hf_b, v_hf_c) = self._decode_two_branches(action)
        # 转为 αβγ
        v_lf_alpha, v_lf_beta, v_lf_gamma = self._abc_to_abg(v_lf_a, v_lf_b, v_lf_c)
        v_hf_alpha, v_hf_beta, v_hf_gamma = self._abc_to_abg(v_hf_a, v_hf_b, v_hf_c)
        # 控制输入向量 (Change the sequence)
        u = np.array([
            v_lf_alpha,
            v_hf_alpha,
            v_lf_beta,
            v_hf_beta,
            v_lf_gamma,
            v_hf_gamma,
        ], dtype=np.float64)
        # 网侧电压
        V_alpha, V_beta = self._grid_voltage_ab(self.time)
        v_vec = np.array([V_alpha, V_beta], dtype=np.float64)

        # 连续时间微分并离散化
        dx = self.A @ x + self.B @ u + self.E @ v_vec

        x_next = x + self.dt * dx
        # 更新时间步
        self.time += self.dt
        self.step_count += 1
        self.state = x_next.astype(np.float32)
        # 参考电流
        I_alpha_ref, I_beta_ref = self._reference_current_ab(self.time)
        # 误差
        err_lf = (x_next[0] - I_alpha_ref) ** 2 + (x_next[2] - I_beta_ref) ** 2
        err_hf = x_next[1] ** 2 + x_next[3] ** 2 + x_next[4] ** 2
        error = err_lf + err_hf
        reward = -error / (self.ref_amp ** 2)
        # 开关惩罚
        if self.prev_action is not None:
            lf_prev = self.prev_action >> 3
            hf_prev = self.prev_action & 0b111
            lf_now = action >> 3
            hf_now = action & 0b111
            lf_switches = sum(((lf_now >> i) & 1) != ((lf_prev >> i) & 1) for i in range(3))
            hf_switches = sum(((hf_now >> i) & 1) != ((hf_prev >> i) & 1) for i in range(3))
            reward -= self.switch_penalty_lf * lf_switches
            reward -= self.switch_penalty_hf * hf_switches
        # 谐波惩罚
        if self.harmonic_penalty > 0.0:
            self._current_window.append(np.array([x_next[0], x_next[2]]))
            if len(self._current_window) == self.harmonic_window:
                arr = np.stack(self._current_window, axis=0)
                mean_val = np.mean(arr, axis=0)
                harmonic_energy = float(np.sum((arr - mean_val) ** 2))
                reward -= self.harmonic_penalty * harmonic_energy
        # 更新 prev_action
        self.prev_action = action
        # 终止条件
        done = self.step_count >= self.episode_length
        return self.state.copy(), float(reward), done, {}

    # ----- 渲染与关闭 -----
    def render(self, mode="human"):
        # 此环境目前不支持图形渲染，可扩展为绘制电流波形等
        pass

    def close(self):
        pass


# ----- 经验回放 -----
class ReplayBuffer:
    def __init__(self, capacity: int = 100000) -> None:
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ----- DQN 网络 -----
class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int] | None = None) -> None:
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


# ----- 模型保存和加载 -----
def save_model(model: nn.Module, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)


def load_model(model: nn.Module, load_path: str) -> None:
    state_dict = torch.load(load_path, map_location="cpu")
    model.load_state_dict(state_dict)


# ----- 训练函数 -----
def train_dqn(
    env: gym.Env,
    num_episodes: int = 200,
    batch_size: int = 64,
    gamma: float = 0.99,
    learning_rate: float = 1e-3,
    target_update: int = 500,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    hidden_sizes: List[int] | None = None,
    save_path: str | None = None,
    onnx_path: str | None = None,
    load_path: str | None = None,
    plot_live: bool = False,
    export_onnx_flag: bool = True,
    *args,
    **kwargs,
) -> nn.Module:
    """
    使用 DQN 在给定环境 `env` 上训练策略。支持模型保存、继续训练和实时绘图。

    参数:
      - env: 环境实例，必须实现 step/reset 接口。
      - num_episodes: 训练回合数。
      - batch_size: 批量大小。
      - gamma: 折扣因子。
      - learning_rate: Adam 学习率。
      - target_update: 每多少步同步一次目标网络。
      - epsilon_start/end/decay: ε-贪心策略参数。
      - hidden_sizes: DQN 隐藏层配置。
      - save_path: 权重保存路径 (.pth)。
      - onnx_path: 导出的 ONNX 模型路径 (.onnx)。
      - load_path: 如果非空，先加载已有模型再继续训练。
      - plot_live: 实时绘制奖励和 epsilon 曲线。
      - export_onnx_flag: 是否导出 ONNX 模型 (需要安装 onnx)。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    policy_net = DQN(obs_dim, act_dim, hidden_sizes).to(device)
    target_net = DQN(obs_dim, act_dim, hidden_sizes).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    if load_path is not None:
        try:
            load_model(policy_net, load_path)
            target_net.load_state_dict(policy_net.state_dict())
            print(f"已加载模型权重 {load_path}")
        except FileNotFoundError:
            print(f"加载模型失败: 文件 {load_path} 不存在")
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer()
    epsilon = epsilon_start
    rewards_history: List[float] = []
    eps_history: List[float] = []
    if plot_live:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    global_step = 0
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            # ε-贪心选择动作
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = policy_net(state_t)
                    action = int(torch.argmax(q_values).item())
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            # 更新策略网络
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                states_b, actions_b, rewards_b, next_states_b, dones_b = batch
                states_t = torch.as_tensor(states_b, dtype=torch.float32, device=device)
                actions_t = torch.as_tensor(actions_b, dtype=torch.int64, device=device).unsqueeze(-1)
                rewards_t = torch.as_tensor(rewards_b, dtype=torch.float32, device=device)
                next_states_t = torch.as_tensor(next_states_b, dtype=torch.float32, device=device)
                dones_t = torch.as_tensor(dones_b, dtype=torch.float32, device=device)
                q_values = policy_net(states_t).gather(1, actions_t).squeeze()
                with torch.no_grad():
                    next_max = target_net(next_states_t).max(1)[0]
                    q_targets = rewards_t + (1 - dones_t) * gamma * next_max
                loss = nn.functional.mse_loss(q_values, q_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # 更新目标网络
                if global_step % target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                global_step += 1
        # 更新 ε
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        eps_history.append(epsilon)
        # 实时绘图
        if plot_live:
            ax1.clear()
            ax2.clear()
            ax1.plot(rewards_history, label="Episode Reward")
            if len(rewards_history) >= 10:
                # 滑动平均
                window = 10
                avg = np.convolve(rewards_history, np.ones(window)/window, mode="valid")
                ax1.plot(range(window-1, len(rewards_history)), avg, label="Avg Reward (10)")
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Total Reward")
            ax1.legend()
            ax2.plot(eps_history)
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Epsilon")
            ax2.set_ylim(0, 1)
            plt.pause(0.001)
        # 打印进度
        if ep % 10 == 0 or ep == 1:
            print(f"Episode {ep}/{num_episodes} | Total Reward: {total_reward:.4f} | Epsilon: {epsilon:.3f}")
    if plot_live:
        plt.ioff()
        plt.savefig("training_curve.png")
        plt.close()
    # 保存模型
    if save_path is not None:
        save_model(policy_net, save_path)
        print(f"模型已保存到 {save_path}")
    # 导出 ONNX
    if export_onnx_flag and onnx_path is not None:
        try:
            import onnx  # noqa: F401
            dummy = torch.randn(1, obs_dim)
            torch.onnx.export(policy_net.cpu(), dummy, onnx_path, input_names=["input"], output_names=["output"])
            print(f"已导出 ONNX 模型到 {onnx_path}")
        except Exception as e:
            print(f"ONNX 导出失败: {e}")
    return policy_net


def validate(
    env: gym.Env,
    model_path: str,
    hidden_sizes: List[int] | None = None,
    episodes: int = 10
) -> None:
    """
    使用训练好的模型在环境中运行一定数量的回合，打印奖励信息。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    model = DQN(obs_dim, act_dim, hidden_sizes).to(device)
    load_model(model, model_path)
    model.eval()
    total_rewards = []
    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = int(torch.argmax(model(state_t)).item())
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state
        total_rewards.append(ep_reward)
        print(f"Validate Episode {ep+1}/{episodes} | Total Reward: {ep_reward:.4f}")
    avg_reward = np.mean(total_rewards)
    print(f"平均奖励: {avg_reward:.4f}")

class ReplayBuffer:
    def __init__(self, capacity: int = 100000) -> None:
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )
    def __len__(self) -> int:
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int] | None = None) -> None:
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


def train_dqn(
    env: PHCAlphaBeta64Env,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    policy_net = DQN(obs_dim, act_dim, hidden_sizes=hidden_sizes).to(device)
    target_net = DQN(obs_dim, act_dim, hidden_sizes=hidden_sizes).to(device)
    if load_path is not None and os.path.exists(load_path):
        state_dict = torch.load(load_path, map_location="cpu")
        policy_net.load_state_dict(state_dict)
        print(f"从 {load_path} 载入模型继续训练")
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayBuffer(100000)
    epsilon = epsilon_start
    all_rewards: List[float] = []
    all_epsilons: List[float] = []
    if plot_live:
        plt.ion()
        fig, ax1 = plt.subplots(figsize=(8, 4))
        line_reward, = ax1.plot([], [], label="episode reward")
        window = 10
        line_avg, = ax1.plot([], [], label=f"moving average ({window})")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total reward")
        ax1.set_title("Training progress")
        ax2 = ax1.twinx()
        line_epsilon, = ax2.plot([], [], color="tab:red", label="epsilon")
        ax2.set_ylabel("Epsilon", color="tab:red")
        ax2.tick_params(axis='y', labelcolor="tab:red")
        lines = [line_reward, line_avg, line_epsilon]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right")
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
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
            if len(memory) >= batch_size:
                states, actions, rewards, next_states, dones = memory.sample(batch_size)
                states_t = torch.from_numpy(states).float().to(device)
                actions_t = torch.from_numpy(actions).long().to(device)
                rewards_t = torch.from_numpy(rewards).float().to(device)
                next_states_t = torch.from_numpy(next_states).float().to(device)
                dones_t = torch.from_numpy(dones).float().to(device)
                q_values = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q = target_net(next_states_t).max(1)[0]
                    target = rewards_t + gamma * (1.0 - dones_t) * max_next_q
                loss = nn.functional.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        all_rewards.append(total_reward)
        all_epsilons.append(epsilon)
        if (episode + 1) % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        print(f"Episode {episode+1}/{num_episodes} | total reward: {total_reward:.4f} | epsilon: {epsilon:.3f}")
        if (episode + 1) % 10 == 0 and len(all_rewards) >= 10:
            avg_r = sum(all_rewards[-10:]) / 10
            print(f" -- Average reward (last 10 episodes): {avg_r:.4f}")
        if plot_live:
            x = list(range(1, len(all_rewards)+1))
            line_reward.set_data(x, all_rewards)
            if len(all_rewards) >= window:
                moving_avg = [
                    sum(all_rewards[i - window : i]) / window
                    for i in range(window, len(all_rewards) + 1)
                ]
                line_avg.set_data(list(range(window, len(all_rewards)+1)), moving_avg)
            line_epsilon.set_data(x, all_epsilons)
            ax1.relim(); ax1.autoscale_view(); ax2.relim(); ax2.autoscale_view()
            plt.pause(0.001)
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
                plt.plot(range(window, len(all_rewards)+1), moving_avg, label=f"moving average ({window})")
            plt.xlabel("Episode")
            plt.ylabel("Total reward")
            plt.title("Training reward over episodes")
            plt.legend(); plt.tight_layout(); plt.savefig("training_curve_phc_ab.png"); plt.close()
        else:
            plt.ioff(); plt.savefig("training_curve_phc_ab.png"); plt.close()
    if save_path is not None:
        torch.save(policy_net.cpu().state_dict(), save_path)
    if onnx_path is not None and export_onnx_flag:
        if importlib.util.find_spec("onnx") is None:
            print("ONNX 库未安装，跳过导出 ONNX 文件。")
        else:
            dummy_input = torch.zeros(1, obs_dim)
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
            torch.onnx.export(policy_net.cpu(), dummy_input, onnx_path, input_names=["state"], output_names=["q_values"], opset_version=11)
            print(f"已将模型导出为 {onnx_path}")
    return policy_net


def main() -> None:
    env = PHCAlphaBeta64Env(
        v_dc=300.0,
        sample_time=1e-5,
        ref_frequency=50.0,
        ref_current=10.0,
        episode_length=2000,
        seed=0,
        switch_penalty_lf=0.02,  # LF 支路开关惩罚
        switch_penalty_hf=0.0,  # HF 支路开关惩罚（可设较小）
        harmonic_penalty=0.0,  # 谐波惩罚系数
        harmonic_window=20  # 谐波计算窗口
    )
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    model_pth = os.path.join(save_dir, "phc_ab_policy.pth")
    model_onnx = os.path.join(save_dir, "phc_ab_policy.onnx")
    train_dqn(
        env,
        num_episodes=100,
        batch_size=64,
        hidden_sizes=[64, 128, 128, 64],
        save_path=model_pth,
        onnx_path=model_onnx,
        plot_live=False,
    )
    # 使用 validate 可以验证模型效果
    # env.reset(); validate_model(...)


if __name__ == "__main__":
    main()