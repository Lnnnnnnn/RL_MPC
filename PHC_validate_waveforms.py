"""
验证 64 状态 PHC 模型并绘制波形脚本
=====================================

本脚本用于加载训练好的 DQN 模型，在并联混合逆变器 64 状态
环境中运行若干个参考电流周期，并记录以下波形：

  - LF 支路（网侧）三相电流与参考电流曲线
  - HF 支路三相电流（目标为 0）
  - LF 支路三相开关电压
  - HF 支路三相开关电压
  - 网侧三相电压

最后将这些波形绘制为多子图。脚本支持命令行指定模型路径、仿真
周期数以及图像保存路径等参数。

示例用法：

.. code-block:: bash

    python validate_phc64_waveforms.py \
        --model-path saved_models/phc64_model.pth \
        --cycles 3 \
        --hidden-sizes 256 256 \
        --switch-penalty-lf 0.02 \
        --switch-penalty-hf 0.0 \
        --save-plot waveform_phc64.png

注意：验证时请确保传入的环境参数（如隐藏层结构、惩罚系数等）与
训练时保持一致，以便正确加载模型和复现控制效果。
"""

import argparse
import os
import math
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from PHC_inverter_rl import PHCAlphaBeta64Env, DQN, load_model


def inverse_clarke(alpha: float, beta: float, gamma: float = 0.0) -> Tuple[float, float, float]:
    """
    逆 Clarke 变换，将 αβγ 坐标下的电流或电压转换回 abc 坐标。
    若 gamma 为 0，可得到平衡三相系统的三相分量。
    """
    a = alpha + gamma
    b = -0.5 * alpha + (math.sqrt(3) / 2.0) * beta + gamma
    c = -0.5 * alpha - (math.sqrt(3) / 2.0) * beta + gamma
    return a, b, c


def simulate_waveforms(
    env: PHCAlphaBeta64Env,
    model: torch.nn.Module,
    cycles: int = 1,
) -> dict:
    """
    在环境中运行指定周期数并记录各类波形。

    返回包含时间轴、LF 电流、HF 电流、参考电流、LF/HF 开关电压
    以及网侧电压的字典。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    state = env.reset()
    # 计算总步数
    period = 1.0 / env.ref_freq
    total_steps = int(cycles * period / env.dt)
    # 存储容器
    data: dict[str, list] = {
        "t": [],
        "i_g_a": [], "i_g_b": [], "i_g_c": [],
        "i_ref_a": [], "i_ref_b": [], "i_ref_c": [],
        "i_hf_a": [], "i_hf_b": [], "i_hf_c": [],
        "v_lf_a": [], "v_lf_b": [], "v_lf_c": [],
        "v_hf_a": [], "v_hf_b": [], "v_hf_c": [],
        "v_g_a": [], "v_g_b": [], "v_g_c": [],
    }
    for step in range(total_steps):
        data["t"].append(env.time)
        # 从状态中提取 αβγ 电流
        I_G_alpha = float(state[0])
        I_HF_alpha = float(state[1])
        I_G_beta = float(state[2])
        I_HF_beta = float(state[3])
        I_HF_gamma = float(state[4])
        # 将 αβγ 转换到 abc
        i_g_a, i_g_b, i_g_c = inverse_clarke(I_G_alpha, I_G_beta)
        i_hf_a, i_hf_b, i_hf_c = inverse_clarke(I_HF_alpha, I_HF_beta, I_HF_gamma)
        data["i_g_a"].append(i_g_a)
        data["i_g_b"].append(i_g_b)
        data["i_g_c"].append(i_g_c)
        data["i_hf_a"].append(i_hf_a)
        data["i_hf_b"].append(i_hf_b)
        data["i_hf_c"].append(i_hf_c)
        # 参考电流 αβ (HF 参考为 0)
        I_alpha_ref, I_beta_ref = env._reference_current_ab(env.time)
        ref_a, ref_b, ref_c = inverse_clarke(I_alpha_ref, I_beta_ref)
        data["i_ref_a"].append(ref_a)
        data["i_ref_b"].append(ref_b)
        data["i_ref_c"].append(ref_c)
        # 使用贪心策略选择动作
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
            q_values = model(state_t)
            action = int(torch.argmax(q_values).item())
        # 记录开关电压（LF 和 HF）
        (v_lf_a, v_lf_b, v_lf_c), (v_hf_a, v_hf_b, v_hf_c) = env._decode_two_branches(action)
        data["v_lf_a"].append(v_lf_a)
        data["v_lf_b"].append(v_lf_b)
        data["v_lf_c"].append(v_lf_c)
        data["v_hf_a"].append(v_hf_a)
        data["v_hf_b"].append(v_hf_b)
        data["v_hf_c"].append(v_hf_c)
        # 记录网侧电压 (将 αβ 转回 abc)
        V_alpha, V_beta = env._grid_voltage_ab(env.time)
        v_g_a, v_g_b, v_g_c = inverse_clarke(V_alpha, V_beta)
        data["v_g_a"].append(v_g_a)
        data["v_g_b"].append(v_g_b)
        data["v_g_c"].append(v_g_c)
        # 环境推进
        next_state, _, done, _ = env.step(action)
        state = next_state
        if done:
            break
    return data


def plot_waveforms(data: dict, save_path: str | None = None, show: bool = True) -> None:
    """
    绘制并联混合逆变器 LF/HF 电流、开关电压与网侧电压波形。

    - 第一行：LF 网侧电流和参考电流（各三相）。
    - 第二行：HF 支路电流（各三相）。
    - 第三行：LF 支路开关电压（各三相）。
    - 第四行：HF 支路开关电压（各三相）。
    - 第五行：网侧电压（各三相）。
    """
    t = data["t"]
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    # LF 电流与参考
    axs[0].plot(t, data["i_g_a"], label="I_Ga")
    axs[0].plot(t, data["i_ref_a"], "--", label="I_ref_a")
    axs[0].plot(t, data["i_g_b"], label="I_Gb")
    axs[0].plot(t, data["i_ref_b"], "--", label="I_ref_b")
    axs[0].plot(t, data["i_g_c"], label="I_Gc")
    axs[0].plot(t, data["i_ref_c"], "--", label="I_ref_c")
    axs[0].set_ylabel("LF Current (A)")
    axs[0].legend(loc="upper right", ncol=3, fontsize=8)
    # HF 电流
    axs[1].plot(t, data["i_hf_a"], label="I_HFa")
    axs[1].plot(t, data["i_hf_b"], label="I_HFb")
    axs[1].plot(t, data["i_hf_c"], label="I_HFc")
    axs[1].set_ylabel("HF Current (A)")
    axs[1].legend(loc="upper right", ncol=3, fontsize=8)
    # LF 电压
    axs[2].plot(t, data["v_lf_a"], label="V_LFa")
    axs[2].plot(t, data["v_lf_b"], label="V_LFb")
    axs[2].plot(t, data["v_lf_c"], label="V_LFc")
    axs[2].set_ylabel("LF Voltage (V)")
    axs[2].legend(loc="upper right", ncol=3, fontsize=8)
    # HF 电压
    axs[3].plot(t, data["v_hf_a"], label="V_HFa")
    axs[3].plot(t, data["v_hf_b"], label="V_HFb")
    axs[3].plot(t, data["v_hf_c"], label="V_HFc")
    axs[3].set_ylabel("HF Voltage (V)")
    axs[3].legend(loc="upper right", ncol=3, fontsize=8)
    # 网侧电压
    axs[4].plot(t, data["v_g_a"], label="V_Ga")
    axs[4].plot(t, data["v_g_b"], label="V_Gb")
    axs[4].plot(t, data["v_g_c"], label="V_Gc")
    axs[4].set_ylabel("Grid Voltage (V)")
    axs[4].set_xlabel("Time (s)")
    axs[4].legend(loc="upper right", ncol=3, fontsize=8)
    fig.suptitle("PHC 64-State Waveforms")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path)
        print(f"波形图已保存到 {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 PHC64 模型并绘制波形")
    parser.add_argument("--model-path", type=str, required=True, help="要加载的模型 .pth 路径")
    parser.add_argument("--cycles", type=int, default=1, help="仿真的参考电流周期数 (>=1)")
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="*",
        default=[128, 128],
        help="隐藏层结构，应与训练时一致，例如 256 256",
    )
    parser.add_argument("--v-dc", type=float, default=300.0, help="直流母线电压 (V)")
    parser.add_argument("--sample-time", type=float, default=1e-5, help="采样时间步长 (s)")
    parser.add_argument("--ref-frequency", type=float, default=50.0, help="参考电流频率 (Hz)")
    parser.add_argument("--ref-current", type=float, default=10.0, help="参考电流幅值 (A)")
    parser.add_argument("--episode-length", type=int, default=2000, help="单回合步数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--switch-penalty-lf", type=float, default=0.0, help="LF 支路开关惩罚系数")
    parser.add_argument("--switch-penalty-hf", type=float, default=0.0, help="HF 支路开关惩罚系数")
    parser.add_argument("--harmonic-penalty", type=float, default=0.0, help="谐波惩罚系数")
    parser.add_argument("--harmonic-window", type=int, default=20, help="谐波窗口长度")
    parser.add_argument("--save-plot", type=str, default=None, help="波形图保存路径，可选")
    parser.add_argument("--no-show", action="store_true", help="生成图像但不显示")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 构建环境，确保与训练时参数一致
    env = PHCAlphaBeta64Env(
        v_dc=args.v_dc,
        sample_time=args.sample_time,
        ref_frequency=args.ref_frequency,
        ref_current=args.ref_current,
        episode_length=args.episode_length,
        seed=args.seed,
        switch_penalty_lf=args.switch_penalty_lf,
        switch_penalty_hf=args.switch_penalty_hf,
        harmonic_penalty=args.harmonic_penalty,
        harmonic_window=args.harmonic_window,
    )
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    model = DQN(obs_dim, act_dim, hidden_sizes=args.hidden_sizes)
    load_model(model, args.model_path)
    model.to(device)
    # 运行仿真
    data = simulate_waveforms(env, model, cycles=args.cycles)
    # 绘制
    plot_waveforms(data, save_path=args.save_plot, show=not args.no_show)


if __name__ == "__main__":
    main()