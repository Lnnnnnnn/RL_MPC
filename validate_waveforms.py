"""
验证模型并绘制波形脚本
========================

本脚本用于加载训练好的 DQN 模型，在自定义的三相逆变器
环境中运行若干个电流周期，并实时记录三相电流、参考电
流和逆变器输出电压。最后绘制波形图，方便用户评估控
制效果。脚本支持命令行参数自定义模型路径、仿真周期
数量、输出图片保存路径等。

示例用法：

.. code-block:: bash

    # 运行 3 个周期并保存波形图
    python validate_waveforms.py \
        --model-path saved_models/model_A.pth \
        --cycles 3 \
        --save-plot waveform_A.png

"""

import argparse
import math
import os
from typing import List

import numpy as np
import torch
import matplotlib.pyplot as plt

from three_phase_inverter_rl import ThreePhaseInverterEnv, DQN, load_model


def simulate_waveforms(
    env: ThreePhaseInverterEnv,
    model: torch.nn.Module,
    cycles: int = 1,
) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float], list[float]]:
    """运行指定周期数并记录电流、电压和参考电流。

    参数：
      - env: 已初始化的环境实例；其参考频率和采样时间决定周期长度。
      - model: 加载的策略网络；将采用贪心策略选择动作。
      - cycles: 记录的参考周期数；整数值≥1。

    返回：
      (t, i_a_list, i_b_list, i_c_list, ref_a_list, v_a_list, v_b_list, v_c_list)
      其中 t 为时间轴（秒）。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    # 重置环境
    state = env.reset()
    # 计算需要的步数：周期数 * 周期长度 / 采样时间
    period = 1.0 / env.ref_freq
    total_steps = int(cycles * period / env.dt)
    # 存储列表
    t_list: List[float] = []
    i_a_list: List[float] = []
    i_b_list: List[float] = []
    i_c_list: List[float] = []
    ref_a_list: List[float] = []
    ref_b_list: List[float] = []
    ref_c_list: List[float] = []
    v_a_list: List[float] = []
    v_b_list: List[float] = []
    v_c_list: List[float] = []
    # 循环采样
    for step in range(total_steps):
        # 当前时间
        t_list.append(env.time)
        # 当前状态拆分
        i_a, i_b, i_c, ref_a, ref_b, ref_c = state
        i_a_list.append(float(i_a))
        i_b_list.append(float(i_b))
        i_c_list.append(float(i_c))
        ref_a_list.append(float(ref_a))
        ref_b_list.append(float(ref_b))
        ref_c_list.append(float(ref_c))
        # 通过模型选择动作（贪心）
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
            q_values = model(state_t)
            action = int(torch.argmax(q_values, dim=1).item())
        # 根据动作计算输出电压，以记录波形；利用环境的内部函数
        v_a, v_b, v_c = env._switch_to_voltage(action)
        v_a_list.append(float(v_a))
        v_b_list.append(float(v_b))
        v_c_list.append(float(v_c))
        # 与环境交互，更新状态
        next_state, _, done, _ = env.step(action)
        state = next_state
        if done:
            break
    return (
        t_list,
        i_a_list,
        i_b_list,
        i_c_list,
        ref_a_list,
        ref_b_list,
        ref_c_list,
        v_a_list,
        v_b_list,
        v_c_list,
    )


def plot_waveforms(
    t: list[float],
    i_a: list[float],
    i_b: list[float],
    i_c: list[float],
    ref_a: list[float],
    ref_b: list[float],
    ref_c: list[float],
    v_a: list[float],
    v_b: list[float],
    v_c: list[float],
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """绘制电流和电压波形。

    将电流和参考电流绘制在三个子图中，再绘制电压波形。
    如果指定保存路径，则将图像保存为文件。
    """
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    # 三相电流及参考
    axs[0].plot(t, i_a, label="i_a")
    axs[0].plot(t, ref_a, "--", label="i_ref_a")
    axs[0].set_ylabel("Current A (A)")
    axs[0].legend()
    axs[1].plot(t, i_b, label="i_b")
    axs[1].plot(t, ref_b, "--", label="i_ref_b")
    axs[1].set_ylabel("Current B (A)")
    axs[1].legend()
    axs[2].plot(t, i_c, label="i_c")
    axs[2].plot(t, ref_c, "--", label="i_ref_c")
    axs[2].set_ylabel("Current C (A)")
    axs[2].set_xlabel("Time (s)")
    axs[2].legend()
    # 三相电压
    axs[3].plot(t, v_a, label="v_a")
    axs[3].plot(t, v_b, label="v_b")
    axs[3].plot(t, v_c, label="v_c")
    axs[3].set_ylabel("Voltage (V)")
    axs[3].set_xlabel("Time (s)")
    axs[3].legend()
    fig.suptitle("Current and Voltage Waveforms")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        # 如果 save_path 中不包含目录，则直接使用当前目录
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path)
        print(f"波形图已保存到 {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="加载模型并绘制波形")
    parser.add_argument(
        "--model-path", type=str, required=True, help="要加载的模型权重 .pth 路径"
    )
    parser.add_argument(
        "--cycles", type=int, default=1, help="仿真的参考电流周期数"
    )
    parser.add_argument(
        "--save-plot", type=str, default=None, help="保存波形图的路径，可选"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="生成图像但不展示（通常与 --save-plot 一起使用）",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="*",
        default=None,
        help="隐藏层尺寸列表，需要与训练时一致，如 256 256。如不指定则使用默认架构 [128,128]。",
    )
    args = parser.parse_args()
    # 创建与训练时相同参数的环境
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
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    model = DQN(obs_dim, act_dim, hidden_sizes=args.hidden_sizes)
    load_model(model, args.model_path)
    model.to(device)
    # 运行仿真
    (
        t_list,
        i_a_list,
        i_b_list,
        i_c_list,
        ref_a_list,
        ref_b_list,
        ref_c_list,
        v_a_list,
        v_b_list,
        v_c_list,
    ) = simulate_waveforms(env, model, cycles=args.cycles)
    # 绘制波形图
    plot_waveforms(
        t_list,
        i_a_list,
        i_b_list,
        i_c_list,
        ref_a_list,
        ref_b_list,
        ref_c_list,
        v_a_list,
        v_b_list,
        v_c_list,
        save_path=args.save_plot,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()