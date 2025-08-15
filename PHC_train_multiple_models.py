"""
命令行脚本：在 64 状态的并联混合逆变器环境中训练强化学习模型。

该脚本基于 `phc_alpha_beta64_only_rl.py` 中实现的 `PHCAlphaBeta64Env`
和 `train_dqn` 函数，允许从命令行灵活设置训练参数、开关惩罚、
谐波惩罚等，并保存训练后的模型（.pth）和可选 ONNX 文件。

示例用法：

.. code-block:: bash

   python train_phc64_model.py \
       --episodes 300 \
       --batch-size 64 \
       --epsilon-decay 0.99 \
       --hidden-sizes 256 256 \
       --switch-penalty-lf 0.02 \
       --switch-penalty-hf 0.0 \
       --harmonic-penalty 0.001 \
       --model-name phc64_model

运行后将在 `saved_models/` 目录下生成 `phc64_model.pth` 和 `phc64_model.onnx`。
如需跳过 ONNX 导出，可加 `--no-onnx`。
"""

import argparse
import os
from typing import List

from PHC_inverter_rl import PHCAlphaBeta64Env, train_dqn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 64 状态 PHC RL 模型")
    # 训练超参数
    parser.add_argument("--episodes", type=int, default=200, help="训练回合数")
    parser.add_argument("--batch-size", type=int, default=64, help="批量大小")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="ε 衰减因子")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="ε 初始值")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="ε 最小值")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--target-update", type=int, default=500, help="目标网络更新频率")
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="*",
        default=[128, 128],
        help="隐藏层神经元列表，例如 128 128",
    )
    # 环境参数
    parser.add_argument("--v-dc", type=float, default=300.0, help="直流母线电压")
    parser.add_argument("--sample-time", type=float, default=1e-5, help="采样时间步长 (s)")
    parser.add_argument("--ref-frequency", type=float, default=50.0, help="参考电流频率 (Hz)")
    parser.add_argument("--ref-current", type=float, default=10.0, help="参考电流幅值 (A)")
    parser.add_argument("--episode-length", type=int, default=2000, help="单回合时间步数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    # 开关与谐波惩罚
    parser.add_argument("--switch-penalty-lf", type=float, default=0.0, help="低频支路开关惩罚系数")
    parser.add_argument("--switch-penalty-hf", type=float, default=0.0, help="高频支路开关惩罚系数")
    parser.add_argument("--harmonic-penalty", type=float, default=0.0, help="谐波惩罚系数")
    parser.add_argument("--harmonic-window", type=int, default=20, help="谐波计算窗口长度")
    # 模型保存
    parser.add_argument("--model-name", type=str, required=True, help="模型名称（不含扩展名）")
    parser.add_argument("--plot-live", action="store_true", help="训练时实时绘制奖励和 ε 曲线")
    parser.add_argument("--no-onnx", action="store_true", help="不导出 ONNX 文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 实例化环境
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
    # 构造保存路径
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    model_pth = os.path.join(save_dir, f"{args.model_name}.pth")
    model_onnx = os.path.join(save_dir, f"{args.model_name}.onnx")
    # 训练
    export_onnx_flag = not args.no_onnx
    train_dqn(
        env,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        target_update=args.target_update,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        hidden_sizes=args.hidden_sizes,
        save_path=model_pth,
        onnx_path=model_onnx,
        load_path=None,
        plot_live=args.plot_live,
        export_onnx_flag=export_onnx_flag,
    )
    print(f"模型训练完成，保存在 {model_pth}，ONNX 保存为 {model_onnx}")


if __name__ == "__main__":
    main()