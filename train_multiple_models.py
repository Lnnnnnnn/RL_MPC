"""
批量训练三相逆变器强化学习模型的脚本
========================================

本脚本用于在命令行下批量训练多个模型，并能够方便调整
超参数、保存模型以及生成不同的模型文件。通过传递不同的
参数，可以训练出不同网络结构、探索策略或奖励权重的模型
，用于后续对比和选择。示例使用方法：

.. code-block:: bash

   python train_multiple_models.py \
       --episodes 300 \
       --batch-size 64 \
       --epsilon-decay 0.99 \
       --hidden-sizes 256 256 \
       --switch-penalty 0.01 \
       --harmonic-penalty 0.0 \
       --model-name model_A

这样将训练 300 个回合，使用隐藏层 [256, 256]，开关惩罚
权重为 0.01，并将模型保存在 `saved_models/model_A.pth` 和
`saved_models/model_A.onnx` 中。
"""

import argparse
import os
from typing import List

import pandas as pd  # 新增：用于把超参数记录到 Excel

from three_phase_inverter_rl import (
    ThreePhaseInverterEnv,
    train_dqn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量训练三相逆变器强化学习模型")
    parser.add_argument("--episodes", type=int, default=200, help="训练回合数")
    parser.add_argument("--batch-size", type=int, default=64, help="批量大小")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="ε 衰减因子")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="ε 初始值")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="ε 最小值")
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="*",
        default=[128, 128],
        help="隐藏层神经元数量列表，如 128 128",
    )
    parser.add_argument("--switch-penalty", type=float, default=0.0, help="开关惩罚系数")
    parser.add_argument("--harmonic-penalty", type=float, default=0.0, help="谐波惩罚系数")
    parser.add_argument("--harmonic-window", type=int, default=20, help="谐波窗口长度")
    parser.add_argument(
        "--model-name", type=str, required=True, help="模型名称，用于保存不同模型"
    )
    # 是否实时绘制奖励和 epsilon 曲线
    parser.add_argument(
        "--plot-live",
        action="store_true",
        help="启用实时绘图功能，训练过程中动态展示奖励和 epsilon 曲线",
    )
    # 是否跳过导出 ONNX 文件
    parser.add_argument(
        "--no-onnx",
        action="store_true",
        help="不导出 ONNX 模型文件（当系统未安装 onnx 库时可使用该选项）",
    )
    parser.add_argument("--excel_path", type=str, default="log_training_runs.xlsx", help="用于记录超参数的 Excel 文件路径(.xlsx)")
    return parser.parse_args()

def log_run_to_excel(args: argparse.Namespace, excel_path: str) -> None:
    """把命令行参数追加写入一个 Excel 文件。"""
    row_dict = vars(args).copy()
    # 隐藏层列表转成字符串，方便在表格里查看
    row_dict["hidden_sizes"] = " ".join(map(str, row_dict["hidden_sizes"]))

    df_new = pd.DataFrame([row_dict])
    if os.path.exists(excel_path):
        try:
            df_exist = pd.read_excel(excel_path)
            df_out = pd.concat([df_exist, df_new], ignore_index=True)
        except Exception:
            # 如果文件损坏或格式不符，直接用新表覆盖
            df_out = df_new
    else:
        df_out = df_new
    df_out.to_excel(excel_path, index=False)



def main() -> None:
    args = parse_args()
    # 创建环境，并传入开关和谐波惩罚权重
    env = ThreePhaseInverterEnv(
        v_dc=400.0,
        inductance=2e-3,
        resistance=1.0,
        sample_time=5e-5,
        ref_frequency=50.0,
        ref_current=10.0,
        episode_length=2000,
        seed=0,
        switch_penalty=args.switch_penalty,
        harmonic_penalty=args.harmonic_penalty,
        harmonic_window=args.harmonic_window,
    )
    # 构造保存路径
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    model_pth = os.path.join(save_dir, f"{args.model_name}.pth")
    model_onnx = os.path.join(save_dir, f"{args.model_name}.onnx")
    # 训练模型
    export_onnx_flag = not args.no_onnx
    train_dqn(
        env,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        hidden_sizes=args.hidden_sizes,
        save_path=model_pth,
        onnx_path=model_onnx,
        load_path="saved_models/model_A.pth", # None
        plot_live=args.plot_live,
        export_onnx_flag=export_onnx_flag,
    )

    # 训练完成后记录参数到 Excel
    log_run_to_excel(args, args.excel_path)
    print(f"记录超参数的 Excel 文件保存在: {args.excel_path}")

    print(
        f"模型训练完成，权重保存在 {model_pth}，ONNX 模型保存在 {model_onnx}"
    )


if __name__ == "__main__":
    main()