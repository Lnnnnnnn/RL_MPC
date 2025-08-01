import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from numpy.fft import rfft, rfftfreq
from tqdm import trange
from scipy.signal import welch

class ThreePhaseInverterEnv(gym.Env):
    """
    两电平三相逆变器 (NPC 不考虑中点偏移) + L 滤波器 + 理想电网
    状态: 电感电流 i_a, i_b, i_c  (A)
    动作: 8 种开关矢量 0~7, 对应 (S_a,S_b,S_c) 三个桥臂 0/1
    奖励: -MSE(i_abc, i_ref_abc) 逐步给; 终局再用 THD 作 info.
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 Vdc=700.0,          # 直流母线电压 [V]
                 L=2e-3,             # 滤波电感 [H]
                 f_grid=50.0,        # 电网频率 [Hz]
                 Vg_rms=230.0,       # 相电压有效值 [V]
                 I_ref_amp=10.0,     # 目标电流幅值 [A]
                 horizon=2000,       # 每回合时间步
                 dt=50e-6):          # 采样周期 [s]
        super().__init__()
        # ----- 常量 -----
        self.Vdc = Vdc
        self.L = L
        self.omega = 2*np.pi*f_grid
        self.Vg_peak = Vg_rms*np.sqrt(2)
        self.I_ref_amp = I_ref_amp
        self.dt = dt
        self.horizon = horizon

        # 状态与计数器
        self.state = np.zeros(3, dtype=np.float32)
        self.t = 0
        self._time = 0.0

        # 动作 0..7 映射到三相桥臂 (Sa,Sb,Sc)
        self.switch_table = np.array(
            [[0,0,0],[0,0,1],[0,1,0],[0,1,1],
             [1,0,0],[1,0,1],[1,1,0],[1,1,1]], dtype=int)

        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(3,), dtype=np.float32)

        # 为计算 THD 收集一个网周期间 (cycle_len=1/f_grid/dt) 的电流
        self.cycle_len = int(1/f_grid/dt)
        self.curr_history = []

        # —— 渲染相关 —— #
        self._render_initialized = False
        self._plot_len = 800                       # 显示最近多少个采样点
        self._time_buf = np.zeros(self._plot_len)
        self._i_buf    = np.zeros((self._plot_len, 3))
        self._i_ref_buf = np.zeros(self._plot_len) # 只画 a 相参考

    # ---------- 帮助函数 ----------
    def _switch_to_phase_voltage(self, sw):
        # sw: ndarray (Sa,Sb,Sc) ∈ {0,1}
        return (2*sw - 1) * self.Vdc/2  # Va,Vb,Vc

    def _grid_voltage(self, t):
        # 理想正弦电网相电压
        angle = self.omega * t
        Va = self.Vg_peak * np.sin(angle)
        Vb = self.Vg_peak * np.sin(angle - 2*np.pi/3)
        Vc = self.Vg_peak * np.sin(angle + 2*np.pi/3)
        return np.array([Va, Vb, Vc])

    def _reference_current(self, t):
        angle = self.omega * t
        Ia = self.I_ref_amp * np.sin(angle)
        Ib = self.I_ref_amp * np.sin(angle - 2*np.pi/3)
        Ic = self.I_ref_amp * np.sin(angle + 2*np.pi/3)
        return np.array([Ia, Ib, Ic])

    # ---------- Gym 接口 ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(3, dtype=np.float32)
        self.t = 0
        self._time = 0.0
        self.curr_history.clear()
        return self.state.copy(), {}

    def step(self, action:int):
        # 1. 解析动作 -> 逆变器相电压
        sw = self.switch_table[action]
        v_inv = self._switch_to_phase_voltage(sw)

        # 2. 网侧电压
        v_grid = self._grid_voltage(self._time)

        # 3. 一阶欧拉: L di/dt = v_inv - v_grid
        di = (v_inv - v_grid) / self.L
        self.state = self.state + di * self.dt  # i_next

        # 4. 奖励 (负均方误差)
        i_ref = self._reference_current(self._time)
        scale = (self.I_ref_amp) ** 2 * 10  # 理论最大 MSE
        mse = np.mean((self.state - i_ref) ** 2) / scale
        reward = -mse  # 量级 ≈ [-1, 0]

        # 5. 记录
        self.curr_history.append(self.state.copy())

        idx = self.t % self._plot_len
        self._time_buf[idx]   = self._time
        self._i_buf[idx]      = self.state
        self._i_ref_buf[idx]  = i_ref[0]           # 参考 a 相

        self.t += 1
        self._time += self.dt
        terminated = False
        truncated = self.t >= self.horizon

        info = {}
        if truncated:
            i_arr = np.asarray(self.curr_history)[-self.cycle_len:]  # shape=(cycle_len,3)
            thd_vals = []
            for ph in range(3):  # a, b, c 三相
                signal = i_arr[:, ph]  # 这一相的电流波形
                fs = 1 / self.dt  # 采样频率, 例 20 kHz
                freqs, psd = welch(signal, fs=fs, nperseg=len(signal))

                # 找离 50 Hz 最近的频点作为基波
                f1_idx = np.argmin(np.abs(freqs - 50))
                # 基波 RMS = PSD(f1) × 带宽，再开方
                bandwidth = freqs[1] - freqs[0]  # 相邻频点间隔
                fundamental_rms = np.sqrt(psd[f1_idx] * bandwidth)

                total_rms = np.sqrt(np.trapz(psd, freqs))  # PSD 积分求总体 RMS
                harm_rms = np.sqrt(max(total_rms ** 2 - fundamental_rms ** 2, 0.0))
                # 加 1e-12 防 0 除
                thd_vals.append(harm_rms / (fundamental_rms + 1e-12))

                # print({
                #     "phase": ph,
                #     "vrms": float(total_rms),
                #     "fund_rms": float(fundamental_rms),
                #     "harm_rms": float(harm_rms),
                # })

            info["THD_mean"] = float(np.mean(thd_vals))  # 传回评估脚本

        return self.state.copy(), reward, terminated, truncated, info

    def render(self, mode="live"):
        # 只支持实时曲线
        if mode != "live":
            raise NotImplementedError("Only mode='live' is supported")

        import matplotlib.pyplot as plt

        # ——— 首次调用：建图、曲线句柄 ———
        if not self._render_initialized:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 4))

            (self.line_ia,)   = self.ax.plot([], [], label="i_a")
            (self.line_ib,)   = self.ax.plot([], [], label="i_b")
            (self.line_ic,)   = self.ax.plot([], [], label="i_c")
            (self.line_ref,)  = self.ax.plot([], [], "k--", label="i_ref (phase-a)")

            self.ax.set_xlabel("time [s]")
            self.ax.set_ylabel("current [A]")
            ylim = 1.2 * self.I_ref_amp
            self.ax.set_ylim(-ylim, ylim)
            self.ax.legend(loc="upper right")
            self.fig.tight_layout()
            self._render_initialized = True

        # ——— 更新数据 ———
        t   = self._time_buf
        cur = self._i_buf
        ref = self._i_ref_buf

        self.line_ia.set_data(t, cur[:, 0])
        self.line_ib.set_data(t, cur[:, 1])
        self.line_ic.set_data(t, cur[:, 2])
        self.line_ref.set_data(t, ref)

        # x 轴始终显示完整缓冲区
        self.ax.set_xlim(t.min(), t.max())

        # 刷新
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# ------------------ 训练脚本 ------------------
if __name__ == "__main__":
    env = ThreePhaseInverterEnv()

    # DQN 对离散 8 动作非常合适
    model = DQN(
        policy="MlpPolicy",
        env=env,
        device="cuda",
        learning_rate=5e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=512,
        verbose=1,
        gamma=0.99,
        target_update_interval=500,
        train_freq=4,
    )

    model.learn(total_timesteps=200_000, progress_bar=True)   # 例：10 秒仿真 × 20 kHz = 200 000 步

    # ------------- 离线评估 -------------

    episodes = 10
    thd_list = []

    for ep in trange(episodes, desc="Evaluating episodes"):
        warm_steps = int(1.5 / 50 / env.dt)
        for _ in range(warm_steps):
            obs, _ = env.reset()
        for _ in range(4000):  # 跑 0.2 s
            action = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(action)
            env.render()  # 每步渲染
            if term or trunc:
                break
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(int(action))
            done = term or trunc
        thd_list.append(info["THD_mean"])
        # 也可以用 tqdm.write(...) 单独打印每回合指标

    print("THD list (%):", [round(v * 100, 2) for v in thd_list])
