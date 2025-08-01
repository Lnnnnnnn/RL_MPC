import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from numpy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import trange

class ThreePhaseInverterEnv(gym.Env):
    """
    两电平三相逆变器 + L 滤波器并网
    动作 0‥7 → (Sa,Sb,Sc)；奖励 = -MSE(i, i_ref)
    """
    metadata = {"render_modes": ["live"]}  # 声明支持 live 渲染

    def __init__(
        self,
        Vdc=700.0,
        L=2e-3,
        f_grid=50.0,
        Vg_rms=230.0,
        I_ref_amp=10.0,
        horizon=2000,
        dt=50e-6,
    ):
        super().__init__()

        # ——— 电路与控制常量 ———
        self.Vdc = Vdc
        self.L = L
        self.omega = 2 * np.pi * f_grid
        self.Vg_peak = Vg_rms * np.sqrt(2)
        self.I_ref_amp = I_ref_amp
        self.dt = dt
        self.horizon = horizon

        # ——— Gym 必需成员 ———
        self.state = np.zeros(3, dtype=np.float32)
        self._time = 0.0
        self.t = 0

        self.switch_table = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=int,
        )

        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

        # ——— 用于 THD 统计 ———
        self.cycle_len = int(1 / f_grid / dt)
        self.curr_history = []

        # ——— 用于实时绘图 ———
        self._render_initialized = False
        self._plot_len = 800  # 画最近多少个采样点（约 40 ms）
        self._time_trace = np.zeros(self._plot_len)
        self._i_trace = np.zeros((self._plot_len, 3))
        self._i_ref_trace = np.zeros((self._plot_len, 3))

    # ---------- 工具函数 ----------
    def _switch_to_phase_voltage(self, sw):
        return (2 * sw - 1) * self.Vdc / 2  # ±Vdc/2

    def _grid_voltage(self, t):
        ang = self.omega * t
        return self.Vg_peak * np.sin(
            ang + np.array([0, -2 * np.pi / 3, 2 * np.pi / 3])
        )

    def _reference_current(self, t):
        ang = self.omega * t
        return self.I_ref_amp * np.sin(
            ang + np.array([0, -2 * np.pi / 3, 2 * np.pi / 3])
        )

    # ---------- Gym 接口 ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state.fill(0.0)
        self._time = 0.0
        self.t = 0
        self.curr_history.clear()
        return self.state.copy(), {}

    def step(self, action: int):
        sw = self.switch_table[action]
        v_inv = self._switch_to_phase_voltage(sw)
        v_grid = self._grid_voltage(self._time)

        di = (v_inv - v_grid) / self.L
        self.state = self.state + di * self.dt

        i_ref = self._reference_current(self._time)
        mse = np.mean((self.state - i_ref) ** 2)
        reward = -mse

        # 记录
        self.curr_history.append(self.state.copy())
        idx = self.t % self._plot_len
        self._time_trace[idx] = self._time
        self._i_trace[idx] = self.state
        self._i_ref_trace[idx] = i_ref

        self.t += 1
        self._time += self.dt
        terminated = False
        truncated = self.t >= self.horizon

        info = {}
        if truncated:
            # 粗算 THD
            i_arr = np.array(self.curr_history)[-self.cycle_len :]
            thd = []
            for ph in range(3):
                s = i_arr[:, ph]
                fft = np.abs(rfft(s))
                freqs = rfftfreq(len(s), self.dt)
                f1 = fft[np.argmin(np.abs(freqs - 50))]
                vrms = np.sqrt(np.mean(s ** 2))
                thd.append(np.sqrt(vrms**2 - (f1 / np.sqrt(2)) ** 2) / (f1 / np.sqrt(2)))
            info["THD_mean"] = float(np.mean(thd))
        return self.state.copy(), reward, terminated, truncated, info

    # ---------- 实时渲染 ----------
    def render(self, mode="live"):
        if mode != "live":
            raise NotImplementedError

        if not self._render_initialized:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 4))
            (self.line_ia,) = self.ax.plot([], [], label="i_a")
            (self.line_ib,) = self.ax.plot([], [], label="i_b")
            (self.line_ic,) = self.ax.plot([], [], label="i_c")
            (self.line_ref,) = self.ax.plot([], [], "k--", label="i_ref (phase-a)")
            self.ax.set_xlabel("time [s]")
            self.ax.set_ylabel("current [A]")
            self.ax.set_ylim(-1.2 * self.I_ref_amp, 1.2 * self.I_ref_amp)
            self.ax.legend(loc="upper right")
            self.fig.tight_layout()
            self._render_initialized = True

        # 取循环缓冲区里“最近一段”
        idx = np.arange(self._plot_len)
        t_window = self._time_trace
        i_window = self._i_trace
        i_ref_window = self._i_ref_trace[:, 0]  # 只画参考相 a

        self.line_ia.set_data(t_window, i_window[:, 0])
        self.line_ib.set_data(t_window, i_window[:, 1])
        self.line_ic.set_data(t_window, i_window[:, 2])
        self.line_ref.set_data(t_window, i_ref_window)

        self.ax.set_xlim(t_window.min(), t_window.max())
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# ============ Demo: 随机动作 + 实时波形 ============
if __name__ == "__main__":
    env = ThreePhaseInverterEnv()
    obs, _ = env.reset()
    steps = 4000  # 0.2 s
    for _ in trange(steps, desc="Simulating"):
        action = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(action)
        env.render()
        if term or trunc:
            break
    plt.ioff()
    plt.show()
