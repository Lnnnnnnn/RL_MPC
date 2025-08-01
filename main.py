import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env


class LinearStateSpaceEnv(gym.Env):
    """
    x_{k+1} = A x_k + B u_k           (discrete-time LTI)
    r_k     = - (xᵀ Q x + uᵀ R u)     (quadratic cost as reward)
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 A=np.array([[1.0, 0.1],
                             [0.0, 1.0]]),
                 B=np.array([[0.0],
                             [0.1]]),
                 Q=np.diag([1.0, 0.1]),
                 R=np.diag([0.01]),
                 x0=np.array([2.0, 0.0]),
                 horizon=200):
        super().__init__()

        self.A, self.B = A, B
        self.Q, self.R = Q, R
        self.x0 = x0.astype(np.float32)
        self.state = self.x0.copy()
        self.horizon = horizon
        self.t = 0

        # 连续动作空间 u ∈ [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(B.shape[1],), dtype=np.float32)
        # 观测就是整个状态向量
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.x0.shape, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.x0.copy()
        self.t = 0
        return self.state, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # 线性动力学
        x_next = self.A @ self.state + self.B @ action
        # 奖励 = 负的二次损失（RL 里“奖励越大越好”）
        reward = - (self.state @ self.Q @ self.state + action @ self.R @ action)

        self.state = x_next.astype(np.float32)
        self.t += 1
        terminated = False
        truncated = self.t >= self.horizon  # 截断而不是失败
        return self.state, reward, terminated, truncated, {}

    def render(self):
        pass  # 需要可视化时在此绘图


if __name__ == "__main__":
    # ——— 环境合法性检查（可选）———
    env = LinearStateSpaceEnv()
    check_env(env, warn=True)

    # ——— 定义 & 训练 SAC ———
    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./tensorboard_logs",
        # 可调超参数
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64,
        tau=0.005,
    )

    model.learn(total_timesteps=50_000)

    # ——— 测试训练好的策略 ———
    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(env.horizon):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        if term or trunc:
            break
    print("Episode reward:", total_reward)
