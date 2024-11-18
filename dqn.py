from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN
from environment import Touhou14Env


try:
    env = Touhou14Env()
    check_env(env, True, True)

    model = DQN(
        "CnnPolicy",
        env,
        buffer_size=10000,
        target_update_interval=100,
        verbose=1,
        device="cuda",
    )
    # about 10s episodes
    model.learn(total_timesteps=1000, log_interval=4)
    model.save("./save/dqn_cuda")
finally:
    env.close()
