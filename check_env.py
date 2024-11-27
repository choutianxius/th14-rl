"""
Checks the env for compatibility with SB3
"""

from stable_baselines3.common.env_checker import check_env
from environment import Touhou14Env

try:
    env = Touhou14Env()
    check_env(env, True, True)
    print("\033[92mEnv check passed!\033[0m")
except Exception:
    print("\033[91mEnv check failed!\033[0m")
    raise
finally:
    if "env" in locals():
        env.close()
