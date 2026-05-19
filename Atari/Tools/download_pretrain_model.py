
import subprocess
subprocess.run(["python", "-m", "rl_zoo3.load_from_hub", "--algo", "ppo", "--env", "BreakoutNoFrameskip-v4", "-orga", "sb3", "-f", "../pretrain_model/"], check=True)  