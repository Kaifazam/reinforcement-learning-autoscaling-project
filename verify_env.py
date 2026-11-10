from autoscaling_env import MultiTierAutoscalingEnv, load_workload

workload = load_workload()
env = MultiTierAutoscalingEnv(workload)

obs, info = env.reset()
steps = 0
done = False

while not done:
    obs, reward, terminated, truncated, metrics = env.step(0)
    steps += 1
    done = terminated or truncated

print("NASA records:", len(workload))
print("Evaluation steps:", steps)
print("Expected steps:", env.max_steps)
print("Verification:", "PASSED" if steps == env.max_steps else "FAILED")
