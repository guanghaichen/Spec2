import json
import numpy as np
from pathlib import Path


def extract_times(data):
    """
    返回所有step的推理耗时
    """

    all_times = []
    for episode in data:
        for step in episode:
            end_t, start_t = step
            all_times.append(end_t - start_t)

    return np.array(all_times)


def analyze(path):

    with open(path, "r") as f:
        data = json.load(f)

    times = extract_times(data)

    return {
        "steps": len(times),
        "mean": np.mean(times),
        "median": np.median(times),
        "std": np.std(times),
        "p95": np.percentile(times, 95),
        "p99": np.percentile(times, 99),
        "max": np.max(times),
        "min": np.min(times),
        "total": np.sum(times),
    }


files = {
    "AR": "/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/SpecVLA-main/openvla/specdecoding/test-speed/libero_goal_AR/EVAL-libero_goal-openvla-2026_06_04-13_41_26libero_goal_AR.json",
    "Spec": "/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/SpecVLA-main/openvla/specdecoding/test-speed/libero_goal_spec/EVAL-libero_goal-openvla-2026_06_04-18_47_12libero_goal_spec.json",
    "SpecRelaxed": "/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/SpecVLA-main/openvla/specdecoding/test-speed/libero_goal_spec_relaxed/EVAL-libero_goal-openvla-2026_06_04-23_41_24libero_goal_spec_relaxed.json",
}

results = {}

for name, path in files.items():

    stats = analyze(path)
    results[name] = stats

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    for k, v in stats.items():

        if k == "steps":
            print(f"{k:10s}: {v}")
        else:
            print(f"{k:10s}: {v:.6f} s")


print("\n")
print("=" * 60)
print("Speedup")
print("=" * 60)

ar_mean = results["AR"]["mean"]

for name in ["Spec", "SpecRelaxed"]:
    speedup = ar_mean / results[name]["mean"]
    print(f"{name:12s}: {speedup:.2f}x")