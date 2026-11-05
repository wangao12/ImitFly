"""Roll out the walking policy on a reference trajectory."""
import argparse
from collections import deque
import json
from pathlib import Path
import numpy as np
import torch
from imitfly.contracts import WALKING_SENSOR_DIMS
from imitfly.models.walking import WalkingPolicy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trajectory-index", type=int, default=316)
    p.add_argument("--sequence-length", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--video", action="store_true")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    device = "cuda" if a.device == "auto" and torch.cuda.is_available() else a.device
    if device == "auto":
        device = "cpu"
    torch.set_num_threads(2)
    checkpoint = torch.load(a.checkpoint, map_location=device, weights_only=True)
    model = WalkingPolicy().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    sequence_length = checkpoint.get("config", {}).get("sequence_length", a.sequence_length)
    from imitfly.env import make_walking_environment
    env = make_walking_environment(a.reference, a.seed, a.trajectory_index)
    a.output.mkdir(parents=True)
    (a.output / "config.json").write_text(json.dumps(vars(a), default=str, indent=2) + "\n")
    history = deque(maxlen=sequence_length)
    video = None
    try:
        timestep = env.reset()
        times = [float(env.physics.data.time)]
        positions = [env.physics.data.qpos[:3].copy()]
        actions = []
        if a.video:
            import cv2
            video = cv2.VideoWriter(str(a.output / "walking.mp4"),
                                    cv2.VideoWriter_fourcc(*"mp4v"), 30, (640,480))
            if not video.isOpened():
                raise RuntimeError("Video writer failed")
        for step in range(a.max_steps):
            sensors = {k:np.asarray(timestep.observation[f"walker/{k}"],np.float32).reshape(-1)
                       for k in WALKING_SENSOR_DIMS}
            history.append(sensors)
            while len(history) < sequence_length:
                history.appendleft({k:np.zeros_like(v) for k,v in sensors.items()})
            inputs = {k:torch.as_tensor(np.stack([row[k] for row in history]),
                                        device=device).unsqueeze(0)
                      for k in WALKING_SENSOR_DIMS}
            with torch.no_grad():
                mean, _ = model(inputs)
            action = np.clip(mean[0].cpu().numpy(), -1, 1)
            actions.append(action.copy())
            timestep = env.step(action)
            times.append(float(env.physics.data.time))
            positions.append(env.physics.data.qpos[:3].copy())
            if video is not None:
                frame = env.physics.render(camera_id=2, width=640, height=480)
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if timestep.last():
                break
        np.savez_compressed(a.output / "trajectory.npz", time=times, position=positions,
                            canonical_action=actions)
        summary = {"steps":len(actions), "states":len(times), "teacher_calls":0,
                   "simulated_seconds":times[-1], "environment_last":bool(timestep.last())}
        (a.output / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        print(json.dumps(summary))
    finally:
        if video is not None:
            video.release()
        env.close()

if __name__ == "__main__":
    main()
