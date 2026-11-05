"""Zero-teacher student rollout; record actual simulated time and terminal states."""
import argparse
from collections import deque
import json
from pathlib import Path
import numpy as np
import torch
from imitfly.checkpoints import load_policy
from imitfly.contracts import SENSOR_DIMS
from imitfly.preprocessing import observation_tensors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--wpg",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=[0])
    p.add_argument("--max-steps",type=int,default=2000)
    p.add_argument("--arena-config",type=Path,default=Path(__file__).resolve().parents[1]/"configs/trench.json")
    p.add_argument("--terrain",choices=["trench","bumps"],default="trench")
    p.add_argument("--model",choices=["imitfly","no_scve","no_hmam","no_lstm","lstm","lstm_attention","feedforward"],default="imitfly")
    p.add_argument("--mask",choices=["none","vision","vestibular","proprioception"],default="none")
    p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    p.add_argument("--video",action="store_true")
    a=p.parse_args()
    if a.max_steps<1 or len(a.seeds)!=len(set(a.seeds)):
        raise ValueError("Positive step limit and distinct seeds required")
    if a.output.exists(): raise FileExistsError(a.output)
    device="cuda" if a.device=="auto" and torch.cuda.is_available() else a.device
    if device=="auto": device="cpu"
    torch.set_num_threads(2)
    model,c=load_policy(a.checkpoint,device,a.model)
    arena=json.loads(a.arena_config.read_text())
    from imitfly.env import make_environment
    import hashlib
    run_config={**vars(a),"device":device,"policy_config":c,"arena":arena,
                "checkpoint_sha256":hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                "wpg_sha256":hashlib.sha256(a.wpg.read_bytes()).hexdigest()}
    a.output.mkdir(parents=True)
    (a.output/"config.json").write_text(json.dumps(run_config,indent=2,default=str)+"\n")
    summary=[]
    for seed in a.seeds:
        env=make_environment(a.wpg,seed,a.terrain,arena)
        writer=None
        try:
            t=env.reset(); history=deque(maxlen=c["sequence_length"])
            times=[float(env.physics.data.time)]
            positions=[env.physics.data.qpos[:3].copy()]
            velocities=[env.physics.data.qvel[:3].copy()]
            requested=[]; applied=[]; rewards=[]
            if a.video:
                import cv2
                writer=cv2.VideoWriter(str(a.output/f"seed_{seed}.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),30,(640,480))
                if not writer.isOpened(): raise RuntimeError("Video writer failed")
            for step in range(a.max_steps):
                x=list(observation_tensors(t.observation))
                masked=[]
                if a.mask=="vision": masked=[0,1]
                if a.mask=="vestibular": masked=[2,3,6,7]
                if a.mask=="proprioception": masked=[4,5]
                for i in masked:x[i]=torch.zeros_like(x[i])
                history.append(x)
                if len(history)<c["sequence_length"]:
                    action=np.zeros(12,dtype=np.float32)
                else:
                    inputs=[torch.stack([row[i] for row in history]).unsqueeze(0).to(device)
                            for i in range(9)]
                    with torch.no_grad():
                        mu,_=model(inputs)
                    action=mu[0].cpu().numpy().copy()
                # Keep requested and clipped actions separate; the environment adds WPG internally.
                requested.append(action.copy()); applied.append(np.clip(action,-1,1).copy())
                t=env.step(action)
                rewards.append(float(t.reward or 0.))
                times.append(float(env.physics.data.time))
                positions.append(env.physics.data.qpos[:3].copy())
                velocities.append(env.physics.data.qvel[:3].copy())
                if writer is not None:
                    pixels=env.physics.render(height=480,width=640,camera_id=1)
                    writer.write(cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR))
                if t.last():break
            np.savez_compressed(a.output/f"seed_{seed}.npz",time=times,position=positions,
                velocity=velocities,requested_action=requested,canonical_action=applied,reward=rewards)
            at_limit=times[-1]>=0.4-1e-7
            row={"seed":seed,"steps":len(requested),"states":len(times),
                 "simulated_seconds":times[-1],"environment_last":bool(t.last()),
                 "termination":"time_limit" if at_limit else "environment_terminal" if t.last() else "step_limit",
                 "discount":float(t.discount) if t.discount is not None else None,
                 "return":sum(rewards),"teacher_calls":0,"mask":a.mask,
                 "target_height_cm":float(env.task.target_height),
                 "target_speed_cm_s":float(env.task.target_speed)}
            summary.append(row); print(json.dumps(row),flush=True)
        finally:
            if writer is not None:writer.release()
            env.close()
        (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")

if __name__ == "__main__":
    main()
