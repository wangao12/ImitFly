"""Collect aligned expert Gaussian distributions from a FlyBody SavedModel."""
import argparse
import json
from pathlib import Path
import numpy as np
from imitfly.contracts import SENSOR_DIMS, WALKING_SENSOR_DIMS
from imitfly.schema import validate_episode, validate_walking_episode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expert",type=Path,required=True)
    p.add_argument("--wpg",type=Path)
    p.add_argument("--task",choices=["flight","walking"],default="flight")
    p.add_argument("--reference",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--episodes",type=int,default=12)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--arena-config",type=Path,default=Path(__file__).resolve().parents[1]/"configs/trench.json")
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    if a.episodes<1: raise ValueError("episodes must be positive")
    import tensorflow as tf
    import tensorflow_probability  # Registers SavedModel distribution types.
    from imitfly.env import make_environment, make_walking_environment
    from PIL import Image
    tf.random.set_seed(a.seed)
    expert=tf.saved_model.load(str(a.expert))
    arena=json.loads(a.arena_config.read_text())
    if a.task=="flight" and a.wpg is None:
        p.error("flight collection requires --wpg")
    if a.task=="walking" and a.reference is None:
        p.error("walking collection requires --reference")
    sensor_dims = SENSOR_DIMS if a.task=="flight" else WALKING_SENSOR_DIMS
    eye_names = ["left_eye", "right_eye"] if a.task=="flight" else []
    a.output.mkdir(parents=True)
    (a.output/"collection.json").write_text(json.dumps(vars(a),indent=2,default=str)+"\n")
    for episode in range(a.episodes):
        env=(make_environment(a.wpg,a.seed+episode,arena=arena) if a.task=="flight"
             else make_walking_environment(a.reference,a.seed+episode))
        try:
            t=env.reset()
            arrays={k:[] for k in [*eye_names,*sensor_dims,"expert_mean","expert_std"]}
            while not t.last():
                obs=t.observation
                batch={k:tf.convert_to_tensor(v[None]) for k,v in obs.items()}
                distribution=expert(batch)
                mean=distribution.mean()[0].numpy(); std=distribution.stddev()[0].numpy()
                action=distribution.sample()[0].numpy()
                for name in sensor_dims:
                    arrays[name].append(np.asarray(obs[f"walker/{name}"],np.float32).reshape(-1))
                # Legacy expert-data exporter takes the RGB channel mean before PNG storage.
                for name in eye_names:
                    eye=np.asarray(obs[f"walker/{name}"])
                    if np.issubdtype(eye.dtype,np.floating):eye=eye*255
                    if eye.ndim==3:eye=eye.mean(axis=-1)
                    image=Image.fromarray(eye.astype(np.uint8)).resize((32,32),Image.Resampling.BILINEAR)
                    arrays[name].append(np.asarray(image))
                arrays["expert_mean"].append(mean); arrays["expert_std"].append(std)
                t=env.step(action)
            arrays={k:np.stack(v) for k,v in arrays.items()}
            n=(validate_episode(arrays) if a.task=="flight" else validate_walking_episode(arrays))
            with (a.output/f"episode_{episode+1:04d}.npz").open("xb") as f:
                np.savez_compressed(f,**arrays)
            print(f"episode={episode+1} frames={n}",flush=True)
        finally:
            env.close()

if __name__ == "__main__":
    main()
