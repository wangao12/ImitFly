# Data format

## Flight demonstrations

Each episode is an independent `.npz` file, loaded with `allow_pickle=False`.

| Array | Shape | Meaning |
|---|---|---|
| `left_eye` | `[N,32,32]` | Left grayscale image, uint8 |
| `right_eye` | `[N,32,32]` | Right grayscale image, uint8 |
| `accelerometer` | `[N,3]` | Accelerometer |
| `gyro` | `[N,3]` | Angular velocity |
| `joints_pos` | `[N,25]` | Joint positions |
| `joints_vel` | `[N,25]` | Joint velocities |
| `velocimeter` | `[N,3]` | Egocentric velocity |
| `world_zaxis` | `[N,3]` | Gravity-reference direction |
| `task_input` | `[N,2]` | Target height, then target speed |
| `expert_mean` | `[N,12]` | Expert Gaussian mean |
| `expert_std` | `[N,12]` | Expert Gaussian standard deviation |

Sensor arrays use float32. Images are transformed to grayscale float tensors in `[0,1]`.
All arrays in an episode refer to the same sequence of observation steps.
A five-frame window predicts the expert distribution at its final frame.
Windows are formed within each episode. Split validation data by episode file.

The model's positional sensor order is accelerometer, gyro, joints_pos, joints_vel,
velocimeter, world_zaxis, task_input. Do not reorder it when supplying tensors directly.
FlyBody uses centimeters, grams and seconds; speed values are in cm/s.

### Legacy text logs and PNG images

```text
dataset_1/
├── inputs_log.txt
├── outputs_log.txt
├── left_eye/left_eye_0.png ...
└── right_eye/right_eye_0.png ...
```

Convert each dataset directory into one episode:

```bash
python -m imitfly.cli.convert_data \
  --source /path/to/data_meanstd/dataset_1 \
  --output data/flight/episode_0001.npz
```

Records and images are joined by numeric frame ID. Missing or mismatched records
produce an error rather than shifting the observation/target alignment.

## Walking demonstrations

Walking NPZ files contain `expert_mean` and `expert_std` with shape `[N,59]`, plus:

| Array | Per-frame dimension |
|---|---:|
| accelerometer, gyro, velocimeter, world_zaxis | 3 each |
| joints_pos, joints_vel | 85 each |
| actuator_activation | 59 |
| appendages_pos | 21 |
| force | 18 |
| touch | 6 |
| ref_displacement | 195, flattened from 65 x 3 |
| ref_root_quat | 260, flattened from 65 x 4 |

Use the collection entry point with `--task walking` to generate this format.
The reference-trajectory channels are inputs to the walking policy.

## Flight action convention

All policies output canonical actions. The environment wrapper clips them to `[-1,1]`
and maps them to the task's action limits. FlyBody then applies its wingbeat controller.

| Indices | Control |
|---|---|
| 0-2 | Head |
| 3-8 | Left/right wing deviations |
| 9-10 | Abdomen |
| 11 | Wingbeat frequency modulation |

The first four flight control steps use zero canonical actions while the five-frame
history is filled. The LSTM is evaluated over the current window at each step.
