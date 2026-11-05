# Experiments

## Flight and structural ablations

All variants share the same data format and training command:

```bash
python -m imitfly.cli.train \
  --config configs/no_scve.json \
  --data data/flight \
  --output outputs/no_scve
```

| Configuration | Model or objective |
|---|---|
| `imitfly_12_demos.json` | Full ImitFly, twelve demonstration episodes |
| `imitfly_v51.json` | Original v51 script settings: eighteen episodes, 260 epochs |
| `no_scve.json` | Replace the LIF layer with ReLU |
| `no_hmam.json` | Disable attention; average sensor features and concatenate vision |
| `no_lstm.json` | Replace recurrent encoders with last-frame linear projections |
| `no_kl.json` | Full architecture with expected Gaussian negative log likelihood |
| `lstm.json` | Historical CNN + LSTM comparator |
| `lstm_attention.json` | Historical CNN + LSTM + attention comparator |
| `feedforward.json` | Historical CNN + feedforward comparator |

The structural switches provide a common interface for training the corresponding
ablation models. Use a separately trained checkpoint for each variant. The historical
comparators are retained as comparison architectures; they differ in more than one
component from the full v51 network.

The full model uses `KL(student || expert)` averaged over batch and action dimensions.
For the NLL configuration, the objective is the expected negative log likelihood of the
student Gaussian under the expert Gaussian. Configuration presets organize experiments;
they do not embed published numerical results.

Use matching data, random seeds, and horizons when comparing configurations.
The trainer records the selected architecture and objective in its checkpoints.

## Sensory modalities

```bash
python -m imitfly.cli.evaluate \
  --checkpoint checkpoints/imitfly_v51.pth \
  --wpg data/assets/wing_pattern_fmech.npy \
  --output outputs/no_vision \
  --seeds 0 1 2 \
  --mask vision
```

- `vision`: zero both eye-image tensors.
- `vestibular`: zero accelerometer, gyro, velocimeter and world-z-axis inputs.
- `proprioception`: zero joint position and joint velocity inputs.

These switches perform inference-time input masking. If an experiment instead requires
retraining a sensory-ablated policy, train with the corresponding inputs removed or zeroed
in its data preparation and use that model's checkpoint.

## Walking adaptation

Collect walking demonstrations in the separate teacher environment:

```bash
python -m imitfly.cli.collect \
  --task walking \
  --expert /path/to/trained-fly-policies/walking \
  --reference /path/to/walking-reference.hdf5 \
  --output data/walking \
  --episodes 12
```

Train the walking policy in the student environment:

```bash
python -m imitfly.cli.train \
  --config configs/walking.json \
  --data data/walking \
  --output outputs/walking
```

Run a chosen reference trajectory:

```bash
python -m imitfly.cli.evaluate_walking \
  --checkpoint checkpoints/imitfly_walking.pth \
  --reference /path/to/walking-reference.hdf5 \
  --trajectory-index 316 \
  --output outputs/walking_demo \
  --video
```

The walking policy uses twelve sensor/reference streams, LSTM encoders with 256 hidden
units, attention aggregation, and 59-dimensional Gaussian heads. Its recurrent history
uses five frames, with zero padding at episode start.

## Terrain and output

`configs/trench.json` specifies the sine-trench arena. To use a bumps arena:

```bash
python -m imitfly.cli.evaluate \
  --checkpoint checkpoints/imitfly_v51.pth \
  --wpg data/assets/wing_pattern_fmech.npy \
  --terrain bumps \
  --arena-config configs/bumps.json \
  --output outputs/bumps
```

Flight rollout files contain `N+1` states and `N` actions, including the terminal state.
The summary distinguishes the environment time limit, earlier environment termination,
and a user-specified step limit. It reports the actual elapsed simulation time.
Rendering at 30 fps is a visualization convention rather than the simulated control rate.

## Implementation notes

The main policy retains the v51 model structure: one shared 32-channel convolution/LIF/
max-pooling encoder, a 16,384-dimensional concatenated binocular feature input to the
visual LSTM, seven 64-unit sensor LSTMs, and four groups of 32 fused features.
The SCVE implementation returns membrane features; its layer state resets at each encoder
call, while temporal context is supplied by the LSTM window.

The paper's algorithms describe the architecture at a higher level. This release organizes
the existing v51 implementation into named modules rather than substituting another
architecture for the separately supplied v51 weights.
The monocular drone/BiLSTM/SE(3) controller belongs to a separate project and is not included.
