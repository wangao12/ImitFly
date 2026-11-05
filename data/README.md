# External data and simulation assets

Demonstration data and external numerical assets are distributed separately.

```text
data/
├── flight/episode_0001.npz ...
├── walking/episode_0001.npz ...
└── assets/
    ├── wing_pattern_fmech.npy
    └── walking-reference.hdf5
```

See `docs/DATA_FORMAT.md` for the episode schema and legacy-log conversion command.
The wingbeat pattern, walking HDF5 references and expert SavedModels belong to the
FlyBody data/policy distribution. Obtain those resources separately and provide
their actual paths to the collection and evaluation commands.
