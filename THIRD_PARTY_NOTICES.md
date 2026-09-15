# Third-Party Notices / 第三方资产声明

## MaleCNS v1.0 connectome data

- Source: Janelia Research Campus, FlyEM team — MaleCNS connectome
  (bulk download: http://male-cns.janelia.org/download/)
- License: **Creative Commons Attribution 4.0 International (CC BY 4.0)**
- Use in this project: `flygo_data.npz` is a **derived** simulation bundle built from the
  MaleCNS v1.0 feather tables (body annotations, predicted neurotransmitters, connectome
  weights, synapse points) by `prepare_data.py`. It contains transformed connectivity,
  polarity and coordinate data for 166,700 neurons / 25,582,938 retained connections.
- Attribution: "Connectome data derived from the MaleCNS v1.0 dataset (FlyEM,
  HHMI Janelia; collaborators incl. Univ. Cambridge, MRC LMB, Google Research),
  used under CC BY 4.0." — modified/derived as described above.

## three.js (bundled `web/three.min.js`, r128)

Copyright © 2010-2021 Three.js authors — MIT License (https://github.com/mrdoob/three.js).

## MuJoCo

Copyright 2021 DeepMind Technologies Limited — Apache License 2.0 (https://mujoco.org).

## flybody (fruit-fly body mesh + MJCF rig)

Derived assets assembled from the flybody project — Apache License 2.0
(DeepMind). `build_fly_model.py` assembles `fly_model.bin` from the flybody
MJCF/mesh hierarchy.
