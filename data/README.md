# Downloaded datasets

Download the shared `data` folder from Google Drive and copy its **contents** into this directory. Keep every dataset directory name unchanged.

Expected layout:

```text
data/
├── all6_100_20260920_223136_raw/                         # 600 scripted raw demonstrations
├── all6_100_20260920_223136_lerobot/                     # 600 scripted LeRobot demonstrations
├── cylinder_rear_50_20260921_0922_raw/                   # supplemental raw demonstrations
├── cylinder_rear_50_20260921_0922_lerobot/               # supplemental 100-episode LeRobot set
├── pi05_policy_rollouts_300_raw/                          # complete 300-policy-rollout reports and videos
├── pi05_ik_corrections_35_fixed_20260923_204117_raw/     # 26 successful IK correction trajectories
├── pi05_policy_failures_73_lerobot/                       # 73 failed policy rollouts
├── pi05_positive_227plus26_lerobot/                       # 227 policy successes + 26 IK suffixes
├── pi05_value_600success_73failure_lerobot/               # merged value-training dataset
├── piperx_advantage_rollout_60_adv3000_seed30000_raw/    # 60 post-bootstrap rollouts
└── piperx_advantage_rollout_60_adv3000_seed30000_lerobot/ # the same 60 rollouts in LeRobot format
```

The workflows do not merge files in place. A LeRobot dataset must remain a complete directory containing its own `meta/`, `data/`, and `videos/` trees.

Old raw-rollout manifests record the original machine's calibration path. The numbered workflows pass `CALIBRATION` explicitly so replay and IK correction use the downloaded `cameras.json` instead. Preserve the raw rollout's `standard/reset_states/` directory and per-trial NPZ files when downloading.
