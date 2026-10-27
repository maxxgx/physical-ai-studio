# pi05 on bucket_sort

The scripts behind the October 2026 bucket_sort experiments, which took pi05 from 29% to 90% success
in the MuJoCo sim. They cover data generation with a scripted IK demonstrator, DAgger recovery data,
training with physicalai or LeRobot, closed-loop evaluation and an interactive language demo.

These are experiment scripts, not library code: they import `physicalai`, LeRobot and the MuJoCo
plugin directly and were run on a 4×H100 server.

## Results

Success on held-out layouts, cameras in their training slots, same 1,589 episodes unless noted.

| Model                                           | Success     |
| ----------------------------------------------- | ----------- |
| Studio pi05, 250 recorded episodes, bf16        | 29% (26/90) |
| physicalai, bf16 (default), 12k steps           | 53% (32/60) |
| bf16 fine-tune of the Studio model on this data | 68% (41/60) |
| LeRobot trainer, fp32, 12k steps                | 88% (79/90) |
| physicalai trainer, fp32, 12k steps             | 90% (54/60) |

Checkpoints are in the private Hugging Face repo `iMaxxgx/pi05-bucket-sort-runs`; ask Max for access.

## Setup

1. The physical-ai-studio library environment (`library/`, `uv sync`), plus `lerobot==0.6.0` and
   `accelerate`. The runs used torch 2.11, transformers 5.15.1, lightning 2.6.6 and mujoco 3.17.
2. The MuJoCo plugin with the bucket_sort scene and the IK demonstrator, from the
   `max/mujoco-bucket-sort-recording` branch of openvinotoolkit/physicalai (PR #404, stacked on #400):

   ```bash
   git clone -b max/mujoco-bucket-sort-recording https://github.com/openvinotoolkit/physicalai.git
   uv pip install -e physicalai/packages/physicalai-mujoco-plugin
   ```

3. Point the launchers at a big disk. Every launcher sources `env.sh`, which `cd`s into `$WORK` and
   keeps datasets, runs, logs and the HF cache there.

   ```bash
   export WORK=/mnt/data/$USER/bucket_sort_pi05   # the default
   export PY=/path/to/library/.venv/bin/python     # default: python
   ```

## Data

The datasets are LeRobot v3 datasets. On the h100s server they are in
`/mnt/data/xiangxi2/xiangxi2_vla/data/`:

| Dataset   | Episodes | Frames | Contents                                                                                    |
| --------- | -------- | ------ | ------------------------------------------------------------------------------------------- |
| `gen_all` | 1,199    | 431k   | IK demonstrator, commanded actions, randomized speed, grasp offset, yaw and gripper opening |
| `gen_dag` | 1,589    | 561k   | `gen_all` plus 390 DAgger recovery episodes                                                 |

Both list their cameras overview first. The Studio-recorded `bucket_sort` dataset lists them wrist first.

## Pipeline

Source `env.sh` first: it sets `$S` (this folder), `$PY` and `$WORK`, and `cd`s into `$WORK`, so the
relative paths below are under `$WORK`. Never render on a GPU that is training.

```bash
source library/examples/bucket_sort_pi05/env.sh

# 1. Record demonstrations: 8 workers × 150 kept episodes; 15% start from a pose the policy gets
#    stuck in. Use seed bases of 10,000,000 and up: evaluation uses seeds 100000-100089.
GPUS="0 1" bash $S/generate.sh gen_new 8 150 10000000 0.15
$PY $S/merge_gen.py data/gen_new_merged --keep data/gen_new

# 2. Train. fp32 is the recipe that reached 90%; bf16 is physicalai's default.
bash $S/train.sh fp32 data/gen_new_merged runs/my_fp32 0,1
bash $S/train.sh bf16 data/gen_new_merged runs/my_bf16 0
bash $S/train_lerobot.sh data/gen_new_merged runs/my_lerobot 0,1

# 3. Evaluate on the held-out layouts: set A is seeds 100000-100029, set B 100030-100059.
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 HF_HUB_OFFLINE=1 $PY $S/eval_bucket_sort.py \
  --policy pi05 --ckpt runs/my_fp32/stepstep=12000.ckpt --out runs/ev_my_fp32_A \
  --episodes 30 --seed0 100000 --videos 4
#    A LeRobot run: --lerobot --ckpt runs/my_lerobot/checkpoints/012000/pretrained_model

# 4. DAgger: the policy drives, the demonstrator takes over, only the takeover is recorded.
bash $S/dagger.sh dag1 runs/my_fp32/stepstep=6000.ckpt 4 80 31000000 0
$PY $S/merge_gen.py data/gen_dag_new --ds data/gen_new_merged --keep data/dag1

# 5. Language demo in the browser on port 8090: type an instruction and watch the policy.
bash $S/demo.sh start pi05 runs/my_fp32/stepstep=12000.ckpt 0
bash $S/demo.sh stop
```

`eval_bucket_sort.py` writes `summary.json` and one JSON line per episode, with an outcome of
`success`, `wrong_bucket`, `fail_no_grasp`, `fail_slipped` or `fail_dropped`. One set of 30 episodes
moves about ±9 points by chance, so compare at least two sets.

Other tools:

- `weight_change.py A.ckpt B.ckpt`: fraction of weights that changed between two checkpoints, per
  component and dtype. This is how the bf16 problem showed up.
- `pai_to_lerobot.py`: write a physicalai pi05 checkpoint as a LeRobot `pretrained_model` dir, to run
  the same weights through LeRobot's inference.

## Pitfalls we hit

- **Camera order.** physicalai's pi05 fills its image slots in the order the batch dict lists the
  cameras, not by name. A model trained on overview-first data and fed wrist-first images silently
  scores 28% instead of 90%. The eval, DAgger and demo scripts read the training order from the
  checkpoint and print `camera order fed to the policy`. Exported models get the fix in the runtime
  (openvinotoolkit/physicalai#411).
- **bf16 barely trains.** pi05 defaults to `dtype="bfloat16"` with no fp32 master copy, so most AdamW
  updates round to zero: about 2% of the language model's weights changed over 2,000 steps. Use
  `train.sh fp32`. fp32 needs about 71 GB per GPU at batch 8, so it runs on 2 GPUs, and `train.py`
  turns on DDP's `gradient_as_bucket_view` to fit.
- **LeRobot 0.6 with transformers 5** skips pi05_base's 437 vision weights and leaves the encoder at
  random init. `lerobot_train_ddp.py` patches the key names; check the log for "All keys loaded".
- **Disk.** `train.py` keeps `best.ckpt`, `last.ckpt` and `stepstep=N.ckpt` at every save: 9.4 GB
  each in bf16, 16.6 GB in fp32. Keep `$WORK` off a shared root disk.
- **Rendering GPU.** MuJoCo picks its render GPU from `MUJOCO_EGL_DEVICE_ID`, not
  `CUDA_VISIBLE_DEVICES`. On h100s, rendering on GPU 2 stalls about 19 s per frame when the same process
  also uses CUDA there; use GPUs 0, 1 or 3.
