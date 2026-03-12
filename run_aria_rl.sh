#!/usr/bin/env bash
set -euo pipefail

# WARNING: plaintext secret in script (user requested). Prefer exporting in shell instead.
export WANDB_API_KEY="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"

# ====== USER CONFIG ======
DATASET_DIR="/home/ycl/projects/workspace/spider/example_datasets"
DATASET_NAME="gigahand"
ROBOT_TYPE="xhand"
EMBODIMENT_TYPE="right"

# Output id in SPIDER processed tree
TASK="mustard_901_1"
DATA_ID=0

# Optional preprocessing
RUN_IK=0
UP_SAMPLE="1"
START_STEP="0"

# RL (single-task, single-demo residual policy; rsl_rl)
RL_DEVICE="cuda:0"
RL_NUM_ENVS="16384"
RL_FUTURE_CTRL_STEPS="0"   # minimal current-step observation (future refs disabled)
RL_EPISODE_CTRL_STEPS="64"
RL_NUM_STEPS_PER_ENV="16"
RL_LEARNING_ITERS="3000"
RL_CKPT_INTERVAL="500"
RL_VIZ_INTERVAL="20"
RL_ROLLOUT_VIDEO_CTRL_STEPS="0"   # 0 => no cap; rollout video runs until terminate/ref_end

# W&B (media-only upload: reward curve png + rollout mp4 only)
WANDB_MEDIA_ONLY="1"
WANDB_PROJECT="egoengine-spider-rl"
WANDB_ENTITY=""
WANDB_RUN_NAME=""

# Reuse MPC timing/reward scales (CTRL_DT must match reference trajectory rate)
SIM_DT="0.01"
CTRL_DT="0.05"   # 20 Hz control, aligned with reference trajectory
HORIZON="1.6"
KNOT_DT="0.4"

OBJ_POS_REW_SCALE="1.0"
OBJ_ROT_REW_SCALE="0.3"
JOINT_REW_SCALE="0" # "0.003"
BASE_POS_REW_SCALE="0" # "1.0"
BASE_ROT_REW_SCALE="0" # "0.3"
VEL_REW_SCALE="0.000"
CONTACT_REW_SCALE="0.0"
OBJECT_POS_THRESHOLD="0.05"
OBJECT_ROT_THRESHOLD="1.0"

# Domain randomization (same parameterization style as MJWP MPC)
NUM_DR="1"
PAIR_MARGIN_RANGE_MIN="-0.005"
PAIR_MARGIN_RANGE_MAX="0.005"
XY_OFFSET_RANGE_MIN="-0.005"
XY_OFFSET_RANGE_MAX="0.005"
OBJECT_MASS_SCALE_RANGE_MIN="0.95"
OBJECT_MASS_SCALE_RANGE_MAX="1.05"

# Residual action scale baseline from MPC noise scales
FIRST_CTRL_NOISE="0.5"
LAST_CTRL_NOISE="1.0"
FINAL_NOISE="0.1"
JOINT_NOISE_SCALE="0.2" # "0.15"
POS_NOISE_SCALE="0.0325" # "0.03"
ROT_NOISE_SCALE="0.05" # "0.03"

# first run the IK
if [[ "${RUN_IK}" == "1" ]]; then
  uv run spider/preprocess/ik.py \
    --dataset-dir "${DATASET_DIR}" \
    --dataset-name "${DATASET_NAME}" \
    --robot-type "${ROBOT_TYPE}" \
    --embodiment-type "${EMBODIMENT_TYPE}" \
    --task "${TASK}" \
    --data-id "${DATA_ID}" \
    --no-show-viewer
fi

# Optional START_STEP + UP_SAMPLE to prepare the reference trajectory file used by RL.
if [[ "${UP_SAMPLE}" != "1" || "${START_STEP}" != "0" ]]; then
  ROBOT_KIN_PATH="${DATASET_DIR}/processed/${DATASET_NAME}/${ROBOT_TYPE}/${EMBODIMENT_TYPE}/${TASK}/${DATA_ID}/trajectory_kinematic.npz"
  uv run python - "${ROBOT_KIN_PATH}" "${UP_SAMPLE}" "${START_STEP}" <<'PY'
import sys
import numpy as np

kin_path = sys.argv[1]
up = int(sys.argv[2])
start_step = int(sys.argv[3])
with np.load(kin_path) as d:
    src = {k: d[k].copy() for k in d.files}
keys = list(src.keys())
if not keys:
    raise SystemExit(f"Empty npz: {kin_path}")
T0 = src[keys[0]].shape[0]
if start_step >= T0:
    raise SystemExit(f"START_STEP={start_step} out of range for length={T0}")

def upsample_time(arr, factor):
    n = arr.shape[0]
    if n <= 1:
        return arr.copy()
    n_up = (n - 1) * factor + 1
    t_old = np.arange(n, dtype=np.float64)
    t_new = np.linspace(0.0, n - 1, n_up, dtype=np.float64)
    flat = arr.reshape(n, -1)
    out = np.empty((n_up, flat.shape[1]), dtype=flat.dtype)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(t_new, t_old, flat[:, j])
    return out.reshape((n_up,) + arr.shape[1:])

out = {}
for k, v in src.items():
    if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] > 0:
        v2 = v[start_step:]
        out[k] = upsample_time(v2, up) if (up > 1 and v2.shape[0] > 1) else v2
    else:
        out[k] = v
np.savez_compressed(kin_path, **out)
print(f"[OK] prepared trajectory_kinematic for RL: {kin_path}")
PY
fi

export PYTHONFAULTHANDLER=1
export SPIDER_DISABLE_CUDA_GRAPH=1

# RL trainer uses the current shell Python (e.g. conda env), because rsl_rl is
# commonly installed there instead of the project's uv-managed .venv.
RL_PYTHON_BIN="${RL_PYTHON_BIN:-python}"
"${RL_PYTHON_BIN}" -c "import rsl_rl" >/dev/null 2>&1 || {
  echo "[ERROR] rsl_rl not found in ${RL_PYTHON_BIN}. Try: pip install rsl-rl-lib"
  exit 1
}

WANDB_MEDIA_FLAG=()
if [[ "${WANDB_MEDIA_ONLY}" == "1" ]]; then
  WANDB_MEDIA_FLAG+=(--wandb-media-only)
fi

# now running the rl.
"${RL_PYTHON_BIN}" examples/run_mjwp_rl_rsl.py \
  --dataset-dir "${DATASET_DIR}" \
  --dataset-name "${DATASET_NAME}" \
  --robot-type "${ROBOT_TYPE}" \
  --embodiment-type "${EMBODIMENT_TYPE}" \
  --task "${TASK}" \
  --data-id "${DATA_ID}" \
  --device "${RL_DEVICE}" \
  --num-envs "${RL_NUM_ENVS}" \
  --future-ctrl-steps "${RL_FUTURE_CTRL_STEPS}" \
  --max-episode-ctrl-steps "${RL_EPISODE_CTRL_STEPS}" \
  --num-steps-per-env "${RL_NUM_STEPS_PER_ENV}" \
  --learning-iterations "${RL_LEARNING_ITERS}" \
  --ckpt-interval "${RL_CKPT_INTERVAL}" \
  --viz-interval "${RL_VIZ_INTERVAL}" \
  --rollout-video-ctrl-steps "${RL_ROLLOUT_VIDEO_CTRL_STEPS}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  "${WANDB_MEDIA_FLAG[@]}" \
  --sim-dt "${SIM_DT}" \
  --ctrl-dt "${CTRL_DT}" \
  --horizon "${HORIZON}" \
  --knot-dt "${KNOT_DT}" \
  --pos-rew-scale "${OBJ_POS_REW_SCALE}" \
  --rot-rew-scale "${OBJ_ROT_REW_SCALE}" \
  --joint-rew-scale "${JOINT_REW_SCALE}" \
  --base-pos-rew-scale "${BASE_POS_REW_SCALE}" \
  --base-rot-rew-scale "${BASE_ROT_REW_SCALE}" \
  --vel-rew-scale "${VEL_REW_SCALE}" \
  --contact-rew-scale "${CONTACT_REW_SCALE}" \
  --object-pos-threshold "${OBJECT_POS_THRESHOLD}" \
  --object-rot-threshold "${OBJECT_ROT_THRESHOLD}" \
  --num-dr "${NUM_DR}" \
  --pair-margin-range-min "${PAIR_MARGIN_RANGE_MIN}" \
  --pair-margin-range-max "${PAIR_MARGIN_RANGE_MAX}" \
  --xy-offset-range-min "${XY_OFFSET_RANGE_MIN}" \
  --xy-offset-range-max "${XY_OFFSET_RANGE_MAX}" \
  --object-mass-scale-range-min "${OBJECT_MASS_SCALE_RANGE_MIN}" \
  --object-mass-scale-range-max "${OBJECT_MASS_SCALE_RANGE_MAX}" \
  --first-ctrl-noise-scale "${FIRST_CTRL_NOISE}" \
  --last-ctrl-noise-scale "${LAST_CTRL_NOISE}" \
  --final-noise-scale "${FINAL_NOISE}" \
  --joint-noise-scale "${JOINT_NOISE_SCALE}" \
  --pos-noise-scale "${POS_NOISE_SCALE}" \
  --rot-noise-scale "${ROT_NOISE_SCALE}"
