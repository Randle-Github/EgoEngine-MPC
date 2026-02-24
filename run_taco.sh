#!/usr/bin/env bash
set -euo pipefail

# ====== USER CONFIG ======
DATASET_DIR="/home/ycl/projects/workspace/spider/example_datasets"
DATASET_NAME="gigahand"
ROBOT_TYPE="xhand"
EMBODIMENT_TYPE="bimanual"   # should stay bimanual for TACO two-hand demos

# TACO sample (bimanual, two objects)
TACO_DIR="/home/ycl/projects/datasets/egoengine/TACO/Hand_Poses/(brush, eraser, bowl)/20231027_037"
RETARGET_JSON="${TACO_DIR}/retarget.json"
ENV_XML="${TACO_DIR}/env.xml"
RIGHT_HAND_PKL="${TACO_DIR}/right_hand.pkl"
LEFT_HAND_PKL="${TACO_DIR}/left_hand.pkl"
RIGHT_HAND_SHAPE_PKL="${TACO_DIR}/right_hand_shape.pkl"
LEFT_HAND_SHAPE_PKL="${TACO_DIR}/left_hand_shape.pkl"

# Output id in SPIDER processed tree
TASK="brush_eraser_bowl_20231027_037"
DATA_ID=0
RIGHT_OBJECT_NAME="taco_tool_176"
LEFT_OBJECT_NAME="taco_target_022"
BASE_OFFSET_XYZ="0.0 0.0 0.0"
ARENA_DIFF_XYZ="0.0 0.0 0.0"
UP_SAMPLE="1"   # trajectory temporal upsample factor (integer >= 1)
START_STEP="0"  # only for MPC input trajectory, start from timestep K

# Optional flags
RUN_CONTACT_DETECT=1   # 1 to enable
SHOW_VIEWER=1

# MPC tuning: default.yaml-like setup
MPC_SIM_DT="0.01"
MPC_CTRL_DT="0.4"
MPC_HORIZON="1.6"
MPC_KNOT_DT="0.4"
MPC_NUM_SAMPLES="1024"
MPC_MAX_ITERS="32"
MPC_IMPROVE_THRESH="0.005"
MPC_TEMPERATURE="0.1"

# Reward tuning: object tracking dominant
OBJ_POS_REW_SCALE="5.0"
OBJ_ROT_REW_SCALE="2.0"
JOINT_REW_SCALE="0.0003"
BASE_POS_REW_SCALE="0.0"
BASE_ROT_REW_SCALE="0.0"
VEL_REW_SCALE="0.0001"
CONTACT_REW_SCALE="2.0"

# ====== 1) Bridge TACO retarget -> SPIDER processed/mano ======
uv run spider/process_datasets/taco_retarget_to_spider.py \
  --dataset-dir "${DATASET_DIR}" \
  --dataset-name "${DATASET_NAME}" \
  --task "${TASK}" \
  --data-id "${DATA_ID}" \
  --embodiment-type "${EMBODIMENT_TYPE}" \
  --retarget-json "${RETARGET_JSON}" \
  --env-xml "${ENV_XML}" \
  --right-hand-pkl "${RIGHT_HAND_PKL}" \
  --left-hand-pkl "${LEFT_HAND_PKL}" \
  --right-hand-shape-pkl "${RIGHT_HAND_SHAPE_PKL}" \
  --left-hand-shape-pkl "${LEFT_HAND_SHAPE_PKL}" \
  --hand-loader-root "/home/ycl/projects/workspace/egoengine" \
  --right-object-name "${RIGHT_OBJECT_NAME}" \
  --left-object-name "${LEFT_OBJECT_NAME}" \
  --ref-dt 0.02 \
  --base-offset-xyz ${BASE_OFFSET_XYZ} \
  --arena-diff-xyz ${ARENA_DIFF_XYZ}

# ====== 2) Object convex decomposition ======
uv run spider/preprocess/decompose_fast.py \
  --dataset-dir "${DATASET_DIR}" \
  --dataset-name "${DATASET_NAME}" \
  --robot-type "${ROBOT_TYPE}" \
  --embodiment-type "${EMBODIMENT_TYPE}" \
  --task "${TASK}" \
  --data-id "${DATA_ID}"

# ====== 3) Contact detection ======
if [[ "${RUN_CONTACT_DETECT}" == "1" ]]; then
  MANO_KIN_PATH="${DATASET_DIR}/processed/${DATASET_NAME}/mano/${EMBODIMENT_TYPE}/${TASK}/${DATA_ID}/trajectory_kinematic.npz"
  MANO_KEY_PATH="${DATASET_DIR}/processed/${DATASET_NAME}/mano/${EMBODIMENT_TYPE}/${TASK}/${DATA_ID}/trajectory_keypoints.npz"
  if [[ ! -f "${MANO_KIN_PATH}" && -f "${MANO_KEY_PATH}" ]]; then
    cp -f "${MANO_KEY_PATH}" "${MANO_KIN_PATH}"
    echo "[INFO] created MANO trajectory_kinematic from trajectory_keypoints for contact detection"
  fi
  if [[ ! -f "${MANO_KIN_PATH}" ]]; then
    echo "[ERROR] contact mode enabled, but MANO trajectory_kinematic is missing: ${MANO_KIN_PATH}"
    exit 1
  fi
  uv run spider/preprocess/detect_contact.py \
    --dataset-dir "${DATASET_DIR}" \
    --dataset-name "${DATASET_NAME}" \
    --embodiment-type "${EMBODIMENT_TYPE}" \
    --task "${TASK}" \
    --data-id "${DATA_ID}" \
    --show-viewer

  # hard check: contact labels must exist after detect_contact
  uv run python - "${MANO_KEY_PATH}" <<'PY'
import numpy as np, sys
path = sys.argv[1]
data = np.load(path)
required = ("contact_left", "contact_right", "contact_pos_left", "contact_pos_right")
missing = [k for k in required if k not in data]
if missing:
    print(f"[ERROR] contact fields missing in {path}: {missing}")
    sys.exit(1)
print("[OK] contact fields found in trajectory_keypoints.npz")
PY
fi

# ====== 4) Generate scene.xml for xhand ======
uv run spider/preprocess/generate_xml.py \
  --dataset-dir "${DATASET_DIR}" \
  --dataset-name "${DATASET_NAME}" \
  --robot-type "${ROBOT_TYPE}" \
  --embodiment-type "${EMBODIMENT_TYPE}" \
  --task "${TASK}" \
  --data-id "${DATA_ID}" \
  --no-show-viewer

# ====== 5) Kinematic IK retargeting ======
uv run spider/preprocess/ik.py \
  --dataset-dir "${DATASET_DIR}" \
  --dataset-name "${DATASET_NAME}" \
  --robot-type "${ROBOT_TYPE}" \
  --embodiment-type "${EMBODIMENT_TYPE}" \
  --task "${TASK}" \
  --data-id "${DATA_ID}" \
  --open-hand \
  --no-show-viewer

# ====== 5.5) Optional START_STEP + UP_SAMPLE for MPC input trajectory only ======
if [[ "${UP_SAMPLE}" != "1" || "${START_STEP}" != "0" ]]; then
  ROBOT_KIN_PATH="${DATASET_DIR}/processed/${DATASET_NAME}/${ROBOT_TYPE}/${EMBODIMENT_TYPE}/${TASK}/${DATA_ID}/trajectory_kinematic.npz"
  uv run python - "${ROBOT_KIN_PATH}" "${UP_SAMPLE}" "${START_STEP}" <<'PY'
import sys
import numpy as np

kin_path = sys.argv[1]
up = int(sys.argv[2])
start_step = int(sys.argv[3])
if up < 1:
    raise SystemExit("UP_SAMPLE must be >= 1")
if start_step < 0:
    raise SystemExit("START_STEP must be >= 0")

with np.load(kin_path) as d:
    keys = list(d.files)
    src = {k: d[k].copy() for k in keys}

if not keys:
    raise SystemExit(f"Empty npz: {kin_path}")

T0 = src[keys[0]].shape[0]
if start_step >= T0:
    raise SystemExit(f"START_STEP={start_step} out of range for length={T0}")

def upsample_time(arr: np.ndarray, factor: int) -> np.ndarray:
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
        if up > 1 and v2.shape[0] > 1:
            out[k] = upsample_time(v2, up)
        else:
            out[k] = v2
    else:
        out[k] = v

np.savez_compressed(kin_path, **out)
print(f"[OK] prepared trajectory_kinematic for MPC: {kin_path}")
print(f"[INFO] START_STEP={start_step}, UP_SAMPLE={up}")
print(f"[INFO] frames: {src[keys[0]].shape[0]} -> {out[keys[0]].shape[0]}")
PY
fi

# ====== 6) Dynamics optimization (MJWP) ======
export PYTHONFAULTHANDLER=1
export SPIDER_DISABLE_CUDA_GRAPH=1

uv run examples/run_mjwp.py \
  +override=${DATASET_NAME} \
  task=${TASK} \
  data_id=${DATA_ID} \
  robot_type=${ROBOT_TYPE} \
  embodiment_type=${EMBODIMENT_TYPE} \
  show_viewer=${SHOW_VIEWER} \
  viewer=mujoco-rerun \
  sim_dt=${MPC_SIM_DT} \
  ctrl_dt=${MPC_CTRL_DT} \
  horizon=${MPC_HORIZON} \
  knot_dt=${MPC_KNOT_DT} \
  num_samples=${MPC_NUM_SAMPLES} \
  max_num_iterations=${MPC_MAX_ITERS} \
  improvement_threshold=${MPC_IMPROVE_THRESH} \
  temperature=${MPC_TEMPERATURE} \
  first_ctrl_noise_scale=0.5 \
  last_ctrl_noise_scale=1.0 \
  final_noise_scale=0.1 \
  exploit_ratio=0.01 \
  exploit_noise_scale=0.01 \
  joint_noise_scale=0.15 \
  pos_noise_scale=0.03 \
  rot_noise_scale=0.03 \
  pos_rew_scale=${OBJ_POS_REW_SCALE} \
  rot_rew_scale=${OBJ_ROT_REW_SCALE} \
  joint_rew_scale=${JOINT_REW_SCALE} \
  base_pos_rew_scale=${BASE_POS_REW_SCALE} \
  base_rot_rew_scale=${BASE_ROT_REW_SCALE} \
  vel_rew_scale=${VEL_REW_SCALE} \
  +contact_rew_scale=${CONTACT_REW_SCALE}
