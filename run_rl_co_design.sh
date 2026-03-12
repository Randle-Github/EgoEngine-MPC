#!/usr/bin/env bash
export PYTHONPATH="$(pwd):$PYTHONPATH"

export HAND_TYPE=right
export participant=p36
export scene=tea
export sequence_id=0010

DATASET_DIR="/home/yhan389/Desktop/EgoEngine-MPC/example_datasets"

# output: task_info.json, and trajectory_kinematic.npz
# python3 spider/process_datasets/gigahand.py \
#   --dataset-dir "${DATASET_DIR}" \
#   --participant "${participant}" \
#   --scene "${scene}" \
#   --sequence-id "${sequence_id}" \
#   --embodiment_type "${HAND_TYPE}" \
#   --save_video True

export DATASET_NAME=gigahand
export TASK=mustard_901_1  # default: p36-tea
export DATA_ID=0 # default: 10 for task p36-tea

# output: modified task_info.json, and object aseets under example_datasets/processed/${DATASET_NAME}/aseets/objects
# python3 spider/preprocess/decompose_fast.py \
#   --dataset-dir "${DATASET_DIR}" \
#   --dataset-name "${DATASET_NAME}" \
#   --embodiment_type "${HAND_TYPE}" \
#   --task "${TASK}" \
#   --data-id "${DATA_ID}"

# output: trajectory_keypoints.npz (with contact data), and contact.mp4
# python3 spider/preprocess/detect_contact.py \
#   --dataset-dir "${DATASET_DIR}" \
#   --dataset-name "${DATASET_NAME}" \
#   --embodiment_type "${HAND_TYPE}" \
#   --task "${TASK}" \
#   --data-id "${DATA_ID}"

export ROBOT_TYPE=mano

# output: scene.xml, scene_eq.xml, and modified task_info.json
# python3 spider/preprocess/generate_xml.py \
#   --dataset-dir "${DATASET_DIR}" \
#   --dataset-name "${DATASET_NAME}" \
#   --robot-type "${ROBOT_TYPE}" \
#   --embodiment_type "${HAND_TYPE}" \
#   --task "${TASK}" \
#   --data-id "${DATA_ID}"

# output: modified trajectory_kinematic.npz, modified trajectory_ikrollout.npz, and visualization_ik.mp4
# python3 spider/preprocess/ik.py \
#   --dataset-dir "${DATASET_DIR}" \
#   --dataset-name "${DATASET_NAME}" \
#   --robot-type "${ROBOT_TYPE}" \
#   --embodiment_type "${HAND_TYPE}" \
#   --task "${TASK}" \
#   --data-id "${DATA_ID}"

export SPIDER_DISABLE_CUDA_GRAPH=1

# output: policy models, reward curves, and the simulation videos
python3 examples/run_mjwp_rl_rsl.py \
  --dataset-dir "${DATASET_DIR}" \
  --dataset-name "${DATASET_NAME}" \
  --robot-type "${ROBOT_TYPE}" \
  --embodiment_type "${HAND_TYPE}" \
  --task "${TASK}" \
  --data-id "${DATA_ID}"
