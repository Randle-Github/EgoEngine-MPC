#!/usr/bin/env bash
set -euo pipefail

TASK=p36-tea
HAND_TYPE=bimanual
DATA_ID=10
ROBOT_TYPE=xhand
DATASET_NAME=gigahand

PARTICIPANT="${TASK%%-*}"
SCENE="${TASK#*-}"
SEQUENCE_ID="$(printf "%04d" "${DATA_ID}")"

# put your raw data under folder raw/{dataset_name}/ in your dataset folder

# read data from self collected dataset
uv run spider/process_datasets/gigahand.py \
  --participant=${PARTICIPANT} \
  --scene=${SCENE} \
  --sequence-id=${SEQUENCE_ID} \
  --embodiment-type=${HAND_TYPE} \
  --show-viewer=True

# decompose object
uv run spider/preprocess/decompose_fast.py --task=${TASK} --dataset-name=${DATASET_NAME} --data-id=${DATA_ID} --embodiment-type=${HAND_TYPE}

# detect contact (optional)
uv run spider/preprocess/detect_contact.py --task=${TASK} --dataset-name=${DATASET_NAME} --data-id=${DATA_ID} --embodiment-type=${HAND_TYPE}

# generate scene
uv run spider/preprocess/generate_xml.py --task=${TASK} --dataset-name=${DATASET_NAME} --data-id=${DATA_ID} --embodiment-type=${HAND_TYPE} --robot-type=${ROBOT_TYPE}

# kinematic retargeting
uv run spider/preprocess/ik.py --task=${TASK} --dataset-name=${DATASET_NAME} --data-id=${DATA_ID} --embodiment-type=${HAND_TYPE} --robot-type=${ROBOT_TYPE} --open-hand

# retargeting
uv run examples/run_mjwp.py +override=${DATASET_NAME} task=${TASK} data_id=${DATA_ID} robot_type=${ROBOT_TYPE} embodiment_type=${HAND_TYPE}

# read data for deployment (optional)
uv run spider/postprocess/read_to_robot.py --task=${TASK} --dataset-name=${DATASET_NAME} --data-id=${DATA_ID} --robot-type=${ROBOT_TYPE} --embodiment-type=${HAND_TYPE}
