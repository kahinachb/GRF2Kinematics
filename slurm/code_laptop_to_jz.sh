#!/bin/bash

source_path="/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics"
target_path="ulm94jm@jean-zay3.idris.fr:/lustre/fswork/projects/rech/vsi/ulm94jm"

# FIRST SYNC RIGHT AWAY
rsync -azh -e "ssh -J kchalabi@ssh.laas.fr" "$source_path" "$target_path" \
      --progress \
      --delete --force \
      --exclude=".git" \
      --exclude="./DATA" \
      --exclude="./checkpoints_diff_local" \
      --exclude="./checkpoints_diff" \
      --exclude="./__pycache__" \
      --exclude="./minimal_model_local"\
      --exclude="./motif"\
      --exclude="./DATA"\
      --exclude="./slurm"
      
while inotifywait -r -e modify,create,delete "$source_path"
do
    rsync -azh "$source_path" "$target_path" \
            --progress \
            --delete --force \
            --exclude=".git" \
            --exclude="./DATA" \
            --exclude="./checkpoints_diff_local" \
            --exclude="./checkpoints_diff" \
            --exclude="./__pycache__" \
            --exclude="./minimal_model_local"\
            --exclude="./motif"\
            --exclude="./DATA"\
            --exclude="./slurm"
done