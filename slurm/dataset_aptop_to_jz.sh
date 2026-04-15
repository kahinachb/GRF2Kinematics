#!/bin/bash

source_path="/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/processed_data_pelvis"
target_path="ulm94jm@jean-zay3.idris.fr:/lustre/fsn1/projects/rech/vsi/ulm94jm/dataset_grf2kine"

# FIRST SYNC RIGHT AWAY
rsync -azh -e "ssh -J kchalabi@ssh.laas.fr" "$source_path" "$target_path" \
      --progress \
      --delete --force \
      --exclude=".git" \
      --exclude="./smpl/amass/smplx/" \
      --exclude="./smpl/amass/smplh/joint_segment_data/" \
      --exclude="./smpl/amass/smplh/joint_segment_data_old/" \
      --exclude="./smpl/amass/smplh/latent_npz_data/" \
      --exclude="./smpl/amass/smplh/raw_npz_data/"
      
while inotifywait -r -e modify,create,delete "$source_path"
do
    rsync -azh "$source_path" "$target_path" \
          --progress \
          --delete --force \
          --exclude=".git" \
          --exclude="./smpl/amass/smplx/" \
          --exclude="./smpl/amass/smplh/joint_segment_data/" \
          --exclude="./smpl/amass/smplh/joint_segment_data_old/" \
          --exclude="./smpl/amass/smplh/latent_npz_data/" \
          --exclude="./smpl/amass/smplh/raw_npz_data/" 
done