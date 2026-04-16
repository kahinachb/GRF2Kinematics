source_path="ulm94jm@jean-zay3.idris.fr:/lustre/fswork/projects/rech/vsi/ulm94jm/GRF2Kinematics/results_pelvis_cop"
target_path="/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics"

rsync -azh -e "ssh -J kchalabi@ssh.laas.fr" "$source_path" "$target_path" 