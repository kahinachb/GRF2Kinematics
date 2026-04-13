import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(os.path.dirname(script_directory))
import torch
import pytorch_kinematics as pk
import pandas as pd
import numpy as np
import pinocchio as pin
import meshcat
from utils.model_utils import build_human_model # Ta fonction locale
from pytorch_kinematics.transforms import Transform3d
import example_robot_data as robex
from pinocchio.visualize import MeshcatVisualizer


which = 'Anais'
subject = 'subject01'
task = 'walk'


# --- CONFIGURATION ---
path_joint = f"/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/{which}/{subject}/{task}/joints_filtered.csv"
joints_to_lock = ["middle_thoracic_X", "middle_thoracic_Y", "middle_thoracic_Z", 
                  "left_wrist_X", "left_wrist_Z", "right_wrist_X", "right_wrist_Z"]

mapping = {
    # Membre Inférieur Gauche
    'left_hip_Z': 'Lhip_flex_ext',
    'left_hip_X': 'Lhip_abd_add',
    'left_hip_Y': 'Lhip_int_ext_rot',
    'left_knee_Z': 'Lknee_flex_ext',
    'left_ankle_Z': 'Lankle_flex_ext',
    'left_ankle_X': 'Lankle_abd_add',
    
    # Tronc / Cervical
    'middle_lumbar_Z': 'Lumbar_flex_ext',
    'middle_lumbar_X': 'Lumbar_lateral_flex',
    'middle_cervical_Z': 'Cervical_flex_ext',
    'middle_cervical_X': 'Cervical_lat_bend',
    'middle_cervical_Y': 'Cervical_int_ext_rot',
    
    # Membre Supérieur Gauche
    'left_clavicle_joint_X': 'Lcalvicule_x',
    'left_shoulder_Z': 'Lshoulder_flex_ext',
    'left_shoulder_X': 'Lshoulder_abd_add',
    'left_shoulder_Y': 'Lshoulder_int_ext_rot',
    'left_elbow_Z': 'Lelbow_flex_ext',
    'left_elbow_Y': 'Lelbow_pron_supi',
    
    # Membre Supérieur Droit
    'right_clavicle_joint_X': 'rcalvicule_x',
    'right_shoulder_Z': 'Rshoulder_flex_ext',
    'right_shoulder_X': 'Rshoulder_abd_add',
    'right_shoulder_Y': 'Rshoulder_int_ext_rot',
    'right_elbow_Z': 'Relbow_flex_ext',
    'right_elbow_Y': 'Relbow_pron_supi',
    
    # Membre Inférieur Droit
    'right_hip_Z': 'Rhip_flex_ext',
    'right_hip_X': 'Rhip_abd_add',
    'right_hip_Y': 'Rhip_int_ext_rot',
    'right_knee_Z': 'Rknee_flex_ext',
    'right_ankle_Z': 'Rankle_flex_ext',
    'right_ankle_X': 'Rankle_abd_add',
}
# 1. CHARGEMENT DES DONNÉES
q_ref_df = pd.read_csv(path_joint)
q_ref_np = q_ref_df.to_numpy(dtype=float)

# 2. CONSTRUCTION DE LA CHAÎNE PYTORCH-KINEMATICS
human = robex.human.HumanLoader(height=1.70, weight=60, gender='male').robot
model_h = human.model
data_h = human.data
coll_h = human.collision_model
vis_h = human.visual_model

################################################################################LOCK JOINTS
###for visu
all_joint_ids = set(range(1, model_h.njoints))
joints_to_lock = ["middle_thoracic_X", "middle_thoracic_Y", "middle_thoracic_Z", "left_wrist_X", "left_wrist_Z", "right_wrist_X","right_wrist_Z"]
joint_ids_to_lock = []
for jn in joints_to_lock:
    if model_h.existJointName(jn):
        joint_ids_to_lock.append(model_h.getJointId(jn))
    else:
        print('Warning: joint ' + str(jn) + ' does not belong to the model!')

q0 = pin.neutral(model_h)
# Build reduced model
model_h, vis_h = pin.buildReducedModel(
    model_h, vis_h, joint_ids_to_lock, q0)

print(model_h.nq)
data_h = pin.Data(model_h)
# ###############################################################################################################


urdf_path = human.urdf

print(f"Loading URDF from: {urdf_path}")

with open(urdf_path, 'r') as f:
    urdf_data = f.read()

chain = pk.build_chain_from_urdf(urdf_data)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chain = chain.to(device=device)

# Récupérer l'ordre des joints attendu par PK
pk_joint_names = chain.get_joint_parameter_names()
n_pk = len(pk_joint_names)
n_samples = q_ref_np.shape[0]
print(pk_joint_names)

# 3. MAPPING DES JOINTS (Modèle Réduit -> Modèle Complet PK)
# On crée un tenseur de zéros de la taille complète de l'URDF
q_pk = torch.zeros((n_samples, n_pk), device=device, dtype=torch.float32)

# On remplit les colonnes correspondantes aux joints actifs
# On suppose que les colonnes de q_ref_df correspondent aux noms des joints
print("Mapping des joints en cours...")
joints_mapped = []
joints_ignored = []



if which =='Anais' or which =='Vinc':
    for i, pk_name in enumerate(pk_joint_names):
        # 1. Vérifier si on doit verrouiller le joint
        if pk_name in joints_to_lock:
            print('joint to lock:', pk_name)
            continue  # reste à 0.0

        csv_name = mapping.get(pk_name)

        # 3. Vérifier si le nom existe dans le CSV
        if csv_name and csv_name in q_ref_df.columns: 
            q_pk[:, i] = torch.tensor(q_ref_df[csv_name].values, device=device)
        else:
            print(f"⚠️ Joint '{pk_name}' (mapped: '{csv_name}') non trouvé dans le CSV. Fixé à 0.")


else : 
    for i, name in enumerate(pk_joint_names):
        # CONDITION : Si le joint est dans le CSV ET n'est PAS dans la liste à verrouiller
        if name in q_ref_df.columns and name not in joints_to_lock:
            q_pk[:, i] = torch.tensor(q_ref_df[name].values, device=device)
            joints_mapped.append(name)
        else:
            # Le joint reste à 0.0 (déjà initialisé par torch.zeros)
            joints_ignored.append(name)

    print(f"✅ Joints actifs ({len(joints_mapped)}) : {', '.join(joints_mapped[:5])}...")
    print(f"🔒 Joints verrouillés ou absents ({len(joints_ignored)}) : {', '.join(joints_ignored[:10])}...")

# FF_X, FF_Y, FF_Z, FF_quatx, FF_quaty, FF_quatz, FF_quatw
if which =='Anais' or which =='Vinc':
    ff_data = q_ref_df[['FF_X', 'FF_Y', 'FF_Z', 'FF_quatx', 'FF_quaty', 'FF_quatz', 'FF_quatw']].to_numpy()
else: 
    ff_data = q_ref_df[['root_joint','root_joint.1','root_joint.2','root_joint.3','root_joint.4','root_joint.5','root_joint.6']].to_numpy()

ff_tensor = torch.tensor(ff_data, device=device, dtype=torch.float32)

pos_base = ff_tensor[:, :3]
quat_xyzw = ff_tensor[:, 3:]

# /!\ IMPORTANT : pytorch-kinematics utilise souvent l'ordre (w, x, y, z)
quat_wxyz = torch.cat([quat_xyzw[:, 3:], quat_xyzw[:, :3]], dim=1)

# Création de la transformation du FreeFlyer
base_transform = Transform3d(pos=pos_base, rot=quat_wxyz, device=device)


with torch.no_grad():
    # 1. Calcul de la FK relative (repère local à la racine du robot)
    relative_transforms = chain.forward_kinematics(q_pk)
    
    # 2. Application du FreeFlyer pour chaque frame
    # On crée un nouveau dictionnaire pour stocker les positions mondiales
    world_transforms = {}
    for frame_name, transform in relative_transforms.items():
        # La composition se fait avec l'opérateur compose() ou simplement '@'
        # base_transform (Monde -> Racine) @ transform (Racine -> Joint)
        world_transforms[frame_name] = base_transform.compose(transform)

# 5. VISUALISATION AVEC MESHCAT (Via Pinocchio pour le rendu mesh)
# Note: On garde le modèle complet ici pour l'affichage visuel simple
viz = meshcat.Visualizer().open()

viz_human = MeshcatVisualizer(model_h, coll_h, vis_h)
viz_human.initViewer(viz, open=True)
viz_human.viewer.delete()  # clear if relaunch
viz_human.loadViewerModel("ref",color=[0.0, 1.0, 0.0, 0.8])

def set_sphere(name, pos, color=0xff0000):
    viz[f"pk_markers/{name}"].set_object(
        meshcat.geometry.Sphere(0.01),
        meshcat.geometry.MeshLambertMaterial(color=color)
    )
    viz[f"pk_markers/{name}"].set_transform(meshcat.transformations.translation_matrix(pos))

# Boucle d'animation
print("Visualisation en cours sur Meshcat...")
data_list = []
from scipy.spatial.transform import Rotation as R

angle = -np.pi / 2 
R_corr = np.array([[1, 0,           0          ],
                   [0, np.cos(angle), -np.sin(angle)],
                   [0, np.sin(angle),  np.cos(angle)]])

for i in range(n_samples): # On saute des frames pour la fluidité
    frame_data = {}
    # Pour chaque frame de l'URDF calculée par PK
    for frame_name, transform in world_transforms.items(): # <--- Utilise world_transforms
        matrix = transform.get_matrix()[i].cpu().numpy()
        pos = matrix[:3, 3]
        set_sphere(frame_name, pos)

        frame_data[f"{frame_name}_X"] = pos[0]
        frame_data[f"{frame_name}_Y"] = pos[1]
        frame_data[f"{frame_name}_Z"] = pos[2]
    
    data_list.append(frame_data)

    q_current = q_ref_df.iloc[i].to_numpy()
    pos_bassin_rnea = q_current[0:3]
    quat_bassin = q_current[3:7] # qx, qy, qz, qw

    quat_original = pin.Quaternion(q_current[6], q_current[3], q_current[4], q_current[5]) #(w,x,y,z) or : q_current[3:7]
    R_original = quat_original.toRotationMatrix()

    R_final = R_corr @ R_original 
    quat_final = pin.Quaternion( R_final)
    
    q_ref_np[i][3:7] = [quat_final.x, quat_final.y, quat_final.z, quat_final.w]
    q_ref_np[i][0:3] = R_corr @ q_current[0:3]

    viz_human.display(q_ref_np[i])
    
df_output = pd.DataFrame(data_list)

# Définir le nom du fichier de sortie (par exemple dans le même dossier que le script)
output_csv_path = f"pk_fk_results_{subject}_{task}.csv"
df_output.to_csv(output_csv_path, index=False)

print(f"✅ Sauvegarde terminée : {output_csv_path}")
print(f"Dimensions du fichier : {df_output.shape}") # (frames, joints * 3)