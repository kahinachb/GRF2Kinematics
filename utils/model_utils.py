import numpy as np
from utils.linear_algebra_utils import *

def get_thighR_pose(mks_positions, knee_offset,gender='male',):
    """
    Calculate the pose of the right thigh based on motion capture marker positions.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers. 
                                Expected keys include 'RHip', 'RKNE', 'r_mknee_study', 
                                'RIAS', 'LIAS', 'RFLE', and 'RFME'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the right thigh. The matrix 
                   includes rotation and translation components.
    """
    if gender == 'male':
        ratio_x = 0.3
        ratio_y = 0.37
        ratio_z = 0.361
    else : 
        ratio_x = 0.3
        ratio_y = 0.336
        ratio_z = 0.372

    pose = np.eye(4,4)
    X, Y, Z = [], [], []
    hip_center = np.zeros((3,1))

    dist_rPL_lPL = np.linalg.norm(mks_positions["RASI"]-mks_positions["LASI"])
    virtual_pelvis_pose = get_virtual_pelvis_pose(mks_positions)
    hip_center = virtual_pelvis_pose[:3, 3].reshape(3,1)

    hip_center = hip_center + virtual_pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(-ratio_x*dist_rPL_lPL, 0.0, 0.0)
    hip_center = hip_center + virtual_pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(0.0, -ratio_y*dist_rPL_lPL, 0.0)
    hip_center = hip_center + virtual_pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(0.0, 0.0, ratio_z*dist_rPL_lPL)

    knee_center = knee_joint_center(
    HJC=hip_center.reshape(3,),
    KNE_lat=mks_positions['RKNE'],
    THI=mks_positions['RTHI'],  # thigh wand marker in Plug-in Gait
    knee_offset=knee_offset,
    side='right').reshape(3,1)

    Y = hip_center - knee_center
    Y = Y/np.linalg.norm(Y)
    Z = (mks_positions['RKNE'] - (knee_center.T).flatten()).reshape(3,1)
    Z = Z/np.linalg.norm(Z)
    X = np.cross(Y, Z, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = hip_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose


def get_thighL_pose(mks_positions,knee_offset, gender='male'):
    """
    Calculate the pose of the left thigh based on motion capture marker positions.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers. 
                                Expected keys are 'LHip', 'LKNE', 'L_mknee_study', 'LIAS', 'RIAS', 'LFLE', and 'LFME'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the left thigh. The matrix includes
                   rotation and translation components.
    """
    if gender == 'male':
        ratio_x = 0.3
        ratio_y = 0.37
        ratio_z = 0.361
    else : 
        ratio_x = 0.3
        ratio_y = 0.336
        ratio_z = 0.372

    pose = np.eye(4,4)
    X, Y, Z = [], [], []
    hip_center = np.zeros((3,1))

    dist_rPL_lPL = np.linalg.norm(mks_positions["LASI"]-mks_positions["RASI"])
    virtual_pelvis_pose = get_virtual_pelvis_pose(mks_positions)
    hip_center = virtual_pelvis_pose[:3, 3].reshape(3,1)
    hip_center = hip_center + virtual_pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(-ratio_x*dist_rPL_lPL, 0.0, 0.0)
    hip_center = hip_center + virtual_pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(0.0, -ratio_y*dist_rPL_lPL, 0.0)
    hip_center = hip_center + virtual_pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(0.0, 0.0, -ratio_z*dist_rPL_lPL)

    knee_center = knee_joint_center(
    HJC=hip_center.reshape(3,),
    KNE_lat=mks_positions['LKNE'],
    THI=mks_positions['LTHI'],  # thigh wand marker in Plug-in Gait
    knee_offset=knee_offset,
    side='left').reshape(3,1)

    Y = hip_center - knee_center
    Y = Y/np.linalg.norm(Y)
    Z = ((knee_center.T).flatten() - mks_positions['LKNE']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)
    X = np.cross(Y, Z, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = hip_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose



#get_virtual_pelvis_pose, used to get thigh pose
def get_virtual_pelvis_pose(mks_positions):
    """
    Calculate the pelvis pose matrix from motion capture marker positions.
    The function computes the pelvis pose based on the positions of specific markers.
    It first determines the center points of the PSIS and ASIS markers, then calculates
    the X, Y, and Z axes of the pelvis coordinate system. Finally, it constructs the 
    pose matrix and ensures it is orthogonal.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of the motion capture markers.
                                The keys can be either 'r.PSIS_study', 'L.PSIS_study', 'RASI', 
                                'LASI', or 'RIPS', 'LIPS', 'RIAS', 'LIAS'.
    Returns:
    numpy.ndarray: A 4x4 pose matrix representing the pelvis pose.
    """

    pose = np.eye(4,4)
    X, Y, Z = [], [], []
    center_PSIS = []
    center_ASIS = []

    center_PSIS = (mks_positions['RPSI'] + mks_positions['LPSI']).reshape(3,1)/2.0
    center_ASIS = (mks_positions['RASI'] + mks_positions['LASI']).reshape(3,1)/2.0
    center = (mks_positions['RASI'] +
                mks_positions['LASI'] +
                mks_positions['RPSI'] +
                mks_positions['LPSI'] )/4.0
    X = center_ASIS - center_PSIS
    X = X/np.linalg.norm(X)
    Z = mks_positions['RASI'] - mks_positions['LASI']
    Z = Z/np.linalg.norm(Z)
    Y = np.cross(Z, X, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = center_ASIS.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose


def get_pelvis_pose(mks_positions, gender = 'male'):
    """
    Calculate the pelvis pose matrix from motion capture marker positions.
    The function computes the pelvis pose based on the positions of specific markers.
    It first determines the center points of the PSIS and ASIS markers, then calculates
    the X, Y, and Z axes of the pelvis coordinate system. Finally, it constructs the 
    pose matrix and ensures it is orthogonal.
    Parameters:
    mocap_mks_positions (dict): A dictionary containing the positions of the motion capture markers.
                                The keys can be either 'r.PSIS_study', 'L.PSIS_study', 'RASI', 
                                'LASI', or 'RIPS', 'LIPS', 'RIAS', 'LIAS'.
    Returns:
    numpy.ndarray: A 4x4 pose matrix representing the pelvis pose.
    """

    if gender == 'male':
        ratio_x = 0.335
        ratio_y = -0.032
        ratio_z = 0.0
    else : 
        ratio_x = 0.34
        ratio_y = 0.049
        ratio_z = 0.0

    pose = np.eye(4,4)
    center_PSIS = []
    center_ASIS = []
    center_right_ASIS_PSIS = []
    center_left_ASIS_PSIS = []
    LJC=np.zeros((3,1))

    dist_rPL_lPL = np.linalg.norm(mks_positions["RASI"]-mks_positions["LASI"])
    virtual_pelvis_pose = get_virtual_pelvis_pose(mks_positions)
    LJC = virtual_pelvis_pose[:3, 3].reshape(3,1)


    center_PSIS = (mks_positions['RPSI'] + mks_positions['LPSI']).reshape(3,1)/2.0
    center_ASIS = (mks_positions['RASI'] + mks_positions['LASI']).reshape(3,1)/2.0
    
    center_right_ASIS_PSIS = (mks_positions['RPSI'] + mks_positions['RASI']).reshape(3,1)/2.0
    center_left_ASIS_PSIS = (mks_positions['LPSI'] + mks_positions['LASI']).reshape(3,1)/2.0
    
    offset_local = col_vector_3D(
                                -ratio_x * dist_rPL_lPL,
                                +ratio_y * dist_rPL_lPL,
                                ratio_z * dist_rPL_lPL
                                )
    LJC = LJC + virtual_pelvis_pose[:3, :3] @ offset_local
 
    X = center_ASIS - center_PSIS
    X = X/np.linalg.norm(X)
    # Z = mks_positions['RASI'] - mks_positions['LASI']
    Z = center_right_ASIS_PSIS - center_left_ASIS_PSIS
    Z = Z/np.linalg.norm(Z)
    Y = np.cross(Z, X, axis=0)
    Z = np.cross(X, Y, axis=0)


    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = ((center_right_ASIS_PSIS + center_left_ASIS_PSIS)/2.0).reshape(3,)
    # pose[:3,3] = LJC.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])

    return pose


def ankle_joint_center(
KJC, ANK_lat, TIB, ankle_offset, side,
ankle_rotation_offset_deg=0.0, MANK_med=None, eps=1e-12
    ):
    """
    Estimate the Ankle Joint Center (AJC) with a chord-like construction.
    Parameters
    ----------
    KJC : (3,) array_like
        Knee joint center (same side), world coords.
    ANK_lat : (3,) array_like
        Lateral malleolus marker, world coords.
    TIB : (3,) array_like
        Shank wand/technical marker on the lateral side, world coords.
    ankle_offset : float
        Distance (m) from lateral malleolus to the ankle JC (medial direction).
        Typically ~ (AnkleWidth)/2.
    side : {'left','right'}
        Anatomical side (only used if you want to tweak sign conventions).
    ankle_rotation_offset_deg : float, optional
        Extra rotation within the medial–posterior plane (around the shank long axis).
        Use this to reflect tibial torsion or a static ankle rotation offset. Default 0.
    MANK_med : (3,) array_like or None
        Optional medial malleolus marker. If provided, AJC is midpoint of lateral & medial.
    eps : float, optional
        Numerical safety threshold.

    Returns
    -------
    AJC : (3,) ndarray
        Ankle joint center (world coords).
    """
    KJC = np.asarray(KJC, dtype=float).reshape(3)
    A = np.asarray(ANK_lat, dtype=float).reshape(3)
    T = np.asarray(TIB, dtype=float).reshape(3)

    # If medial malleolus is available, just use the midpoint (best anatomical definition)
    if MANK_med is not None:
        M = np.asarray(MANK_med, dtype=float).reshape(3)
        return 0.5 * (A + M)

    # 1) Shank long axis: knee JC -> lateral ankle
    a = A - KJC
    na = np.linalg.norm(a)
    if na < eps:
        raise ValueError("Knee and ankle are too close; cannot define shank axis.")
    a = a / na  # unit shank axis (prox→dist)

    # 2) Lateral direction: from ankle to wand, projected orthogonal to the long axis
    lat_raw = T - A
    lat_raw -= a * np.dot(lat_raw, a)  # remove component along shank axis
    nlat = np.linalg.norm(lat_raw)
    if nlat < eps:
        raise ValueError("Shank wand collinear with shank axis; cannot define lateral direction.")
    lat = lat_raw / nlat  # unit lateral

    # 3) Medial direction (wand is lateral by convention)
    med = -lat

    # If your lab’s right-side convention comes out mirrored, you can flip here:
    # if side.lower().startswith('r'):
    #     med = -med

    # 4) Posterior direction to complete an orthonormal frame
    post = np.cross(a, med)
    npost = np.linalg.norm(post)
    if npost < eps:
        raise ValueError("Degenerate configuration: cannot define posterior direction.")
    post = post / npost

    # 5) Optional rotation in the medial–posterior plane (tibial torsion / static ankle rot)
    theta = np.deg2rad(ankle_rotation_offset_deg)
    dir_vec = (np.cos(theta) * med) + (np.sin(theta) * post)

    # 6) Final AJC: from lateral malleolus toward medial/posterior
    AJC = A + ankle_offset * dir_vec
    return AJC

def knee_joint_center(
    HJC, KNE_lat, THI, knee_offset, side,
    knee_rotation_offset_deg=0.0, MKNE_med=None, eps=1e-12
):
    """
    Estimate the Knee Joint Center (KJC) using a chord function.

    Parameters
    ----------
    HJC : (3,) array_like
        Hip joint center (same side), world coords.
    KNE_lat : (3,) array_like
        Lateral knee marker, world coords.
    THI : (3,) array_like
        Thigh wand/technical marker on the lateral side, world coords.
    knee_offset : float
        Distance (m) from lateral epicondyle to knee joint center (≈ KneeWidth / 2).
    side : {'left','right'}
        Anatomical side.
    knee_rotation_offset_deg : float, optional
        Rotation around the thigh long axis (to reflect femoral rotation offset).
    MKNE_med : (3,) array_like or None
        Medial knee marker; if provided, KJC = midpoint.
    eps : float
        Numerical safety threshold.

    Returns
    -------
    KJC : (3,) ndarray
        Knee joint center (world coords).
    """
    H = np.asarray(HJC, dtype=float).reshape(3)
    K = np.asarray(KNE_lat, dtype=float).reshape(3)
    T = np.asarray(THI, dtype=float).reshape(3)

    if MKNE_med is not None:
        M = np.asarray(MKNE_med, dtype=float).reshape(3)
        return 0.5 * (K + M)

    # 1) Thigh long axis (prox→dist)
    a = K - H
    na = np.linalg.norm(a)
    if na < eps:
        raise ValueError("Hip and knee are too close; cannot define thigh axis.")
    a /= na

    # 2) Lateral direction (wand minus lateral knee, projected)
    lat_raw = T - K
    lat_raw -= a * np.dot(lat_raw, a)
    nlat = np.linalg.norm(lat_raw)
    if nlat < eps:
        raise ValueError("Thigh wand collinear with thigh axis; cannot define lateral direction.")
    lat = lat_raw / nlat

    # 3) Medial direction
    med = -lat
    # if side.lower().startswith('r'):
    #     med = -med

    # 4) Posterior direction
    post = np.cross(a, med)
    post /= np.linalg.norm(post)

    # 5) Optional rotation about the long axis
    theta = np.deg2rad(knee_rotation_offset_deg)
    dir_vec = np.cos(theta) * med + np.sin(theta) * post

    # 6) Apply chord offset
    KJC = K + knee_offset * dir_vec
    return KJC

def medial_from_ajc(JC, lat, width):
    v = JC - lat
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("AJC and lateral ankle coincide.")
    #normalisation
    med_dir = v / n
    return JC + (width/2.0) * med_dir


def _unit(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > eps, v / n, np.zeros_like(v))

def create_virtual_foot_marker(heel, toe, ankle, offset=0.01):
    """
    Create a virtual foot marker located 'offset' meters above the local foot plane.
    Inputs are arrays of shape (T, 3) in meters for HEE, TOE, ANK.
    Returns:
    virt (T, 3): virtual marker position (e.g., 'FOOT_VIRT')
    R (T, 3, 3): foot rotation matrices [x=longitudinal, y=medial-lateral, z=superior]
    origin (T, 3): chosen foot origin (here: ankle)
    """
    heel = np.asarray(heel, dtype=float)
    toe = np.asarray(toe, dtype=float)
    ankle= np.asarray(ankle,dtype=float)
    # Long axis (x): from heel to toe
    x_axis = _unit(toe - heel)

    # Foot plane normal from (HEE, TOE, ANK)
    n_plane = _unit(np.cross(toe - heel, ankle - heel))

    # y-axis (medial-lateral) lies in the plane, orthogonal to x
    y_axis = _unit(np.cross(n_plane, x_axis))

    # Re-orthogonalize z to ensure right-handed frame
    z_axis = _unit(np.cross(x_axis, y_axis))

    # Origin at ankle (you can choose MID = 0.5*(heel+toe) if you prefer)
    origin = ankle.copy()

    # Virtual marker: slightly "above" the foot plane along +z
    virt = origin + offset * z_axis

    # Rotation matrices for debugging/export (rows are basis vectors)
    R = np.stack([x_axis, y_axis, z_axis], axis=1)  # shape (T, 3, 3)
    return virt, R, origin


#If you also want a midfoot helper point (sometimes handy):

def midfoot(heel, toe):
    return 0.5*(np.asarray(heel, float) + np.asarray(toe, float))

def _unit(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > eps, v / n, np.zeros_like(v))

def create_virtual_MTOE(heel, big_toe, ankle,mankle,side, foot_width=0.05):
    """
    Create a virtual marker for the little toe (5th metatarsal head), called MTOE.
    Parameters
    ----------
    heel : array_like, shape (T,3)
        Heel marker (HEE)
    big_toe : array_like, shape (T,3)
        Big toe marker (TOE)
    ankle : array_like, shape (T,3)
        Lateral malleolus (ANK)
    foot_width : float
        Approximate foot width in meters (default 0.08 m = 8 cm)

    Returns
    -------
    MTOE : ndarray, shape (T,3)
        Virtual little toe marker (right foot = lateral side)
    """

    heel = np.asarray(heel, dtype=float)
    big_toe = np.asarray(big_toe, dtype=float)
    ankle = np.asarray(ankle, dtype=float)

    # Define local foot axes
    x_axis = _unit(big_toe - heel)  
    z_axis =_unit(np.cross(big_toe - heel, ankle - mankle)) 
    y_axis = _unit(np.cross(z_axis, x_axis))             # Lateral direction (outward)

    # Compute virtual MTOE position
    MTOE = big_toe + foot_width*y_axis
    return MTOE
