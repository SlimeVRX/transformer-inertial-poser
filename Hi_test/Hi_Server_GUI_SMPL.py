import time
import json
import threading
import numpy as np
import tkinter as tk
# from llv.socketconn import SocketConn


cos = np.cos
sin = np.sin

char_data = {
    'w_head_X': 0,
    'w_head_Y': 0,
    'w_head_Z': 0,

    'w_neck_01_X': 0,
    'w_neck_01_Y': 0,
    'w_neck_01_Z': 0,

    #----------------------
    'w_clavicle_r_X': 0,
    'w_clavicle_r_Y': 0,
    'w_clavicle_r_Z': 0,

    'w_upperarm_r_X': 0,
    'w_upperarm_r_Y': 0,
    'w_upperarm_r_Z': 0,

    'w_lowerarm_r_X': 0,
    'w_lowerarm_r_Y': 0,
    'w_lowerarm_r_Z': 0,

    'w_hand_r_X': 0,
    'w_hand_r_Y': 0,
    'w_hand_r_Z': 0,

    #----------------------
    'w_clavicle_l_X': 0,
    'w_clavicle_l_Y': 0,
    'w_clavicle_l_Z': 0,

    'w_upperarm_l_X': 0,
    'w_upperarm_l_Y': 0,
    'w_upperarm_l_Z': 0,

    'w_lowerarm_l_X': 0,
    'w_lowerarm_l_Y': 0,
    'w_lowerarm_l_Z': 0,

    'w_hand_l_X': 0,
    'w_hand_l_Y': 0,
    'w_hand_l_Z': 0,

    #----------------------
    'w_spine_03_X': 0,
    'w_spine_03_Y': 0,
    'w_spine_03_Z': 0,

    'w_spine_02_X': 0,
    'w_spine_02_Y': 0,
    'w_spine_02_Z': 0,

    'w_spine_01_X': 0,
    'w_spine_01_Y': 0,
    'w_spine_01_Z': 0,

    'w_pelvis_X': 0,
    'w_pelvis_Y': 0,
    'w_pelvis_Z': 0,

    #----------------------
    'w_thigh_r_X': 0,
    'w_thigh_r_Y': 0,
    'w_thigh_r_Z': 0,

    'w_calf_r_X': 0,
    'w_calf_r_Y': 0,
    'w_calf_r_Z': 0,

    'w_foot_r_X': 0,
    'w_foot_r_Y': 0,
    'w_foot_r_Z': 0,

    'w_ball_r_X': 0,
    'w_ball_r_Y': 0,
    'w_ball_r_Z': 0,

    #----------------------
    'w_thigh_l_X': 0,
    'w_thigh_l_Y': 0,
    'w_thigh_l_Z': 0,

    'w_calf_l_X': 0,
    'w_calf_l_Y': 0,
    'w_calf_l_Z': 0,

    'w_foot_l_X': 0,
    'w_foot_l_Y': 0,
    'w_foot_l_Z': 0,

    'w_ball_l_X': 0,
    'w_ball_l_Y': 0,
    'w_ball_l_Z': 0,

}

def ch_rotx(theta):
    return np.array([[1, 0, 0],
                     [0, cos(theta), -sin(theta)],
                     [0, sin(theta), cos(theta)]])

def ch_roty(theta):
    return np.array([[cos(theta), 0, sin(theta)],
                     [0, 1, 0],
                     [-sin(theta), 0, cos(theta)]])

def ch_rotz(theta):
    return np.array([[cos(theta), -sin(theta), 0],
                     [sin(theta), cos(theta), 0],
                     [0, 0, 1]])

def ch_m2q(Cb2n):
    # Attitude array to Quaternion
    # Input: Cb2n
    # Output: Qb2n
    C11 = Cb2n[0,0]; C12 = Cb2n[0,1]; C13 = Cb2n[0,2]
    C21 = Cb2n[1,0]; C22 = Cb2n[1,1]; C23 = Cb2n[1,2]
    C31 = Cb2n[2,0]; C32 = Cb2n[2,1]; C33 = Cb2n[2,2]

    if C11 >= C22+C33:
        q1 = 0.5*np.sqrt(1+C11-C22-C33)
        q0 = (C32-C23)/(4*q1); q2 = (C12+C21)/(4*q1); q3 = (C13+C31)/(4*q1)
    elif C22>= C11+C33:
        q2 = 0.5*np.sqrt(1-C11+C22-C33)
        q0 = (C13-C31)/(4*q2); q1 = (C12+C21)/(4*q2); q3 = (C23+C32)/(4*q2)
    elif C33>=C11+C22:
        q3 = 0.5*np.sqrt(1-C11-C22+C33)
        q0 = (C21-C12)/(4*q3); q1 = (C13+C31)/(4*q3); q2 = (C23+C32)/(4*q3)
    else:
        q0 = 0.5*np.sqrt(1+C11+C22+C33)
        q1 = (C32-C23)/(4*q0); q2 = (C13-C31)/(4*q0); q3 = (C21-C12)/(4*q0)
    Qb2n = np.array([q1, q2, q3, q0], dtype=np.float64)
    return Qb2n

# handshake = {
#     "HANDSHAKE": {
#         "version": "1.3.1",
#         "userName": "Azzurri2Python",
#         "UUID": "8EF5BA6A-A73E-419A-9BDC-7E18759C180C",
#         "deviceName": "iPhone13,4",
#         "rig": "MetaHuman",
#         "name": "FUCKYOU LiveLink",
#         "mode": "Room",
#         "syncFPS": 13,
#         "useRootMotion": True,
#         "whoami": "YourMother",
#     },
#     "version": "1.3.1",
#     "userName": "Azzurri2Python"
# }

import torch
import articulate as art
import sys
import time
# from vmc_romp.log import Log
from vmc_romp.configuration import Configuration
from scipy.spatial.transform import Rotation as R
from vmc_romp.vmc import Assistant as VMCAssistant, Bone, Position, Quaternion, Timestamp


# Configuration
configuration: dict = {
    "host"  : "127.0.0.1",
    "port"  : 39539,
    # "port"  : 39546,
    "name"  : "example",
    "delta" : 0.0
}

# # Logging
# sys.stdout = Log(filename="vmc.log", is_error=False)
# sys.stderr = Log(filename="vmc.log", is_error=True)

# ROMP
root_position_offset = Position(0.0, 1, 0.0)
hips_bone_position = Position(0.0, 0.0, 0.0)

vrm_bone_names = {
    0 : "Hips", # Pelvis
    1 : "LeftUpperLeg", # L_Hip
    2 : "RightUpperLeg", # R_Hip
    3 : "Spine", # Spine1
    4 : "LeftLowerLeg", # L_Knee
    5 : "RightLowerLeg", # R_Knee
    6 : "Chest", # Spine2
    7 : "LeftFoot", # L_Ankle
    8 : "RightFoot", # R_Ankle
    9 : "UpperChest", # Spine3
    10 : "LeftToes", # L_Foot
    11 : "RightToes", # R_Foot
    12 : "Neck", # Neck
    13 : "LeftShoulder", # L_Collar
    14 : "RightShoulder", # R_Collar
    15 : "Head", # Head
    16 : "LeftUpperArm", # L_Shoulder
    17 : "RightUpperArm", # R_Shoulder
    18 : "LeftLowerArm", # L_Elbow
    19 : "RightLowerArm", # R_Elbow
    20 : "LeftHand", # L_Wrist
    21 : "RightHand", # R_Wrist
    22 : "LeftMiddleProximal", # L_Hand
    23 : "RightMiddleProximal" # R_hand
}

# VMC
configuration = Configuration("vmc.yml", configuration)
vmc = VMCAssistant(
    configuration['host'],
    configuration["port"],
    configuration["name"]
)

started_at = Timestamp()
start = time.time()

gui = tk.Tk()
gui.geometry('960x960')

class ZMQ_REP_Task:
    def __init__(self):
        self._running = True
        # self.socket_conn = SocketConn("127.0.0.1", port=8081)
        # self.socket_conn.send_json_encoded(json.dumps(handshake, ensure_ascii=False).encode("utf-8"))

    def run(self, args):
        global char_data
        fps = 13
        sleep_time = 1 / fps

        while self._running:

            eul_rad = np.deg2rad([char_data['w_head_X'], char_data['w_head_Y'], char_data['w_head_Z']])
            head = list(ch_m2q(ch_rotz(eul_rad[2]). dot(ch_rotx(eul_rad[0])). dot(ch_roty(eul_rad[1]))))

            eul_rad = np.deg2rad([char_data['w_neck_01_X'], char_data['w_neck_01_Y'], char_data['w_neck_01_Z']])
            neck_01 = list(ch_m2q(ch_rotz(eul_rad[2]). dot(ch_rotx(eul_rad[0])). dot(ch_roty(eul_rad[1]))))

            eul_rad = np.deg2rad([char_data['w_spine_03_X'], char_data['w_spine_03_Y'], char_data['w_spine_03_Z']])
            spine_03 = list(ch_m2q(ch_rotz(eul_rad[2]). dot(ch_rotx(eul_rad[0])). dot(ch_roty(eul_rad[1]))))

            eul_rad = np.deg2rad([char_data['w_spine_02_X'], char_data['w_spine_02_Y'], char_data['w_spine_02_Z']])
            spine_02 = list(ch_m2q(ch_rotz(eul_rad[2]). dot(ch_rotx(eul_rad[0])). dot(ch_roty(eul_rad[1]))))

            eul_rad = np.deg2rad([char_data['w_spine_01_X'], char_data['w_spine_01_Y'], char_data['w_spine_01_Z']])
            spine_01 = list(ch_m2q(ch_rotz(eul_rad[2]). dot(ch_rotx(eul_rad[0])). dot(ch_roty(eul_rad[1]))))

            eul_rad = np.deg2rad([char_data['w_pelvis_X'], char_data['w_pelvis_Y'], char_data['w_pelvis_Z']])
            pelvis = list(ch_m2q(ch_rotz(eul_rad[2]). dot(ch_rotx(eul_rad[0])). dot(ch_roty(eul_rad[1]))))

            eul_rad = np.deg2rad([char_data['w_clavicle_r_X'], char_data['w_clavicle_r_Y'], char_data['w_clavicle_r_Z']])
            clavicle_r = list(ch_m2q(ch_rotz(eul_rad[2]). dot(ch_rotx(eul_rad[0])). dot(ch_roty(eul_rad[1]))))

            eul_rad = np.deg2rad([char_data['w_upperarm_r_X'], char_data['w_upperarm_r_Y'], char_data['w_upperarm_r_Z']])
            upperarm_r = list(ch_m2q(ch_rotz(eul_rad[2]). dot(ch_rotx(eul_rad[0])). dot(ch_roty(eul_rad[1]))))

            eul_rad = np.deg2rad([char_data['w_lowerarm_r_X'], char_data['w_lowerarm_r_Y'], char_data['w_lowerarm_r_Z']])
            lowerarm_r = list(ch_m2q(ch_rotz(eul_rad[2]). dot(ch_rotx(eul_rad[0])). dot(ch_roty(eul_rad[1]))))

            eul_rad = np.deg2rad([char_data['w_hand_r_X'], char_data['w_hand_r_Y'], char_data['w_hand_r_Z']])
            hand_r = list(ch_m2q(ch_rotz(eul_rad[2]). dot(ch_rotx(eul_rad[0])). dot(ch_roty(eul_rad[1]))))

            eul_rad = np.deg2rad([char_data['w_clavicle_l_X'], char_data['w_clavicle_l_Y'], char_data['w_clavicle_l_Z']])
            clavicle_l = list(ch_m2q(ch_rotz(eul_rad[2]). dot(ch_rotx(eul_rad[0])). dot(ch_roty(eul_rad[1]))))

            eul_rad = np.deg2rad([char_data['w_upperarm_l_X'], char_data['w_upperarm_l_Y'], char_data['w_upperarm_l_Z']])
            upperarm_l = list(ch_m2q(ch_rotz(eul_rad[2]). dot(ch_rotx(eul_rad[0])). dot(ch_roty(eul_rad[1]))))

            eul_rad = np.deg2rad([char_data['w_lowerarm_l_X'], char_data['w_lowerarm_l_Y'], char_data['w_lowerarm_l_Z']])
            lowerarm_l = list(ch_m2q(ch_rotz(eul_rad[2]). dot(ch_rotx(eul_rad[0])). dot(ch_roty(eul_rad[1]))))

            eul_rad = np.deg2rad([char_data['w_hand_l_X'], char_data['w_hand_l_Y'], char_data['w_hand_l_Z']])
            hand_l = list(ch_m2q(ch_rotz(eul_rad[2]). dot(ch_rotx(eul_rad[0])). dot(ch_roty(eul_rad[1]))))

            eul_rad = np.deg2rad([char_data['w_thigh_r_X'], char_data['w_thigh_r_Y'], char_data['w_thigh_r_Z']])
            thigh_r = list(ch_m2q(ch_rotz(eul_rad[2]). dot(ch_rotx(eul_rad[0])). dot(ch_roty(eul_rad[1]))))

            eul_rad = np.deg2rad([char_data['w_calf_r_X'], char_data['w_calf_r_Y'], char_data['w_calf_r_Z']])
            calf_r = list(ch_m2q(ch_rotz(eul_rad[2]). dot(ch_rotx(eul_rad[0])). dot(ch_roty(eul_rad[1]))))

            eul_rad = np.deg2rad([char_data['w_foot_r_X'], char_data['w_foot_r_Y'], char_data['w_foot_r_Z']])
            foot_r = list(ch_m2q(ch_rotz(eul_rad[2]). dot(ch_rotx(eul_rad[0])). dot(ch_roty(eul_rad[1]))))

            eul_rad = np.deg2rad([char_data['w_ball_r_X'], char_data['w_ball_r_Y'], char_data['w_ball_r_Z']])
            ball_r = list(ch_m2q(ch_rotz(eul_rad[2]). dot(ch_rotx(eul_rad[0])). dot(ch_roty(eul_rad[1]))))

            eul_rad = np.deg2rad([char_data['w_thigh_l_X'], char_data['w_thigh_l_Y'], char_data['w_thigh_l_Z']])
            thigh_l = list(ch_m2q(ch_rotz(eul_rad[2]). dot(ch_rotx(eul_rad[0])). dot(ch_roty(eul_rad[1]))))

            eul_rad = np.deg2rad([char_data['w_calf_l_X'], char_data['w_calf_l_Y'], char_data['w_calf_l_Z']])
            calf_l = list(ch_m2q(ch_rotz(eul_rad[2]). dot(ch_rotx(eul_rad[0])). dot(ch_roty(eul_rad[1]))))

            eul_rad = np.deg2rad([char_data['w_foot_l_X'], char_data['w_foot_l_Y'], char_data['w_foot_l_Z']])
            foot_l = list(ch_m2q(ch_rotz(eul_rad[2]). dot(ch_rotx(eul_rad[0])). dot(ch_roty(eul_rad[1]))))

            eul_rad = np.deg2rad([char_data['w_ball_l_X'], char_data['w_ball_l_Y'], char_data['w_ball_l_Z']])
            ball_l = list(ch_m2q(ch_rotz(eul_rad[2]). dot(ch_rotx(eul_rad[0])). dot(ch_roty(eul_rad[1]))))


            fc = {'PF': 0, 'Rig': 'MetaHuman', 'LeftHand': {'Rotations': {'hand_l': hand_l, 'lowerarm_twist_01_l': [0.0, 0.0, 0.0, 1.0], 'index_01_l': [0.0, 0.0, 0.0, 1.0], 'index_02_l': [0.0, 0.0, 0.0, 1.0], 'index_03_l': [0.0, 0.0, 0.0, 1.0], 'middle_01_l': [0.0, 0.0, 0.0, 1.0], 'middle_02_l': [0.0, 0.0, 0.0, 1.0], 'middle_03_l': [0.0, 0.0, 0.0, 1.0], 'ring_01_l': [0.0, 0.0, 0.0, 1.0], 'ring_02_l': [0.0, 0.0, 0.0, 1.0], 'ring_03_l': [0.0, 0.0, 0.0, 1.0], 'pinky_01_l': [0.0, 0.0, 0.0, 1.0], 'pinky_02_l': [0.0, 0.0, 0.0, 1.0], 'pinky_03_l': [0.0, 0.0, 0.0, 1.0], 'thumb_01_l': [0.0, 0.0, 0.0, 1.0], 'thumb_02_l': [0.0, 0.0, 0.0, 1.0], 'thumb_03_l': [0.0, 0.0, 0.0, 1.0]}, 'Vectors': {'PointScreen': [-0.215, -0.204], 'FingerIk': [1.488, 0.471, -1.293]}}, 'RightHand': {'Rotations': {'hand_r': hand_r, 'lowerarm_twist_01_r': [0.0, 0.0, 0.0, 1.0], 'index_01_r': [0.0, 0.0, 0.0, 1.0], 'index_02_r': [0.0, 0.0, 0.0, 1.0], 'index_03_r': [0.0, 0.0, 0.0, 1.0], 'middle_01_r': [0.0, 0.0, 0.0, 1.0], 'middle_02_r': [0.0, 0.0, 0.0, 1.0], 'middle_03_r': [0.0, 0.0, 0.0, 1.0], 'ring_01_r': [0.0, 0.0, 0.0, 1.0], 'ring_02_r': [0.0, 0.0, 0.0, 1.0], 'ring_03_r': [0.0, 0.0, 0.0, 1.0], 'pinky_01_r': [0.0, 0.0, 0.0, 1.0], 'pinky_02_r': [0.0, 0.0, 0.0, 1.0], 'pinky_03_r': [0.0, 0.0, 0.0, 1.0], 'thumb_01_r': [0.0, 0.0, 0.0, 1.0], 'thumb_02_r': [0.0, 0.0, 0.0, 1.0], 'thumb_03_r': [0.0, 0.0, 0.0, 1.0]}, 'Vectors': {'PointScreen': [0.07, -0.159], 'FingerIk': [1.858, -1.952, 1.183]}}, 'Body': {'Rotations': {'pelvis': pelvis, 'thigh_r': thigh_r, 'calf_r': calf_r, 'foot_r': foot_r, 'ball_r': ball_r, 'thigh_l': thigh_l, 'calf_l': calf_l, 'foot_l': foot_l, 'ball_l': ball_l, 'spine_01': spine_01, 'spine_02': spine_02, 'spine_03': spine_03, 'neck_01': neck_01, 'head': head, 'clavicle_l': clavicle_l, 'upperarm_l': upperarm_l, 'lowerarm_l': lowerarm_l, 'clavicle_r': clavicle_r, 'upperarm_r': upperarm_r, 'lowerarm_r': lowerarm_r}, 'Scalars': {'VisTorso': 1, 'VisLegL': 1, 'VisLegR': 1, 'VisArmL': 1, 'VisArmR': 1, 'BodyHeight': 0.955, 'StableFoot': 0.0, 'ChestYaw': -0.028, 'StanceYaw': -0.056, 'HandZoneL': 1, 'HandZoneR': 3, 'IsCrouching': 1, 'ShrugL': 0.0, 'ShrugR': 0.0}, 'Events': {'Footstep': {'Count': 5, 'Magnitude': 0.247}, 'SidestepL': {'Count': 4, 'Magnitude': 1.0}, 'SidestepR': {'Count': 5, 'Magnitude': -1.0}, 'JumpCount': {'Count': 0, 'Magnitude': 0.121}, 'FeetSplit': {'Count': 0, 'Magnitude': 0.0}, 'ArmPump': {'Count': 1, 'Magnitude': -0.063}, 'ArmFlex': {'Count': 0, 'Magnitude': 0.0}, 'ArmGestureL': {'Count': 10, 'Current': 10}, 'ArmGestureR': {'Count': 7, 'Current': 10}}, 'Vectors': {'HipLean': [-0.01, 0.064], 'HipScreen': [0.046, -0.088], 'ChestScreen': [0.057, 0.055], 'HandIkL': [1.11, 0.706, -1.088], 'HandIkR': [1.075, -1.075, 0.647]}, 'Loco': 'AAAA'}, 'ModelLatency': 32, 'Timestamp': 6429.719, 'rig': 'MetaHuman'}
            # fc_right_json = json.dumps(fc, ensure_ascii=False)
            # self.socket_conn.send_json_encoded(fc_right_json.encode("utf-8"))
            print("pelvis = ", fc['Body']['Rotations']['pelvis'])
            # # print("spine_01 = ", fc['Body']['Rotations']['spine_01'])

            pose_q = np.array([
                np.array([pelvis[3], pelvis[0], pelvis[1], pelvis[2]]),
                np.array([thigh_l[3], thigh_l[0], thigh_l[1], thigh_l[2]]),
                np.array([thigh_r[3], thigh_r[0], thigh_r[1], thigh_r[2]]),
                np.array([spine_01[3], spine_01[0], spine_01[1], spine_01[2]]),
                np.array([calf_l[3], calf_l[0], calf_l[1], calf_l[2]]),
                np.array([calf_r[3], calf_r[0], calf_r[1], calf_r[2]]),
                np.array([spine_02[3], spine_02[0], spine_02[1], spine_02[2]]),
                np.array([foot_l[3], foot_l[0], foot_l[1], foot_l[2]]),
                np.array([foot_r[3], foot_r[0], foot_r[1], foot_r[2]]),
                np.array([spine_03[3], spine_03[0], spine_03[1], spine_03[2]]),
                np.array([ball_l[3], ball_l[0], ball_l[1], ball_l[2]]),
                np.array([ball_r[3], ball_r[0], ball_r[1], ball_r[2]]),
                np.array([neck_01[3], neck_01[0], neck_01[1], neck_01[2]]),
                np.array([clavicle_l[3], clavicle_l[0], clavicle_l[1], clavicle_l[2]]),
                np.array([clavicle_r[3], clavicle_r[0], clavicle_r[1], clavicle_r[2]]),
                np.array([head[3], head[0], head[1], head[2]]),
                np.array([upperarm_l[3], upperarm_l[0], upperarm_l[1], upperarm_l[2]]),
                np.array([upperarm_r[3], upperarm_r[0], upperarm_r[1], upperarm_r[2]]),
                np.array([lowerarm_l[3], lowerarm_l[0], lowerarm_l[1], lowerarm_l[2]]),
                np.array([lowerarm_r[3], lowerarm_r[0], lowerarm_r[1], lowerarm_r[2]]),
                np.array([hand_l[3], hand_l[0], hand_l[1], hand_l[2]]),
                np.array([hand_r[3], hand_r[0], hand_r[1], hand_r[2]]),
                np.array([hand_l[3], hand_l[0], hand_l[1], hand_l[2]]),
                np.array([hand_r[3], hand_r[0], hand_r[1], hand_r[2]])
            ])
            pose_q = torch.from_numpy(np.array(pose_q)).float()

            # vrm_bone_names = {
            #     0 : "Hips", # Pelvis                    pelvis
            #     1 : "LeftUpperLeg", # L_Hip             thigh_l
            #     2 : "RightUpperLeg", # R_Hip            thigh_r
            #     3 : "Spine", # Spine1                   spine_01
            #     4 : "LeftLowerLeg", # L_Knee            calf_l
            #     5 : "RightLowerLeg", # R_Knee           calf_r
            #     6 : "Chest", # Spine2                   spine_02
            #     7 : "LeftFoot", # L_Ankle               foot_l
            #     8 : "RightFoot", # R_Ankle              foot_r
            #     9 : "UpperChest", # Spine3              spine_03
            #     10 : "LeftToes", # L_Foot               ball_l
            #     11 : "RightToes", # R_Foot              ball_r
            #     12 : "Neck", # Neck                     neck_01
            #     13 : "LeftShoulder", # L_Collar         clavicle_l
            #     14 : "RightShoulder", # R_Collar        clavicle_r
            #     15 : "Head", # Head                     head
            #     16 : "LeftUpperArm", # L_Shoulder       upperarm_l
            #     17 : "RightUpperArm", # R_Shoulder      upperarm_r
            #     18 : "LeftLowerArm", # L_Elbow          lowerarm_l
            #     19 : "RightLowerArm", # R_Elbow         lowerarm_r
            #     20 : "LeftHand", # L_Wrist              hand_l
            #     21 : "RightHand", # R_Wrist             hand_r
            #     22 : "LeftMiddleProximal", # L_Hand     hand_l
            #     23 : "RightMiddleProximal" # R_hand     hand_r
            # }

            smpl_rotations_by_axis = art.math.quaternion_to_axis_angle(pose_q)
            smpl_root_position = np.array([0, 0, 0])

            bones = []
            for index, rot in enumerate(smpl_rotations_by_axis):
                bone_name = vrm_bone_names[index]
                rot = R.from_rotvec(rot).as_quat()
                rotation = Quaternion(-rot[0], rot[1], rot[2], rot[3])
                bones.append(
                    [
                        Bone(bone_name),
                        Position.identity(),
                        rotation.conjugate()
                    ]
                )

            # Sending
            vmc.send_root_transform(
                # Position(
                #     -smpl_root_position[0] - hips_bone_position.x + root_position_offset.x,
                #     -smpl_root_position[1] - hips_bone_position.y + root_position_offset.y,
                #     -smpl_root_position[2] - hips_bone_position.z + root_position_offset.z
                # ),
                Position(
                    -smpl_root_position[0] - hips_bone_position.x + root_position_offset.x,
                    smpl_root_position[1] - hips_bone_position.y + root_position_offset.y,
                    smpl_root_position[2] - hips_bone_position.z + root_position_offset.z
                ),
                # Position(
                #     -smpl_root_position[0],
                #     smpl_root_position[1],
                #     smpl_root_position[2]
                # ),
                # Quaternion.identity().multiply_by(Quaternion.from_euler(-180, 0, 0, 12), 12)
                Quaternion.identity().multiply_by(Quaternion.from_euler(0, 0, 0, 12), 12)
            )

            vmc.send_bones_transform(bones)
            vmc.send_available_states(1)
            # delta = started_at.delta(configuration["delta"])
            # vmc.send_relative_time(delta)
            # configuration["delta"] = delta
            delta = started_at.delta(time.time() - start)
            vmc.send_relative_time(delta)
            configuration["delta"] = delta
            # print(delta)


            # print("pelvis = ", pelvis)
            # print("spine_01 = ", spine_01)
            time.sleep(sleep_time)

        print('Shut down')
        return

def send_value(event):
    global char_data
    mapping= {0:0, 1:90, 2:180, 3:270}

    char_data['w_head_X'] = int(mapping.get(w_head_X.get()))
    char_data['w_head_Y'] = int(mapping.get(w_head_Y.get()))
    char_data['w_head_Z'] = int(mapping.get(w_head_Z.get()))

    char_data['w_neck_01_X'] = int(mapping.get(w_neck_01_X.get()))
    char_data['w_neck_01_Y'] = int(mapping.get(w_neck_01_Y.get()))
    char_data['w_neck_01_Z'] = int(mapping.get(w_neck_01_Z.get()))

    char_data['w_spine_03_X'] = int(mapping.get(w_spine_03_X.get()))
    char_data['w_spine_03_Y'] = int(mapping.get(w_spine_03_Y.get()))
    char_data['w_spine_03_Z'] = int(mapping.get(w_spine_03_Z.get()))

    char_data['w_spine_02_X'] = int(mapping.get(w_spine_02_X.get()))
    char_data['w_spine_02_Y'] = int(mapping.get(w_spine_02_Y.get()))
    char_data['w_spine_02_Z'] = int(mapping.get(w_spine_02_Z.get()))

    char_data['w_spine_01_X'] = int(mapping.get(w_spine_01_X.get()))
    char_data['w_spine_01_Y'] = int(mapping.get(w_spine_01_Y.get()))
    char_data['w_spine_01_Z'] = int(mapping.get(w_spine_01_Z.get()))

    char_data['w_pelvis_X'] = int(mapping.get(w_pelvis_X.get()))
    char_data['w_pelvis_Y'] = int(mapping.get(w_pelvis_Y.get()))
    char_data['w_pelvis_Z'] = int(mapping.get(w_pelvis_Z.get()))

    #------------------
    char_data['w_clavicle_r_X'] = int(mapping.get(w_clavicle_r_X.get()))
    char_data['w_clavicle_r_Y'] = int(mapping.get(w_clavicle_r_Y.get()))
    char_data['w_clavicle_r_Z'] = int(mapping.get(w_clavicle_r_Z.get()))

    char_data['w_upperarm_r_X'] = int(mapping.get(w_upperarm_r_X.get()))
    char_data['w_upperarm_r_Y'] = int(mapping.get(w_upperarm_r_Y.get()))
    char_data['w_upperarm_r_Z'] = int(mapping.get(w_upperarm_r_Z.get()))

    char_data['w_lowerarm_r_X'] = int(mapping.get(w_lowerarm_r_X.get()))
    char_data['w_lowerarm_r_Y'] = int(mapping.get(w_lowerarm_r_Y.get()))
    char_data['w_lowerarm_r_Z'] = int(mapping.get(w_lowerarm_r_Z.get()))

    char_data['w_hand_r_X'] = int(mapping.get(w_hand_r_X.get()))
    char_data['w_hand_r_Y'] = int(mapping.get(w_hand_r_Y.get()))
    char_data['w_hand_r_Z'] = int(mapping.get(w_hand_r_Z.get()))

    #------------------
    char_data['w_clavicle_l_X'] = int(mapping.get(w_clavicle_l_X.get()))
    char_data['w_clavicle_l_Y'] = int(mapping.get(w_clavicle_l_Y.get()))
    char_data['w_clavicle_l_Z'] = int(mapping.get(w_clavicle_l_Z.get()))

    char_data['w_upperarm_l_X'] = int(mapping.get(w_upperarm_l_X.get()))
    char_data['w_upperarm_l_Y'] = int(mapping.get(w_upperarm_l_Y.get()))
    char_data['w_upperarm_l_Z'] = int(mapping.get(w_upperarm_l_Z.get()))

    char_data['w_lowerarm_l_X'] = int(mapping.get(w_lowerarm_l_X.get()))
    char_data['w_lowerarm_l_Y'] = int(mapping.get(w_lowerarm_l_Y.get()))
    char_data['w_lowerarm_l_Z'] = int(mapping.get(w_lowerarm_l_Z.get()))

    char_data['w_hand_l_X'] = int(mapping.get(w_hand_l_X.get()))
    char_data['w_hand_l_Y'] = int(mapping.get(w_hand_l_Y.get()))
    char_data['w_hand_l_Z'] = int(mapping.get(w_hand_l_Z.get()))

    #------------------
    char_data['w_thigh_r_X'] = int(mapping.get(w_thigh_r_X.get()))
    char_data['w_thigh_r_Y'] = int(mapping.get(w_thigh_r_Y.get()))
    char_data['w_thigh_r_Z'] = int(mapping.get(w_thigh_r_Z.get()))

    char_data['w_calf_r_X'] = int(mapping.get(w_calf_r_X.get()))
    char_data['w_calf_r_Y'] = int(mapping.get(w_calf_r_Y.get()))
    char_data['w_calf_r_Z'] = int(mapping.get(w_calf_r_Z.get()))

    char_data['w_foot_r_X'] = int(mapping.get(w_foot_r_X.get()))
    char_data['w_foot_r_Y'] = int(mapping.get(w_foot_r_Y.get()))
    char_data['w_foot_r_Z'] = int(mapping.get(w_foot_r_Z.get()))

    char_data['w_ball_r_X'] = int(mapping.get(w_ball_r_X.get()))
    char_data['w_ball_r_Y'] = int(mapping.get(w_ball_r_Y.get()))
    char_data['w_ball_r_Z'] = int(mapping.get(w_ball_r_Z.get()))

    #------------------
    char_data['w_thigh_l_X'] = int(mapping.get(w_thigh_l_X.get()))
    char_data['w_thigh_l_Y'] = int(mapping.get(w_thigh_l_Y.get()))
    char_data['w_thigh_l_Z'] = int(mapping.get(w_thigh_l_Z.get()))

    char_data['w_calf_l_X'] = int(mapping.get(w_calf_l_X.get()))
    char_data['w_calf_l_Y'] = int(mapping.get(w_calf_l_Y.get()))
    char_data['w_calf_l_Z'] = int(mapping.get(w_calf_l_Z.get()))

    char_data['w_foot_l_X'] = int(mapping.get(w_foot_l_X.get()))
    char_data['w_foot_l_Y'] = int(mapping.get(w_foot_l_Y.get()))
    char_data['w_foot_l_Z'] = int(mapping.get(w_foot_l_Z.get()))

    char_data['w_ball_l_X'] = int(mapping.get(w_ball_l_X.get()))
    char_data['w_ball_l_Y'] = int(mapping.get(w_ball_l_Y.get()))
    char_data['w_ball_l_Z'] = int(mapping.get(w_ball_l_Z.get()))

class obj:
    def __init__(self, name, i, j):
        self.w_X = tk.Scale(gui,
                                 label= str(name) + '_X',
                                 orient=tk.HORIZONTAL,
                                 from_=0,
                                 to=3,
                                 # tickinterval=1,
                                 command=send_value)
        self.w_X.set(0)
        self.w_X.grid(row=i,column=j)

        self.w_Y = tk.Scale(gui,
                                 label= str(name) + '_Y',
                                 orient=tk.HORIZONTAL,
                                 from_=0,
                                 to=3,
                                 # tickinterval=1,
                                 command=send_value)
        self.w_Y.set(0)
        self.w_Y.grid(row=i,column=j+1)

        self.w_Z = tk.Scale(gui,
                                 label= str(name) + '_Z',
                                 orient=tk.HORIZONTAL,
                                 from_=0,
                                 to=3,
                                 # tickinterval=1,
                                 command=send_value)
        self.w_Z.set(0)
        self.w_Z.grid(row=i,column=j+2)

head = obj('head', 0, 2)
w_head_X = head.w_X
w_head_Y = head.w_Y
w_head_Z = head.w_Z

neck_01 = obj('neck_01', 1, 2)
w_neck_01_X = neck_01.w_X
w_neck_01_Y = neck_01.w_Y
w_neck_01_Z = neck_01.w_Z

#-------------------------------
clavicle_r = obj('clavicle_r', 2, 0)
w_clavicle_r_X = clavicle_r.w_X
w_clavicle_r_Y = clavicle_r.w_Y
w_clavicle_r_Z = clavicle_r.w_Z

upperarm_r = obj('upperarm_r', 3, 0)
w_upperarm_r_X = upperarm_r.w_X
w_upperarm_r_Y = upperarm_r.w_Y
w_upperarm_r_Z = upperarm_r.w_Z

lowerarm_r = obj('lowerarm_r', 4, 0)
w_lowerarm_r_X = lowerarm_r.w_X
w_lowerarm_r_Y = lowerarm_r.w_Y
w_lowerarm_r_Z = lowerarm_r.w_Z

hand_r = obj('hand_r', 5, 0)
w_hand_r_X = hand_r.w_X
w_hand_r_Y = hand_r.w_Y
w_hand_r_Z = hand_r.w_Z

# head = obj('head', 0, 2)
# w_head_X = head.w_X
# w_head_Y = head.w_Y
# w_head_Z = head.w_Z

#-------------------------------
clavicle_l = obj('clavicle_l', 2, 4)
w_clavicle_l_X = clavicle_l.w_X
w_clavicle_l_Y = clavicle_l.w_Y
w_clavicle_l_Z = clavicle_l.w_Z

upperarm_l = obj('upperarm_l', 3, 4)
w_upperarm_l_X = upperarm_l.w_X
w_upperarm_l_Y = upperarm_l.w_Y
w_upperarm_l_Z = upperarm_l.w_Z

lowerarm_l = obj('lowerarm_l', 4, 4)
w_lowerarm_l_X = lowerarm_l.w_X
w_lowerarm_l_Y = lowerarm_l.w_Y
w_lowerarm_l_Z = lowerarm_l.w_Z

hand_l = obj('hand_l', 5, 4)
w_hand_l_X = hand_l.w_X
w_hand_l_Y = hand_l.w_Y
w_hand_l_Z = hand_l.w_Z

#-------------------------------
spine_03 = obj('spine_03', 7, 2)
w_spine_03_X = spine_03.w_X
w_spine_03_Y = spine_03.w_Y
w_spine_03_Z = spine_03.w_Z

spine_02 = obj('spine_02', 8, 2)
w_spine_02_X = spine_02.w_X
w_spine_02_Y = spine_02.w_Y
w_spine_02_Z = spine_02.w_Z

spine_01 = obj('spine_01', 9, 2)
w_spine_01_X = spine_01.w_X
w_spine_01_Y = spine_01.w_Y
w_spine_01_Z = spine_01.w_Z

pelvis = obj('pelvis', 10, 2)
w_pelvis_X = pelvis.w_X
w_pelvis_Y = pelvis.w_Y
w_pelvis_Z = pelvis.w_Z

#-------------------------------
thigh_r = obj('thigh_r', 11, 0)
w_thigh_r_X = thigh_r.w_X
w_thigh_r_Y = thigh_r.w_Y
w_thigh_r_Z = thigh_r.w_Z

calf_r = obj('calf_r', 12, 0)
w_calf_r_X = calf_r.w_X
w_calf_r_Y = calf_r.w_Y
w_calf_r_Z = calf_r.w_Z

foot_r = obj('foot_r', 13, 0)
w_foot_r_X = foot_r.w_X
w_foot_r_Y = foot_r.w_Y
w_foot_r_Z = foot_r.w_Z

ball_r = obj('ball_r', 14, 0)
w_ball_r_X = ball_r.w_X
w_ball_r_Y = ball_r.w_Y
w_ball_r_Z = ball_r.w_Z

#-------------------------------
thigh_l = obj('thigh_l', 11, 4)
w_thigh_l_X = thigh_l.w_X
w_thigh_l_Y = thigh_l.w_Y
w_thigh_l_Z = thigh_l.w_Z

calf_l = obj('calf_l', 12, 4)
w_calf_l_X = calf_l.w_X
w_calf_l_Y = calf_l.w_Y
w_calf_l_Z = calf_l.w_Z

foot_l = obj('foot_l', 13, 4)
w_foot_l_X = foot_l.w_X
w_foot_l_Y = foot_l.w_Y
w_foot_l_Z = foot_l.w_Z

ball_l = obj('ball_l', 14, 4)
w_ball_l_X = ball_l.w_X
w_ball_l_Y = ball_l.w_Y
w_ball_l_Z = ball_l.w_Z

#---------------------------------------------------------------------
print('Server Listening')

zmq_rep_task = ZMQ_REP_Task()
# zmq_rep_task.run()
zmq_thread = threading.Thread(target=zmq_rep_task.run, args=(10,))
zmq_thread.start()

tk.mainloop()
