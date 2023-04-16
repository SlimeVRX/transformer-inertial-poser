from scipy.spatial.transform import Rotation
import numpy as np
import pybullet as pb
import pybullet_data
import collections
import math
import warnings

# connect to server pybullet
pb_client = pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
pb.resetSimulation()


scale = 1.0
char_create_flags = pb.URDF_MAINTAIN_LINK_ORDER
if 1:
    char_create_flags = char_create_flags | \
                        pb.URDF_USE_SELF_COLLISION | \
                        pb.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
# load 3D model
robot = pb.loadURDF("./data/amass.urdf",
            [0, 0, 0],
            globalScaling=scale,
            useFixedBase=False,
            flags=char_create_flags)

class conversions:
    @staticmethod
    def batch_auto_reshape(x, fn, shape_in, shape_out):
        reshape = x.ndim - len(shape_in) > 1
        xx = x.reshape(-1, *shape_in) if reshape else x
        y = fn(xx)
        return y.reshape(x.shape[: -len(shape_in)] + shape_out) if reshape else y

    @staticmethod
    def T2Rp(T):
        R = T[..., :3, :3]
        p = T[..., :3, 3]
        return R, p

    @staticmethod
    def R2Q(R):
        return conversions.batch_auto_reshape(
            R, lambda x: Rotation.from_matrix(x).as_quat(), (3, 3), (4,),
        )

    @staticmethod
    def T2Qp(T):
        R, p = conversions.T2Rp(T)
        Q = conversions.R2Q(R)
        return Q, p

    @staticmethod
    def Q2R(Q):
        return conversions.batch_auto_reshape(
            Q, lambda x: Rotation.from_quat(x).as_matrix(), (4,), (3, 3),
        )

    @staticmethod
    def Rp2T(R, p):
        input_shape = R.shape[:-2] if R.ndim > 2 else p.shape[:-1]
        R_flat = R.reshape((-1, 3, 3))
        p_flat = p.reshape((-1, 3))
        T = np.zeros((int(np.prod(input_shape)), 4, 4))
        T[...] = constants.eye_T()
        T[..., :3, :3] = R_flat
        T[..., :3, 3] = p_flat
        return T.reshape(list(input_shape) + [4, 4])

    @staticmethod
    def Qp2T(Q, p):
        R = conversions.Q2R(Q)
        return conversions.Rp2T(R, p)

class utils:
    @staticmethod
    def _apply_fn_agnostic_to_vec_mat(input, fn):
        output = np.array([input]) if input.ndim == 1 else input
        output = np.apply_along_axis(fn, 1, output)
        return output[0] if input.ndim == 1 else output

class constants:
    EYE_T = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        float,
    )
    ZERO_P = np.array([0.0, 0.0, 0.0], float)
    EYE_R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], float)

    EPSILON = np.finfo(float).eps

    @staticmethod
    def eye_T():
        return constants.EYE_T.copy()

    @staticmethod
    def eye_R():
        return constants.EYE_R.copy()

    @staticmethod
    def zero_p():
        return constants.ZERO_P.copy()

class quaternion:
    @staticmethod
    def Q_op(Q, op, xyzw_in=True):
        def q2q(q):
            result = q.copy()
            if "normalize" in op:
                norm = np.linalg.norm(result)
                if norm < constants.EPSILON:
                    raise Exception("Invalid input with zero length")
                result /= norm
            if "halfspace" in op:
                w_idx = 3 if xyzw_in else 0
                if result[w_idx] < 0.0:
                    result *= -1.0
            if "change_order" in op:
                result = result[[3, 0, 1, 2]] if xyzw_in else result[[1, 2, 3, 0]]
            return result

        return utils._apply_fn_agnostic_to_vec_mat(Q, q2q)

    @staticmethod
    def Q_closest(Q1, Q2, axis):
        """
        This computes optimal-in-place orientation given a target orientation Q1
        and a geodesic curve (Q2, axis). In tutively speaking, the optimal-in-place
        orientation is the closest orientation to Q1 when we are able to rotate Q2
        along the given axis. We assume Q is given in the order of xyzw.
        """
        ws, vs = Q1[3], Q1[0:3]
        w0, v0 = Q2[3], Q2[0:3]
        u = f_math.normalize(axis)

        a = ws * w0 + np.dot(vs, v0)
        b = -ws * np.dot(u, v0) + w0 * np.dot(vs, u) + np.dot(vs, np.cross(u, v0))
        alpha = math.atan2(a, b)

        theta1 = -2 * alpha + math.pi
        theta2 = -2 * alpha - math.pi
        G1 = conversions.A2Q(theta1 * u)
        G2 = conversions.A2Q(theta2 * u)

        if np.dot(Q1, G1) > np.dot(Q1, G2):
            theta = theta1
            Qnearest = quaternion.Q_mult(G1, Q2)
        else:
            theta = theta2
            Qnearest = quaternion.Q_mult(G1, Q2)

        return Qnearest, theta

    @staticmethod
    def Q_mult(Q1, Q2):
        """
        Multiply two quaternions.
        Q1, Q2 array like (N,4) or (4,)
        """
        ax, ay, az, aw = Q1.T       # .T no effect when 1D
        bx, by, bz, bw = Q2.T
        ow = aw * bw - ax * bx - ay * by - az * bz
        ox = aw * bx + ax * bw + ay * bz - az * by
        oy = aw * by - ax * bz + ay * bw + az * bx
        oz = aw * bz + ax * by - ay * bx + az * bw
        return np.array([ox, oy, oz, ow]).T

class f_math:
    @staticmethod
    def invertT(T):
        R = T[:3, :3]
        p = T[:3, 3]
        invT = constants.eye_T()
        R_trans = R.transpose()
        R_trans_p = np.dot(R_trans, p)
        invT[:3, :3] = R_trans
        invT[:3, 3] = -R_trans_p
        return invT

    @staticmethod
    def slerp(R1, R2, t):
        return np.dot(
            R1, conversions.A2R(t * conversions.R2A(np.dot(R1.transpose(), R2)))
        )

    @staticmethod
    def lerp(v0, v1, t):
        return v0 + (v1 - v0) * t

    @staticmethod
    def project_rotation_1D(R, axis):
        Q, angle = quaternion.Q_closest(
            conversions.R2Q(R), [1.0, 0.0, 0.0, 0.0], axis,
        )
        return angle

    @staticmethod
    def normalize(v):
        is_list = type(v) == list
        length = np.linalg.norm(v)
        if length > constants.EPSILON:
            norm_v = np.array(v) / length
            if is_list:
                return list(norm_v)
            else:
                return norm_v
        else:
            warnings.warn("!!!The length of input vector is almost zero!!!")
            return v

    @staticmethod
    def project_angular_vel_1D(w, axis):
        return np.linalg.norm(np.dot(w, axis))

''' 
Definition of Link/Joint (In our character definition, one joint can only have one link)
'''
root = -1
lhip = 0
lknee = 1
lankle = 2
rhip = 3
rknee = 4
rankle = 5
lowerback = 6
upperback = 7
chest = 8
lowerneck = 9
upperneck = 10
lclavicle = 11
lshoulder = 12
lelbow = 13
lwrist = 14
rclavicle = 15
rshoulder = 16
relbow = 17
rwrist = 18

''' 
Mapping from character's joint indicies to bvh's joint names.
Some entry could have no mapping (by assigning None).
'''

bvh_map = collections.OrderedDict()

bvh_map[root] = "root"
bvh_map[lhip] = "lhip"
bvh_map[lknee] = "lknee"
bvh_map[lankle] = "lankle"
bvh_map[rhip] = "rhip"
bvh_map[rknee] = "rknee"
bvh_map[rankle] = "rankle"
bvh_map[lowerback] = "lowerback"
bvh_map[upperback] = "upperback"
bvh_map[chest] = "chest"
bvh_map[lowerneck] = "lowerneck"
bvh_map[upperneck] = "upperneck"
bvh_map[lclavicle] = "lclavicle"
bvh_map[lshoulder] = "lshoulder"
bvh_map[lelbow] = "lelbow"
bvh_map[lwrist] = "lwrist"
bvh_map[rclavicle] = "rclavicle"
bvh_map[rshoulder] = "rshoulder"
bvh_map[relbow] = "relbow"
bvh_map[rwrist] = "rwrist"

with open(('pose_smpl.npy'), 'rb') as f:
    pose_smpl = np.load(f)

key = 'root'

xform_from_parent_joint = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])



rot_up_Q = np.array([0.5, 0.5, 0.5, 0.5])
rot_up_R = conversions.Q2R(rot_up_Q)
root_z_offset = 0.95
T0 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])









import time
import threading
import numpy as np
import tkinter as tk

cos, sin = np.cos, np.sin
# char_data = {'w_head_X': 0, 'w_head_Y': 0, 'w_head_Z': 0, 'w_neck_01_X': 0, 'w_neck_01_Y': 0, 'w_neck_01_Z': 0}
char_data = {'w_head_X': 0, 'w_head_Y': 0, 'w_head_Z': 0, 'w_neck_01_X': 0, 'w_neck_01_Y': 0, 'w_neck_01_Z': 0, 'w_clavicle_r_X': 0, 'w_clavicle_r_Y': 0, 'w_clavicle_r_Z': 0, 'w_upperarm_r_X': 0, 'w_upperarm_r_Y': 0, 'w_upperarm_r_Z': 0, 'w_lowerarm_r_X': 0, 'w_lowerarm_r_Y': 0, 'w_lowerarm_r_Z': 0, 'w_hand_r_X': 0, 'w_hand_r_Y': 0, 'w_hand_r_Z': 0,
             'w_clavicle_l_X': 0, 'w_clavicle_l_Y': 0, 'w_clavicle_l_Z': 0, 'w_upperarm_l_X': 0, 'w_upperarm_l_Y': 0, 'w_upperarm_l_Z': 0, 'w_lowerarm_l_X': 0, 'w_lowerarm_l_Y': 0, 'w_lowerarm_l_Z': 0, 'w_hand_l_X': 0, 'w_hand_l_Y': 0, 'w_hand_l_Z': 0,
             'w_spine_03_X': 0, 'w_spine_03_Y': 0, 'w_spine_03_Z': 0, 'w_spine_02_X': 0, 'w_spine_02_Y': 0, 'w_spine_02_Z': 0, 'w_spine_01_X': 0, 'w_spine_01_Y': 0, 'w_spine_01_Z': 0, 'w_pelvis_X': 0, 'w_pelvis_Y': 0, 'w_pelvis_Z': 0,
             'w_thigh_r_X': 0, 'w_thigh_r_Y': 0, 'w_thigh_r_Z': 0, 'w_calf_r_X': 0, 'w_calf_r_Y': 0, 'w_calf_r_Z': 0, 'w_foot_r_X': 0, 'w_foot_r_Y': 0, 'w_foot_r_Z': 0, 'w_ball_r_X': 0, 'w_ball_r_Y': 0, 'w_ball_r_Z': 0,
             'w_thigh_l_X': 0, 'w_thigh_l_Y': 0, 'w_thigh_l_Z': 0, 'w_calf_l_X': 0, 'w_calf_l_Y': 0, 'w_calf_l_Z': 0, 'w_foot_l_X': 0, 'w_foot_l_Y': 0, 'w_foot_l_Z': 0, 'w_ball_l_X': 0, 'w_ball_l_Y': 0, 'w_ball_l_Z': 0
             }

# char_keys = {'w_head_X': None, 'w_head_Y': None, 'w_head_Z': None, 'w_neck_01_X': None, 'w_neck_01_Y': None, 'w_neck_01_Z': None}
char_keys = {'w_head_X': None, 'w_head_Y': None, 'w_head_Z': None, 'w_neck_01_X': None, 'w_neck_01_Y': None, 'w_neck_01_Z': None, 'w_clavicle_r_X': None, 'w_clavicle_r_Y': None, 'w_clavicle_r_Z': None, 'w_upperarm_r_X': None, 'w_upperarm_r_Y': None, 'w_upperarm_r_Z': None, 'w_lowerarm_r_X': None, 'w_lowerarm_r_Y': None, 'w_lowerarm_r_Z': None, 'w_hand_r_X': None, 'w_hand_r_Y': None, 'w_hand_r_Z': None,
            'w_clavicle_l_X': None, 'w_clavicle_l_Y': None, 'w_clavicle_l_Z': None, 'w_upperarm_l_X': None, 'w_upperarm_l_Y': None, 'w_upperarm_l_Z': None, 'w_lowerarm_l_X': None, 'w_lowerarm_l_Y': None, 'w_lowerarm_l_Z': None, 'w_hand_l_X': None, 'w_hand_l_Y': None, 'w_hand_l_Z': None,
            'w_spine_03_X': None, 'w_spine_03_Y': None, 'w_spine_03_Z': None, 'w_spine_02_X': None, 'w_spine_02_Y': None, 'w_spine_02_Z': None, 'w_spine_01_X': None, 'w_spine_01_Y': None, 'w_spine_01_Z': None, 'w_pelvis_X': None, 'w_pelvis_Y': None, 'w_pelvis_Z': None,
            'w_thigh_r_X': None, 'w_thigh_r_Y': None, 'w_thigh_r_Z': None, 'w_calf_r_X': None, 'w_calf_r_Y': None, 'w_calf_r_Z': None, 'w_foot_r_X': None, 'w_foot_r_Y': None, 'w_foot_r_Z': None, 'w_ball_r_X': None, 'w_ball_r_Y': None, 'w_ball_r_Z': None,
            'w_thigh_l_X': None, 'w_thigh_l_Y': None, 'w_thigh_l_Z': None, 'w_calf_l_X': None, 'w_calf_l_Y': None, 'w_calf_l_Z': None, 'w_foot_l_X': None, 'w_foot_l_Y': None, 'w_foot_l_Z': None, 'w_ball_l_X': None, 'w_ball_l_Y': None, 'w_ball_l_Z': None
             }

def ch_rot(axis, theta):
    if axis == 'x': a, b, c, d = 1, 0, 0, [0, cos(theta), -sin(theta), 0, sin(theta), cos(theta)]
    elif axis == 'y': a, b, c, d = cos(theta), 0, sin(theta), [0, 1, 0, -sin(theta), 0, cos(theta)]
    else: a, b, c, d = cos(theta), -sin(theta), 0, [sin(theta), cos(theta), 0, 0, 0, 1]
    return np.array([[a, b, c], d[:3], d[3:]])

def ch_m2q(C):
    Q = [0.5 * np.sqrt(max(0, 1 + sum(C[i][i] for i in range(3)) - 2 * C[j][j])) for j in range(3)]
    Q = [q if C[(j + 1) % 3][(j + 2) % 3] > C[(j + 2) % 3][(j + 1) % 3] else -q for j, q in enumerate(Q)]
    Q.append(0.5 * np.sqrt(max(0, 1 + sum(C[i][i] for i in range(3)))))
    return np.array(Q)

gui = tk.Tk()
gui.geometry('960x960')

class ZMQ_REP_Task:
    def __init__(self): self._running = True

    def run(self, args):
        global char_data
        fps, sleep_time = 60, 1 / 60

        cur_time = 0.0 / 2.0
        imu_idx = 0
        DT = 1. / 60

        while self._running:
            eul_rad = {k: np.deg2rad(v) for k, v in char_data.items()}

            head = ch_rot('z', eul_rad['w_head_Z']).dot(ch_rot('x', eul_rad['w_head_X'])).dot(ch_rot('y', eul_rad['w_head_Y']))
            neck_01 = ch_rot('z', eul_rad['w_neck_01_Z']).dot(ch_rot('x', eul_rad['w_neck_01_X'])).dot(ch_rot('y', eul_rad['w_neck_01_Y']))
            spine_03 = ch_rot('z', eul_rad['w_spine_03_Z']).dot(ch_rot('x', eul_rad['w_spine_03_X'])).dot(ch_rot('y', eul_rad['w_spine_03_Y']))
            spine_02 = ch_rot('z', eul_rad['w_spine_02_Z']).dot(ch_rot('x', eul_rad['w_spine_02_X'])).dot(ch_rot('y', eul_rad['w_spine_02_Y']))
            spine_01 = ch_rot('z', eul_rad['w_spine_01_Z']).dot(ch_rot('x', eul_rad['w_spine_01_X'])).dot(ch_rot('y', eul_rad['w_spine_01_Y']))
            pelvis = ch_rot('z', eul_rad['w_pelvis_Z']).dot(ch_rot('x', eul_rad['w_pelvis_X'])).dot(ch_rot('y', eul_rad['w_pelvis_Y']))
            clavicle_r = ch_rot('z', eul_rad['w_clavicle_r_Z']).dot(ch_rot('x', eul_rad['w_clavicle_r_X'])).dot(ch_rot('y', eul_rad['w_clavicle_r_Y']))
            upperarm_r = ch_rot('z', eul_rad['w_upperarm_r_Z']).dot(ch_rot('x', eul_rad['w_upperarm_r_X'])).dot(ch_rot('y', eul_rad['w_upperarm_r_Y']))
            lowerarm_r = ch_rot('z', eul_rad['w_lowerarm_r_Z']).dot(ch_rot('x', eul_rad['w_lowerarm_r_X'])).dot(ch_rot('y', eul_rad['w_lowerarm_r_Y']))
            hand_r = ch_rot('z', eul_rad['w_hand_r_Z']).dot(ch_rot('x', eul_rad['w_hand_r_X'])).dot(ch_rot('y', eul_rad['w_hand_r_Y']))
            clavicle_l = ch_rot('z', eul_rad['w_clavicle_l_Z']).dot(ch_rot('x', eul_rad['w_clavicle_l_X'])).dot(ch_rot('y', eul_rad['w_clavicle_l_Y']))
            upperarm_l = ch_rot('z', eul_rad['w_upperarm_l_Z']).dot(ch_rot('x', eul_rad['w_upperarm_l_X'])).dot(ch_rot('y', eul_rad['w_upperarm_l_Y']))
            lowerarm_l = ch_rot('z', eul_rad['w_lowerarm_l_Z']).dot(ch_rot('x', eul_rad['w_lowerarm_l_X'])).dot(ch_rot('y', eul_rad['w_lowerarm_l_Y']))
            hand_l = ch_rot('z', eul_rad['w_hand_l_Z']).dot(ch_rot('x', eul_rad['w_hand_l_X'])).dot(ch_rot('y', eul_rad['w_hand_l_Y']))
            thigh_r = ch_rot('z', eul_rad['w_thigh_r_Z']).dot(ch_rot('x', eul_rad['w_thigh_r_X'])).dot(ch_rot('y', eul_rad['w_thigh_r_Y']))
            calf_r = ch_rot('z', eul_rad['w_calf_r_Z']).dot(ch_rot('x', eul_rad['w_calf_r_X'])).dot(ch_rot('y', eul_rad['w_calf_r_Y']))
            foot_r = ch_rot('z', eul_rad['w_foot_r_Z']).dot(ch_rot('x', eul_rad['w_foot_r_X'])).dot(ch_rot('y', eul_rad['w_foot_r_Y']))
            ball_r = ch_rot('z', eul_rad['w_ball_r_Z']).dot(ch_rot('x', eul_rad['w_ball_r_X'])).dot(ch_rot('y', eul_rad['w_ball_r_Y']))
            thigh_l = ch_rot('z', eul_rad['w_thigh_l_Z']).dot(ch_rot('x', eul_rad['w_thigh_l_X'])).dot(ch_rot('y', eul_rad['w_thigh_l_Y']))
            calf_l = ch_rot('z', eul_rad['w_calf_l_Z']).dot(ch_rot('x', eul_rad['w_calf_l_X'])).dot(ch_rot('y', eul_rad['w_calf_l_Y']))
            foot_l = ch_rot('z', eul_rad['w_foot_l_Z']).dot(ch_rot('x', eul_rad['w_foot_l_X'])).dot(ch_rot('y', eul_rad['w_foot_l_Y']))
            ball_l = ch_rot('z', eul_rad['w_ball_l_Z']).dot(ch_rot('x', eul_rad['w_ball_l_X'])).dot(ch_rot('y', eul_rad['w_ball_l_Y']))

            # list = [head, neck_01, spine_03, spine_02, spine_01, pelvis, clavicle_r, upperarm_r, lowerarm_r, hand_r, clavicle_l, upperarm_l, lowerarm_l, hand_l, thigh_r, calf_r, foot_r, ball_r, thigh_l, calf_l, foot_l, ball_l]
            list = [pelvis, thigh_l, thigh_r, spine_01, calf_l, calf_r, spine_02, foot_l, foot_r, spine_03, ball_l, ball_r, neck_01, clavicle_l, clavicle_r, head, upperarm_l, upperarm_r, lowerarm_l, lowerarm_r, hand_l, hand_r, hand_l, hand_r]
            print("spine = ", spine_01, spine_02, spine_03)
            pose_smpl = np.array([np.eye(4) for _ in range(len(list))])
            for i, rotation in enumerate(list):
                pose_smpl[i][:3, :3] = np.array(rotation)
            pose_smpl = np.array(pose_smpl)

            get_index_joint = pose_smpl[0]

            # xoay he truc toa do 90 do va di chuyen Z len 9.5
            #--------------------------
            belly_R = np.dot(
                xform_from_parent_joint,
                get_index_joint
            )[:3, :3]
            root_R = rot_up_R.dot(belly_R)
            p = np.array([0, 0, root_z_offset])

            T_ = conversions.Rp2T(root_R, p)
            T1 = np.dot(f_math.invertT(T0), T_)

            Q_, p_ = conversions.T2Qp(T1)
            Q_ = quaternion.Q_op(Q_, op=["normalize"])
            T1_ = conversions.Qp2T(Q_, p_)

            pose_smpl[0] = T1_
            #--------------------------

            T = np.dot(
                xform_from_parent_joint,
                get_index_joint
            )
            # print(T)

            Q, p = conversions.T2Qp(T)

            # bullet_utils.set_base_pQvw(self._pb_client, self._body_id, p, Q, v, w)
            # def set_base_pQvw(pb_client, body_id, p, Q, v=None, w=None):

            pb.resetBasePositionAndOrientation(robot, p, Q)

            _joint_type = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4]
            JOINT_FIXED = 4
            JOINT_SPHERICAL = 2
            JOINT_REVOLUTE = 0

            index_joint = {'root': 0, 'lhip': 1, 'rhip': 2, 'lowerback': 3, 'lknee': 4, 'rknee': 5, 'upperback': 6, 'lankle': 7, 'rankle': 8, 'chest': 9, 'ltoe': 10, 'rtoe': 11, 'lowerneck': 12, 'lclavicle': 13, 'rclavicle': 14, 'upperneck': 15, 'lshoulder': 16, 'rshoulder': 17, 'lelbow': 18, 'relbow': 19, 'lwrist': 20, 'rwrist': 21, 'lhand': 22, 'rhand': 23}
            _joint_axis = [np.zeros(3)] * 19

            indices = []
            state_pos = []
            state_vel = []
            for j in range(0,19):
                joint_type = _joint_type[j]
                if joint_type == JOINT_FIXED:
                    continue
                T = pose_smpl[index_joint[bvh_map[j]]]
                if joint_type == JOINT_SPHERICAL:
                    Q, p = conversions.T2Qp(T)
                    w = np.zeros(3)
                    state_pos.append(Q)
                    state_vel.append(w)
                elif joint_type == JOINT_REVOLUTE:
                    joint_axis = _joint_axis[j]
                    R, p = conversions.T2Rp(T)
                    w = np.zeros(3)
                    state_pos.append([f_math.project_rotation_1D(R, joint_axis)])
                    state_vel.append([f_math.project_angular_vel_1D(w, joint_axis)])
                else:
                    raise NotImplementedError()
                indices.append(j)

            # bullet_utils.set_joint_pv(self._pb_client, self._body_id, indices, state_pos, state_vel)
            # print("state_pos = ", state_pos)
            pb.resetJointStatesMultiDof(robot, indices, state_pos, state_vel)



            cur_time += DT
            imu_idx += 1
            time.sleep(1.)

            # time.sleep(sleep_time)

        print('Shut down')
        return

def send_value(event):
    global char_data
    mapping = {0: 0, 1: 90, 2: 180, 3: 270}
    for k, v in char_data.items(): char_data[k] = mapping.get(char_keys[k].get())

class Obj:
    def __init__(self, name, i, j):
        for l, axis in enumerate('XYZ'):
            w = tk.Scale(gui, label=f"{name}_{axis}", orient=tk.HORIZONTAL, from_=0, to=3, command=send_value)
            w.set(0)
            w.grid(row=i, column=j + l)
            setattr(self, f"w_{axis}", w)


head = Obj('head', 0, 2)
char_keys['w_head_X'], char_keys['w_head_Y'], char_keys['w_head_Z'] = head.w_X, head.w_Y, head.w_Z

neck_01 = Obj('neck_01', 1, 2)
char_keys['w_neck_01_X'], char_keys['w_neck_01_Y'], char_keys['w_neck_01_Z'] = neck_01.w_X, neck_01.w_Y, neck_01.w_Z

clavicle_r = Obj('clavicle_r', 2, 0)
char_keys['w_clavicle_r_X'], char_keys['w_clavicle_r_Y'], char_keys['w_clavicle_r_Z'] = clavicle_r.w_X, clavicle_r.w_Y, clavicle_r.w_Z

upperarm_r = Obj('upperarm_r', 3, 0)
char_keys['w_upperarm_r_X'], char_keys['w_upperarm_r_Y'], char_keys['w_upperarm_r_Z'] = upperarm_r.w_X, upperarm_r.w_Y, upperarm_r.w_Z

lowerarm_r = Obj('lowerarm_r', 4, 0)
char_keys['w_lowerarm_r_X'], char_keys['w_lowerarm_r_Y'], char_keys['w_lowerarm_r_Z'] = lowerarm_r.w_X, lowerarm_r.w_Y, lowerarm_r.w_Z

hand_r = Obj('hand_r', 5, 0)
char_keys['w_hand_r_X'], char_keys['w_hand_r_Y'], char_keys['w_hand_r_Z'] = hand_r.w_X, hand_r.w_Y, hand_r.w_Z

clavicle_l = Obj('clavicle_l', 2, 4)
char_keys['w_clavicle_l_X'], char_keys['w_clavicle_l_Y'], char_keys['w_clavicle_l_Z'] = clavicle_l.w_X, clavicle_l.w_Y, clavicle_l.w_Z

upperarm_l = Obj('upperarm_l', 3, 4)
char_keys['w_upperarm_l_X'], char_keys['w_upperarm_l_Y'], char_keys['w_upperarm_l_Z'] = upperarm_l.w_X, upperarm_l.w_Y, upperarm_l.w_Z

lowerarm_l = Obj('lowerarm_l', 4, 4)
char_keys['w_lowerarm_l_X'], char_keys['w_lowerarm_l_Y'], char_keys['w_lowerarm_l_Z'] = lowerarm_l.w_X, lowerarm_l.w_Y, lowerarm_l.w_Z

hand_l = Obj('hand_l', 5, 4)
char_keys['w_hand_l_X'], char_keys['w_hand_l_Y'], char_keys['w_hand_l_Z'] = hand_l.w_X, hand_l.w_Y, hand_l.w_Z

spine_03 = Obj('spine_03', 7, 2)
char_keys['w_spine_03_X'], char_keys['w_spine_03_Y'], char_keys['w_spine_03_Z'] = spine_03.w_X, spine_03.w_Y, spine_03.w_Z

spine_02 = Obj('spine_02', 8, 2)
char_keys['w_spine_02_X'], char_keys['w_spine_02_Y'], char_keys['w_spine_02_Z'] = spine_02.w_X, spine_02.w_Y, spine_02.w_Z

spine_01 = Obj('spine_01', 9, 2)
char_keys['w_spine_01_X'], char_keys['w_spine_01_Y'], char_keys['w_spine_01_Z'] = spine_01.w_X, spine_01.w_Y, spine_01.w_Z

pelvis = Obj('pelvis', 10, 2)
char_keys['w_pelvis_X'], char_keys['w_pelvis_Y'], char_keys['w_pelvis_Z'] = pelvis.w_X, pelvis.w_Y, pelvis.w_Z

thigh_r = Obj('thigh_r', 11, 0)
char_keys['w_thigh_r_X'], char_keys['w_thigh_r_Y'], char_keys['w_thigh_r_Z'] = thigh_r.w_X, thigh_r.w_Y, thigh_r.w_Z

calf_r = Obj('calf_r', 12, 0)
char_keys['w_calf_r_X'], char_keys['w_calf_r_Y'], char_keys['w_calf_r_Z'] = calf_r.w_X, calf_r.w_Y, calf_r.w_Z

foot_r = Obj('foot_r', 13, 0)
char_keys['w_foot_r_X'], char_keys['w_foot_r_Y'], char_keys['w_foot_r_Z'] = foot_r.w_X, foot_r.w_Y, foot_r.w_Z

ball_r = Obj('ball_r', 14, 0)
char_keys['w_ball_r_X'], char_keys['w_ball_r_Y'], char_keys['w_ball_r_Z'] = ball_r.w_X, ball_r.w_Y, ball_r.w_Z

thigh_l = Obj('thigh_l', 11, 4)
char_keys['w_thigh_l_X'], char_keys['w_thigh_l_Y'], char_keys['w_thigh_l_Z'] = thigh_l.w_X, thigh_l.w_Y, thigh_l.w_Z

calf_l = Obj('calf_l', 12, 4)
char_keys['w_calf_l_X'], char_keys['w_calf_l_Y'], char_keys['w_calf_l_Z'] = calf_l.w_X, calf_l.w_Y, calf_l.w_Z

foot_l = Obj('foot_l', 13, 4)
char_keys['w_foot_l_X'], char_keys['w_foot_l_Y'], char_keys['w_foot_l_Z'] = foot_l.w_X, foot_l.w_Y, foot_l.w_Z

ball_l = Obj('ball_l', 14, 4)
char_keys['w_ball_l_X'], char_keys['w_ball_l_Y'], char_keys['w_ball_l_Z'] = ball_l.w_X, ball_l.w_Y, ball_l.w_Z

print('Server Listening')

zmq_rep_task = ZMQ_REP_Task()
zmq_thread = threading.Thread(target=zmq_rep_task.run, args=(10,))
zmq_thread.start()

tk.mainloop()
