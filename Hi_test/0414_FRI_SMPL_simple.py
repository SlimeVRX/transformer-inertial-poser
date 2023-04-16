import time

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
        # root_R (3,3) p (3,)
        a = R.shape[:-2]
        b = 0
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

# with open(('pose_smpl_UI.npy'), 'rb') as f:
#     _pose_smpl = np.load(f)
_pose_smpl = [np.eye(4)]*24
_pose_smpl = np.array(_pose_smpl)

key = 'root'

xform_from_parent_joint = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
T0 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

cur_time = 0.0 / 2.0
imu_idx = 0
DT = 1. / 60

rot_up_Q = np.array([0.5, 0.5, 0.5, 0.5])
rot_up_R = conversions.Q2R(rot_up_Q)
root_z_offset = 0.95

while cur_time < 20.0: # 20.0

    pose_smpl = _pose_smpl.copy()

    # get_transform
    get_index_joint = pose_smpl[0]
    belly_R = np.dot(
        xform_from_parent_joint,
        get_index_joint
    )[:3, :3]

    root_R = rot_up_R.dot(belly_R)
    p = np.array([0, 0, root_z_offset])

    T_ = conversions.Rp2T(root_R, p)    # root_R (3,3) p (3,)
    T1 = np.dot(f_math.invertT(T0), T_)

    Q_, p_ = conversions.T2Qp(T1)
    Q_ = quaternion.Q_op(Q_, op=["normalize"])
    T1_ = conversions.Qp2T(Q_, p_)

    pose_smpl[0] = T1_
    #---------------------------------------
    T = np.dot(
        xform_from_parent_joint,
        get_index_joint
    )
    print(T)

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
        # elif joint_type == JOINT_REVOLUTE:
        #     joint_axis = _joint_axis[j]
        #     R, p = conversions.T2Rp(T)
        #     w = np.zeros(3)
        #     state_pos.append([f_math.project_rotation_1D(R, joint_axis)])
        #     state_vel.append([f_math.project_angular_vel_1D(w, joint_axis)])
        else:
            raise NotImplementedError()
        indices.append(j)

    # bullet_utils.set_joint_pv(self._pb_client, self._body_id, indices, state_pos, state_vel)
    pb.resetJointStatesMultiDof(robot, indices, state_pos, state_vel)

    a = 0
    time.sleep(1. / 240)
    cur_time += DT
    imu_idx += 1
