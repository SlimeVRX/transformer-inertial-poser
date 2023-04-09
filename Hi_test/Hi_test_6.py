import pickle

# f = r'data\\preprocessed_DIP_IMU_v1\\dipimu_s_09_01_a.pkl'

# data = pickle.load(open(f, "rb"))

file = './dipimu_s_09_01_a.pkl'
with open(file, "rb") as f:
    data = pickle.load(f, encoding="latin1")

gt_list = []

Y = data['nimble_qdq']

start = 0
end = Y.shape[0]
Y = Y[start: end, :]

Y[:, 2] += 0.05       # move motion root 5 cm up

gt_list.append(Y)

a = 0

import pybullet as pb
import pybullet_data

import inspect
import functools
import numpy as np

class BulletClient(object):
    def __init__(self, connection_mode=pb.DIRECT, options=""):
        self._client = pb.connect(pb.SHARED_MEMORY)
        if (self._client < 0):
            self._client = pb.connect(connection_mode, options=options)
        self._shapes = {}

    def __del__(self):
        try:
            pb.disconnect(physicsClientId=self._client)
        except pb.error:
            pass

    def __getattr__(self, name):
        attribute = getattr(pb, name)
        if inspect.isbuiltin(attribute):
            attribute = functools.partial(attribute, physicsClientId=self._client)
        return attribute

Mode = pb.GUI
pb_client = BulletClient(connection_mode=Mode)
# Add path
pb_client.setAdditionalSearchPath(pybullet_data.getDataPath())
pb_client.resetSimulation()

import importlib.util
spec = importlib.util.spec_from_file_location(
    "char_info", "../amass_char_info.py")
char_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(char_info)

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

class utils:
    @staticmethod
    def _apply_fn_agnostic_to_vec_mat(input, fn):
        output = np.array([input]) if input.ndim == 1 else input
        output = np.apply_along_axis(fn, 1, output)
        return output[0] if input.ndim == 1 else output

from scipy.spatial.transform import Rotation

class conversions:

    @staticmethod
    def E2R(theta):
        return Rotation.from_euler("xyz", theta).as_matrix()

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
    def p2T(p):
        return conversions.Rp2T(constants.eye_R(), np.array(p))

    @staticmethod
    def R2T(R):
        return conversions.Rp2T(R, constants.zero_p())

    @staticmethod
    def batch_auto_reshape(x, fn, shape_in, shape_out):
        reshape = x.ndim - len(shape_in) > 1
        xx = x.reshape(-1, *shape_in) if reshape else x
        y = fn(xx)
        return y.reshape(x.shape[: -len(shape_in)] + shape_out) if reshape else y

    @staticmethod
    def A2R(A):
        return conversions.batch_auto_reshape(
            A, lambda x: Rotation.from_rotvec(x).as_matrix(), (3,), (3, 3),
        )

    @staticmethod
    def Q2R(Q):
        return conversions.batch_auto_reshape(
            Q, lambda x: Rotation.from_quat(x).as_matrix(), (4,), (3, 3),
        )

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
    def Qp2T(Q, p):
        R = conversions.Q2R(Q)
        return conversions.Rp2T(R, p)

    @staticmethod
    def R2A(R):
        return conversions.batch_auto_reshape(
            R, lambda x: Rotation.from_matrix(x).as_rotvec(), (3, 3), (3,),
        )

    @staticmethod
    def A2Q(A):
        return conversions.batch_auto_reshape(
            A, lambda x: Rotation.from_rotvec(x).as_quat(), (3,), (4,),
        )

import warnings

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

import math

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

class bullet_utils:
    xyzw_in = True

    @staticmethod
    def set_base_pQvw(pb_client, body_id, p, Q, v=None, w=None):
        """
        Set positions, orientations, linear and angular velocities of the base link.
        """
        if not bullet_utils.xyzw_in:
            Q = quaternion.Q_op(Q, op=['change_order'], xyzw_in=False)
        pb_client.resetBasePositionAndOrientation(body_id, p, Q)
        if v is not None and w is not None:
            pb_client.resetBaseVelocity(body_id, v, w)

    @staticmethod
    def get_joint_pv(pb_client, body_id, indices=None):
        """
        Return positions and velocities given joint indices.
        Please note that the values are locally repsented w.r.t. its parent joint
        """
        if indices is None:
            indices = range(pb_client.getNumJoints(body_id))

        num_indices = len(indices)
        assert num_indices > 0

        js = pb_client.getJointStatesMultiDof(body_id, indices)

        ps = []
        vs = []
        for j in range(num_indices):
            p = np.array(js[j][0])
            v = np.array(js[j][1])
            if len(p) == 4 and not bullet_utils.xyzw_in:
                p = quaternion.Q_op(p, op=['change_order'], xyzw_in=True)
            ps.append(p)
            vs.append(v)

        if num_indices == 1:
            return ps[0], vs[0]
        else:
            return ps, vs

    @staticmethod
    def set_joint_pv(pb_client, body_id, indices, ps, vs):
        """
        Set positions and velocities given joint indices.
        Please note that the values are locally repsented w.r.t. its parent joint
        """
        ps_processed = ps.copy()
        for i in range(len(ps_processed)):
            if len(ps_processed[i]) == 4 and not bullet_utils.xyzw_in:
                ps_processed[i] = \
                    quaternion.Q_op(ps_processed[i], op=['change_order'], xyzw_in=False)
        print("ps_processed", ps_processed)
        print("vs", vs)
        pb_client.resetJointStatesMultiDof(body_id, indices, ps_processed, vs)

class SimAgent(object):
    def __init__(self,
                 pybullet_client,
                 model_file,
                 char_info,
                 scale=1.0,
                 ref_scale=1.0,
                 self_collision=True,
                 ):
        self._char_info = char_info
        self._ref_scale = ref_scale
        self._pb_client = pybullet_client

        char_create_flags = self._pb_client.URDF_MAINTAIN_LINK_ORDER
        if self_collision:
            char_create_flags = char_create_flags | \
                                self._pb_client.URDF_USE_SELF_COLLISION | \
                                self._pb_client.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS

        self._body_id = self._pb_client.loadURDF(model_file,
                                                 [0, 0, 0],
                                                 globalScaling=scale,
                                                 useFixedBase=False,
                                                 flags=char_create_flags)
        self._num_joint = self._pb_client.getNumJoints(self._body_id)
        self._joint_indices = range(self._num_joint)

        self._joint_indices_movable = []
        self._joint_type = []
        self._joint_axis = []
        self._joint_dofs = []

        for j in self._joint_indices:
            joint_info = self._pb_client.getJointInfo(self._body_id, j)
            self._joint_type.append(joint_info[2])
            self._joint_axis.append(np.array(joint_info[13]))

        for j in self._joint_indices:
            if self._joint_type[j] == self._pb_client.JOINT_SPHERICAL:
                self._joint_dofs.append(3)
                self._joint_indices_movable.append(j)
            elif self._joint_type[j] == self._pb_client.JOINT_REVOLUTE:
                self._joint_dofs.append(1)
                self._joint_indices_movable.append(j)
            elif self._joint_type[j] == self._pb_client.JOINT_FIXED:
                self._joint_dofs.append(0)
            else:
                raise NotImplementedError()

        self._joint_pose_init, self._joint_vel_init = self.get_joint_states()

        # get all non-root active indices
        def get_all_non_root_active_index():
            all_joint_idx = self._char_info.joint_idx.values()
            indices = []  # excluding root
            for idx in all_joint_idx:
                if self.get_joint_type(idx) == self._pb_client.JOINT_FIXED:  # 14&18, l&r wrists
                    continue
                if idx == self._char_info.root:
                    continue
                indices.append(idx)
            return indices

        self.non_root_active_idx = get_all_non_root_active_index()

    def get_joint_type(self, idx):
        return self._joint_type[idx]

    def get_joint_states(self, indices=None):
        return bullet_utils.get_joint_pv(self._pb_client, self._body_id, indices)

    def get_char_info(self):
        return self._char_info

    def set_pose(self, pose, vel=None):
        """
        Velocity should be represented w.r.t. local frame
        """
        # Root joint
        T = pose.get_transform(
            self._char_info.bvh_map[self._char_info.ROOT],
            local=False)
        Q, p = conversions.T2Qp(T)
        p *= self._ref_scale

        v, w = None, None
        if vel is not None:
            # Here we give a root orientation to get velocities represeted in world frame.
            R = conversions.Q2R(Q)
            w = vel.get_angular(
                self._char_info.bvh_map[self._char_info.ROOT], False, R)
            v = vel.get_linear(
                self._char_info.bvh_map[self._char_info.ROOT], False, R)
            v *= self._ref_scale

        bullet_utils.set_base_pQvw(self._pb_client, self._body_id, p, Q, v, w)

        # Other joints
        indices = []
        state_pos = []
        state_vel = []
        for j in self._joint_indices:
            joint_type = self._joint_type[j]
            # When the target joint do not have dof, we simply ignore it
            if joint_type == self._pb_client.JOINT_FIXED:
                continue
            # When there is no matching between the given pose and the simulated character,
            # the character just tries to hold its initial pose
            if self._char_info.bvh_map[j] is None:
                state_pos.append(self._joint_pose_init[j])
                state_vel.append(self._joint_vel_init[j])
            else:
                T = pose.get_transform(self._char_info.bvh_map[j], local=True)
                if joint_type == self._pb_client.JOINT_SPHERICAL:
                    Q, p = conversions.T2Qp(T)
                    w = np.zeros(3) if vel is None else vel.get_angular(self._char_info.bvh_map[j], local=True)
                    state_pos.append(Q)
                    state_vel.append(w)
                elif joint_type == self._pb_client.JOINT_REVOLUTE:
                    joint_axis = self.get_joint_axis(j)
                    R, p = conversions.T2Rp(T)
                    w = np.zeros(3) if vel is None else vel.get_angular(self._char_info.bvh_map[j], local=True)
                    state_pos.append([f_math.project_rotation_1D(R, joint_axis)])
                    state_vel.append([f_math.project_angular_vel_1D(w, joint_axis)])
                else:
                    raise NotImplementedError()
            indices.append(j)
        bullet_utils.set_joint_pv(self._pb_client, self._body_id, indices, state_pos, state_vel)

        # bullet_utils.set_base_pQvw
        # bullet_utils.set_joint_pv
        #

    def get_joint_axis(self, idx):
        return self._joint_axis[idx]

    def set_root_pQvw(self, p, Q, v, w):
        bullet_utils.set_base_pQvw(self._pb_client, self._body_id, p, Q, v, w)

    def set_joints_pv(self, indices, ps, vs):
        bullet_utils.set_joint_pv(self._pb_client,
                        self._body_id,
                        indices,
                        ps,
                        vs)

class data_util:
    @staticmethod
    def our_pose_2_bullet_format(
            char: SimAgent,
            s_np: np.ndarray
    ) -> np.ndarray:
        bullet_q = []
        bullet_q += list(s_np[:6])

        for idx in char.non_root_active_idx:
            nimble_state_start = (char.get_char_info().nimble_state_map[idx] - 1) * 3 + 6
            aa = s_np[nimble_state_start:(nimble_state_start + 3)]
            bullet_q += list(aa)

        assert len(bullet_q) == len(s_np) // 2
        return np.array(bullet_q)

robot = SimAgent(pybullet_client=pb_client,
                 model_file="../data/amass.urdf",
                 char_info=char_info,
                 scale=1.0,
                 ref_scale=1.0,
                 self_collision=True)

def post_processing_our_model(
        char: SimAgent,
        ours_out: np.ndarray) -> np.ndarray:
    poses_post = []
    for pose in ours_out:
        pose_post = data_util.our_pose_2_bullet_format(char, pose)
        poses_post.append(pose_post.tolist())
    poses_post = np.array(poses_post)

    return poses_post

def viz_current_frame_and_store_fk_info_include_fixed(
        char: SimAgent,
        state_bullet: np.ndarray,
):
    # use the bullet character for both visualization and FK

    state_pos = []
    state_pq_g = []
    state_pq_g_jf = []

    v = w = np.array([0., 0., 0.])
    Q_root = conversions.A2Q(state_bullet[3:6])
    char.set_root_pQvw(state_bullet[:3], Q_root, v, w)

    for i, j_idx in enumerate(char.non_root_active_idx):
        aa = state_bullet[i*3+6: i*3+9]
        state_pos.append(conversions.A2Q(aa))

    char.set_joints_pv(
        char.non_root_active_idx,
        state_pos,
        [np.zeros(3)] * len(char.non_root_active_idx)
    )

for i in range(len(gt_list)):
    traj_1 = post_processing_our_model(robot, gt_list[i])
    start_t, end_t = 30, 6
    m_len = len(traj_1)
    for t in range(start_t, m_len-end_t):
        pq_g_1 = viz_current_frame_and_store_fk_info_include_fixed(robot, traj_1[t])
    a = 0


# IMU -> qdq -> pybullet
# AMASS -> qdq -> pybullet
# AMASS -> (SMPL) -> pybullet
# AMASS -> SMPL -> VseeFace

# qdq -> AMASS -> SMPL
# fairmotion POSE SMPL


