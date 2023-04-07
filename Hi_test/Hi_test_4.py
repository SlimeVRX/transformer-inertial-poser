import numpy as np
import pickle as pkl
import math
import random
import warnings
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

class utils:
    @staticmethod
    def _apply_fn_agnostic_to_vec_mat(input, fn):
        output = np.array([input]) if input.ndim == 1 else input
        output = np.apply_along_axis(fn, 1, output)
        return output[0] if input.ndim == 1 else output

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

class Motion(object):
    def __init__(
        self, name="motion", skel=None, fps=60,
    ):
        self.name = name
        self.skel = skel
        self.poses = []
        self.fps = fps
        self.fps_inv = 1.0 / fps
        self.info = {}

    def set_fps(self, fps):
        self.fps = fps
        self.fps_inv = 1.0 / fps

    def add_one_frame(self, pose_data):
        """Adds a pose at the end of motion object.

        Args:
            pose_data: List of pose data, where each pose
        """
        self.poses.append(Pose(self.skel, pose_data))

    def get_pose_by_time(self, time):
        """
        If specified time is close to an integral multiple of (1/fps), returns
        the pose at that time. Else, returns an interpolated version
        """
        time = np.clip(time, 0, self.length())
        frame1 = self.time_to_frame(time)
        frame2 = min(frame1 + 1, self.num_frames() - 1)
        if frame1 == frame2:
            return self.poses[frame1]

        t1 = self.frame_to_time(frame1)
        # t1 0.0
        t2 = self.frame_to_time(frame2)
        # t2 0.016666666666666666
        alpha = np.clip((time - t1) / (t2 - t1), 0.0, 1.0)

        return Pose.interpolate(self.poses[frame1], self.poses[frame2], alpha)

    def length(self):
        return (len(self.poses) - 1) * self.fps_inv

    def time_to_frame(self, time):
        return int(time * self.fps + 1e-05)

    def num_frames(self):
        return len(self.poses)

    def frame_to_time(self, frame):
        frame = np.clip(frame, 0, len(self.poses) - 1)
        return frame * self.fps_inv

class Pose(object):
    def __init__(self, skel, data=None):
        assert isinstance(skel, Skeleton)
        if data is None:
            data = [constants.eye_T for _ in range(skel.num_joints())]
        assert skel.num_joints() == len(data), "{} vs. {}".format(
            skel.num_joints(), len(data)
        )
        self.skel = skel
        self.data = data

    def get_transform(self, key, local):
        skel = self.skel
        if local:
            return self.data[skel.get_index_joint(key)]
        else:
            joint = skel.get_joint(key)
            T = np.dot(
                joint.xform_from_parent_joint,
                self.data[skel.get_index_joint(joint)],
            )
            while joint.parent_joint is not None:
                T_j = np.dot(
                    joint.parent_joint.xform_from_parent_joint,
                    self.data[skel.get_index_joint(joint.parent_joint)],
                )
                T = np.dot(T_j, T)
                joint = joint.parent_joint
            return T

    def set_transform(self, key, T, local, do_ortho_norm=True):
        if local:
            T1 = T
        else:
            T0 = self.skel.get_joint(key).xform_global
            T1 = np.dot(f_math.invertT(T0), T)
        if do_ortho_norm:
            """
            This insures that the rotation part of
            the given transformation is valid
            """
            Q, p = conversions.T2Qp(T1)
            Q = quaternion.Q_op(Q, op=["normalize"])
            T1 = conversions.Qp2T(Q, p)
        self.data[self.skel.get_index_joint(key)] = T1

    @classmethod
    def interpolate(cls, pose1, pose2, alpha):
        skel = pose1.skel
        data = []
        for j in skel.joints:
            R1, p1 = conversions.T2Rp(pose1.get_transform(j, local=True))
            R2, p2 = conversions.T2Rp(pose2.get_transform(j, local=True))
            R, p = (
                f_math.slerp(R1, R2, alpha),
                f_math.lerp(p1, p2, alpha),
            )
            data.append(conversions.Rp2T(R, p))
        return cls(pose1.skel, data)

class Skeleton(object):
    def __init__(
        self,
        name="skeleton",
        v_up=np.array([0.0, 1.0, 0.0]),
        v_face=np.array([0.0, 0.0, 1.0]),
        v_up_env=np.array([0.0, 1.0, 0.0]),
    ):
        self.name = name
        self.joints = []
        self.index_joint = {}
        self.root_joint = None
        self.num_dofs = 0
        self.v_up = v_up
        self.v_face = v_face
        self.v_up_env = v_up_env

    def add_joint(self, joint, parent_joint):
        if parent_joint is None:
            assert self.num_joints() == 0
            self.root_joint = joint
        else:
            parent_joint = self.get_joint(parent_joint)
            parent_joint.add_child_joint(joint)
        self.index_joint[joint.name] = len(self.joints)
        self.joints.append(joint)
        self.num_dofs += joint.info["dof"]

    def num_joints(self):
        return len(self.joints)

    @staticmethod
    def get_index(index_dict, key):
        if isinstance(key, int):
            return key
        elif isinstance(key, str):
            return index_dict[key]
        else:
            return index_dict[key.name]

    def get_index_joint(self, key):
        return Skeleton.get_index(self.index_joint, key)

    def get_joint(self, key):
        return self.joints[self.get_index_joint(key)]

class Joint(object):
    def __init__(
        self,
        name=None,
        dof=3,
        xform_from_parent_joint=constants.eye_T(),
        parent_joint=None,
        limits=None,
        direction=None,
        length=None,
        axis=None,
    ):
        self.name = name if name else f"joint_{random.getrandbits(32)}"
        self.child_joints = []
        self.index_child_joint = {}
        self.xform_global = constants.eye_T()
        self.xform_from_parent_joint = xform_from_parent_joint
        self.parent_joint = self.set_parent_joint(parent_joint)
        self.info = {"dof": dof}  # set ball joint by default

        self.length = length

        if axis is not None:
            axis = np.deg2rad(axis)
            self.C = conversions.E2R(axis)
            self.Cinv = np.linalg.inv(self.C)
            self.matrix = None
            self.degree = np.zeros(3)
            self.coordinate = None
        if direction is not None:
            self.direction = direction.squeeze()
        if limits is not None:
            self.limits = np.zeros([3, 2])
            for lm, nm in zip(limits, dof):
                if nm == "rx":
                    self.limits[0] = lm
                elif nm == "ry":
                    self.limits[1] = lm
                else:
                    self.limits[2] = lm

    def set_parent_joint(self, joint):
        if joint is None:
            self.parent_joint = None
            return
        assert isinstance(joint, Joint)
        self.parent_joint = joint
        self.xform_global = np.dot(
            self.parent_joint.xform_global, self.xform_from_parent_joint,
        )

    def add_child_joint(self, joint):
        assert isinstance(joint, Joint)
        assert joint.name not in self.index_child_joint.keys()
        self.index_child_joint[joint.name] = len(self.child_joints)
        self.child_joints.append(joint)
        joint.set_parent_joint(self)

SMPL_NR_JOINTS = 24
OFFSETS = np.array(
    [
        [-8.76308970e-04, -2.11418723e-01, 2.78211200e-02],
        [7.04848876e-02, -3.01002533e-01, 1.97749280e-02],
        [-6.98883278e-02, -3.00379160e-01, 2.30254335e-02],
        [-3.38451650e-03, -1.08161861e-01, 5.63597909e-03],
        [1.01153808e-01, -6.65211904e-01, 1.30860155e-02],
        [-1.06040718e-01, -6.71029623e-01, 1.38401121e-02],
        [1.96440985e-04, 1.94957852e-02, 3.92296547e-03],
        [8.95999143e-02, -1.04856032e00, -3.04155922e-02],
        [-9.20120818e-02, -1.05466743e00, -2.80514913e-02],
        [2.22362284e-03, 6.85680141e-02, 3.17901760e-02],
        [1.12937580e-01, -1.10320516e00, 8.39545265e-02],
        [-1.14055299e-01, -1.10107698e00, 8.98482216e-02],
        [2.60992373e-04, 2.76811197e-01, -1.79753042e-02],
        [7.75218998e-02, 1.86348444e-01, -5.08464100e-03],
        [-7.48091986e-02, 1.84174211e-01, -1.00204779e-02],
        [3.77815350e-03, 3.39133394e-01, 3.22299558e-02],
        [1.62839013e-01, 2.18087461e-01, -1.23774789e-02],
        [-1.64012068e-01, 2.16959041e-01, -1.98226746e-02],
        [4.14086325e-01, 2.06120683e-01, -3.98959248e-02],
        [-4.10001734e-01, 2.03806676e-01, -3.99843890e-02],
        [6.52105424e-01, 2.15127546e-01, -3.98521818e-02],
        [-6.55178550e-01, 2.12428626e-01, -4.35159074e-02],
        [7.31773168e-01, 2.05445019e-01, -5.30577698e-02],
        [-7.35578759e-01, 2.05180646e-01, -5.39352281e-02],
    ]
)
SMPL_PARENTS = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
]
SMPL_JOINTS = [
    "root",
    "lhip",
    "rhip",
    "lowerback",
    "lknee",
    "rknee",
    "upperback",
    "lankle",
    "rankle",
    "chest",
    "ltoe",
    "rtoe",
    "lowerneck",
    "lclavicle",
    "rclavicle",
    "upperneck",
    "lshoulder",
    "rshoulder",
    "lelbow",
    "relbow",
    "lwrist",
    "rwrist",
    "lhand",
    "rhand",
]

class dip_loader:
    @staticmethod
    def load(
        file,
        motion=None,
        scale=1.0,
        load_skel=True,
        load_motion=True,
        v_up_skel=np.array([0.0, 1.0, 0.0]),
        v_face_skel=np.array([0.0, 0.0, 1.0]),
        v_up_env=np.array([0.0, 1.0, 0.0]),
    ):
        if not motion:
            motion = Motion(fps=60)
        if load_skel:
            skel = Skeleton(
                v_up=v_up_skel, v_face=v_face_skel, v_up_env=v_up_env,
            )
            smpl_offsets = np.zeros([SMPL_NR_JOINTS, 3])
            smpl_offsets[0] = OFFSETS[0]
            for idx, pid in enumerate(SMPL_PARENTS[1:]):
                smpl_offsets[idx + 1] = OFFSETS[idx + 1] - OFFSETS[pid]
            for joint_name, parent_joint, offset in zip(
                SMPL_JOINTS, SMPL_PARENTS, smpl_offsets
            ):
                joint = Joint(name=joint_name)
                if parent_joint == -1:
                    parent_joint_name = None
                    joint.info["dof"] = 6  # root joint is free
                    offset -= offset
                else:
                    parent_joint_name = SMPL_JOINTS[parent_joint]
                offset = offset / np.linalg.norm(smpl_offsets[4])       # tai sao lai chia cho norm cua joint 4
                T1 = conversions.p2T(scale * offset)
                joint.xform_from_parent_joint = T1
                skel.add_joint(joint, parent_joint_name)
            motion.skel = skel
        else:
            assert motion.skel is not None

        if load_motion:
            assert motion.skel is not None

            if file.endswith("npz"):
                data = np.load(file)
            elif file.endswith("pkl"):
                with open(file, "rb") as f:
                    data = pkl.load(f, encoding="latin1")
            else:
                assert False

            if "mocap_framerate" in data:
                fps = float(data["mocap_framerate"])
            elif "frame_rate" in data:
                fps = float(data["frame_rate"])
            else:
                fps = 60.0      # Assume 60fps
            motion.set_fps(fps)

            if "gt" in data:
                poses = np.array(data["gt"])[:, :SMPL_NR_JOINTS * 3]  # shape (seq_length, 72)
            else:
                poses = np.array(data["poses"])  # shape (seq_length, 72)
            assert len(poses) > 0, "file is empty"
            # poses = poses.reshape((-1, len(SMPL_MAJOR_JOINTS), 3, 3))

            for pose_id, pose in enumerate(poses):
                pose_data = [
                    constants.eye_T() for _ in range(len(SMPL_JOINTS))
                ]

                for j, joint_name in enumerate(SMPL_JOINTS):
                    T = conversions.R2T(
                        conversions.A2R(
                            pose[j * 3: j * 3 + 3]
                        )
                    )
                    pose_data[
                        motion.skel.get_index_joint(joint_name)
                    ] = T

                motion.add_one_frame(pose_data)
        return motion

motion_file = './04.pkl'
v_up_skel = np.array([0., 1., 0.])
v_face_skel = np.array([0., 0., 1.])
v_up_env = np.array([0., 0., 1.])

motion = dip_loader.load(motion=None,
                    file=motion_file,
                    scale=1.0,
                    load_skel=True,
                    load_motion=True,
                    v_up_skel=v_up_skel,
                    v_face_skel=v_face_skel,
                    v_up_env=v_up_env)

a = 0
rot_up_Q = np.array([0.5, 0.5, 0.5, 0.5])
rot_up_R = conversions.Q2R(rot_up_Q)
root_z_offset = 0.95

for pose_id, pose in enumerate(motion.poses):
    belly_R = pose.get_transform("root", local=False)[:3, :3]
    root_R = rot_up_R.dot(belly_R)
    p = np.array([0, 0, root_z_offset])
    pose.set_transform("root", conversions.Rp2T(root_R, p), local=False)

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

import pybullet as pb
import pybullet_data

import inspect
import functools

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

robot = SimAgent(pybullet_client=pb_client,
                 model_file="../data/amass.urdf",
                 char_info=char_info,
                 scale=1.0,
                 ref_scale=1.0,
                 self_collision=True)

a = 0

cur_time = 0.0 / 2.0
imu_idx = 0
DT = 1. / 60

while cur_time < 20.0:      # test half
    cur_pose = motion.get_pose_by_time(cur_time)
    robot.set_pose(cur_pose, None)
    cur_time += DT
    imu_idx += 1
