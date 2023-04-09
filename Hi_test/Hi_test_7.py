import time
import numpy as np
import pybullet as pb
import pybullet_data
import pickle
import collections

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
robot = pb.loadURDF("../data/amass.urdf",
            [0, 0, 0],
            globalScaling=scale,
            useFixedBase=False,
            flags=char_create_flags)

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
Mapping from Bullet indices to Nimble state indices (no weld joints)
'''
nimble_state_map = collections.OrderedDict()
nimble_state_map[root] = 0
nimble_state_map[lhip] = 1
nimble_state_map[lknee] = 2
nimble_state_map[lankle] = 3
nimble_state_map[rhip] = 15
nimble_state_map[rknee] = 16
nimble_state_map[rankle] = 17
nimble_state_map[lowerback] = 4
nimble_state_map[upperback] = 5
nimble_state_map[chest] = 6
nimble_state_map[lowerneck] = 10
nimble_state_map[upperneck] = 11
nimble_state_map[lclavicle] = 7
nimble_state_map[lshoulder] = 8
nimble_state_map[lelbow] = 9
nimble_state_map[lwrist] = None
nimble_state_map[rclavicle] = 12
nimble_state_map[rshoulder] = 13
nimble_state_map[relbow] = 14
nimble_state_map[rwrist] = None

non_root_active_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17]

from scipy.spatial.transform import Rotation

class conversions:
    @staticmethod
    def batch_auto_reshape(x, fn, shape_in, shape_out):
        reshape = x.ndim - len(shape_in) > 1
        xx = x.reshape(-1, *shape_in) if reshape else x
        y = fn(xx)
        return y.reshape(x.shape[: -len(shape_in)] + shape_out) if reshape else y

    @staticmethod
    def A2Q(A):
        return conversions.batch_auto_reshape(
            A, lambda x: Rotation.from_rotvec(x).as_quat(), (3,), (4,),
        )

for i in range(len(gt_list)):
    poses_post = []
    for pose in gt_list[i]:
        bullet_q = []
        bullet_q += list(pose[:6])
        for idx in non_root_active_idx:
            nimble_state_start = (nimble_state_map[idx] - 1) * 3 + 6
            aa = pose[nimble_state_start:(nimble_state_start + 3)]
            bullet_q += list(aa)
        assert len(bullet_q) == len(pose) // 2
        poses_post.append(np.array(bullet_q).tolist())
    poses_post = np.array(poses_post)
    start_t, end_t = 30, 6
    m_len = len(poses_post)

    for t in range(start_t, m_len-end_t):
        state_bullet = poses_post[t]
        state_pos = []
        v = w = np.array([0., 0., 0.])
        Q_root = conversions.A2Q(state_bullet[3:6])

        # set_base_pQvw
        pb.resetBasePositionAndOrientation(robot, state_bullet[:3], Q_root)

        for i, j_idx in enumerate(non_root_active_idx):
            aa = state_bullet[i*3+6: i*3+9]
            state_pos.append(conversions.A2Q(aa))

        # set_joint_pv
        pb.resetJointStatesMultiDof(
            robot,
            non_root_active_idx,
            state_pos,
            [np.zeros(3)] * len(non_root_active_idx)
        )
        # time.sleep(1./240.)
