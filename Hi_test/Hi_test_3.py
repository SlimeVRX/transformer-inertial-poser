# '''
#     import pybullet as p
#     import time
#     import pybullet_data
#     physicsClient = pb.connect(pb.GUI)#or pb.DIRECT for non-graphical version
#     pb.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
#     pb.setGravity(0,0,-10)
#     planeId = pb.loadURDF("plane.urdf")
#     startPos = [0,0,1]
#     startOrientation = pb.getQuaternionFromEuler([0,0,0])
#     boxId = pb.loadURDF("r2d2.urdf",startPos, startOrientation)
#     #set the center of mass frame (loadURDF sets base link frame) startPos/Ornpb.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
#     for i in range (10000):
#         pb.stepSimulation()
#         time.sleep(1./240.)
#     cubePos, cubeOrn = pb.getBasePositionAndOrientation(boxId)
#     print(cubePos,cubeOrn)
#     pb.disconnect()
# '''
#
# # set vitual environment
# # set 3d model
# # set condination
#
# # don gian nhat, ket noi server, load 3D model, gui data control 3D model
#
# import time
# import pybullet as pb
# import pybullet_data
#
# pb_client = pb.connect(pb.GUI)
# pb_client.setAdditionalSearchPath(pybullet_data.getDataPath())
# pb_client.resetSimulation()
#
# planeId = pb.loadURDF("data/amass.urdf")
#
# for i in range (10000):
#
#     time.sleep(1./240.)
#
#
# class BulletClient(object):
#     def __init__(self, connection_mode=pb.DIRECT, options=""):
#         pass
#     def __del__(self):
#         pass
#     def __getattr__(self, name):
#         pass

# class Person:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
#
#     def run(self, key):
#         print("Inside `__getitem__` method!")
#         return getattr(self, key)
#
# p = Person("Subhayan",32)
# print(pb.run("age"))

# pb_client = pb.connect(pb.GUI)
# pb_client.setAdditionalSearchPath(pybullet_data.getDataPath())
# pb_client.resetSimulation()

# import inspect
# import pybullet as pb
# import pybullet_data
#
# class PybulletClient(object):
#     def __init__(self, connection_mode=pb.GUI):
#         self._client = pb.connect(connection_mode)
#
#     def __getattr__(self, name):
#         attribute = getattr(pb, name)
#         print(inspect.isbuiltin(attribute))
#
# pb_client = PybulletClient(connection_mode=pb.GUI)
# pb_client.setAdditionalSearchPath(pybullet_data.getDataPath())
# pb_client.resetSimulation()
import time

import numpy as np
import pybullet as pb
import inspect
import pybullet_data
import functools

# class PybulletClient(object):
#     def __init__(self, connection_mode=pb.DIRECT, options=""):
#         self._client = pb.connect(pb.SHARED_MEMORY)
#         if (self._client < 0):
#             self._client = pb.connect(connection_mode, options=options)
#         self._shapes = {}
#
#     def __del__(self):
#         try:
#             pb.disconnect(physicsClientId=self._client)
#         except pb.error:
#             pass
#
#     def __getattr__(self, name):
#         attribute = getattr(pb, name)
#         if inspect.isbuiltin(attribute):
#             attribute = functools.partial(attribute, physicsClientId=self._client)
#         return attribute
#
# pb_client = PybulletClient(connection_mode=pb.GUI)
# pb_client.setAdditionalSearchPath(pybullet_data.getDataPath())
# pb_client.resetSimulation()



scale=1.0
char_create_flags = pb.URDF_MAINTAIN_LINK_ORDER
if 1:
    char_create_flags = char_create_flags | \
                        pb.URDF_USE_SELF_COLLISION | \
                        pb.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS

indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
ps_processed = [
    np.array([ 0.01199631, -0.0282913 , -0.02999078,  0.9990777 ]),
    np.array([ 0.01199631, -0.0282913 , -0.02999078,  0.9990777 ]),
    np.array([ 0.01199631, -0.0282913 , -0.02999078,  0.9990777 ]),
    np.array([ 0.01199631, -0.0282913 , -0.02999078,  0.9990777 ]),
    np.array([ 0.01199631, -0.0282913 , -0.02999078,  0.9990777 ]),
    np.array([ 0.01199631, -0.0282913 , -0.02999078,  0.9990777 ]),
    np.array([ 0.01199631, -0.0282913 , -0.02999078,  0.9990777 ]),
    np.array([ 0.01199631, -0.0282913 , -0.02999078,  0.9990777 ]),
    np.array([ 0.01199631, -0.0282913 , -0.02999078,  0.9990777 ]),
    np.array([ 0.01199631, -0.0282913 , -0.02999078,  0.9990777 ]),
    np.array([ 0.01199631, -0.0282913 , -0.02999078,  0.9990777 ]),
    np.array([ 0.01199631, -0.0282913 , -0.02999078,  0.9990777 ]),
    np.array([ 0.01199631, -0.0282913 , -0.02999078,  0.9990777 ]),
    np.array([ 0.01199631, -0.0282913 , -0.02999078,  0.9990777 ]),
    np.array([ 0.01199631, -0.0282913 , -0.02999078,  0.9990777 ]),
    np.array([ 0.01199631, -0.0282913 , -0.02999078,  0.9990777 ]),
    np.array([ 0.01199631, -0.0282913 , -0.02999078,  0.9990777 ]),
    np.array([ 0.01199631, -0.0282913 , -0.02999078,  0.9990777 ]),
]
vs = [
    np.zeros([3]),
    np.zeros([3]),
    np.zeros([3]),
    np.zeros([3]),
    np.zeros([3]),
    np.zeros([3]),
    np.zeros([3]),
    np.zeros([3]),
    np.zeros([3]),
    np.zeros([3]),
    np.zeros([3]),
    np.zeros([3]),
    np.zeros([3]),
    np.zeros([3]),
    np.zeros([3]),
    np.zeros([3]),
    np.zeros([3]),
    np.zeros([3]),
]

DT = 1. / 60
cur_time = 0.0 / 2.0
imu_idx = 0

# connect to server pybullet
pb_client = pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
pb.resetSimulation()

# load 3D model
robot = pb.loadURDF("data/amass.urdf",
            [0, 0, 0],
            globalScaling=scale,
            useFixedBase=False,
            flags=char_create_flags)

while cur_time < 20.0:

    # send data to server
    pb.resetBasePositionAndOrientation(robot, [0, 0, 0.95], [0.5, 0.5, 0.5, 0.5])
    pb.resetJointStatesMultiDof(robot, indices, ps_processed, vs)

    cur_time += DT
    imu_idx += 1
    time.sleep(1./240.)
    print(imu_idx)

# set_base_pQvw():

# set_joint_pv():


a = 0

# amass -> qdq -> pybullet
#

