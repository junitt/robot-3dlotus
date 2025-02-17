# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import numpy as np
import copy
from rlbench.action_modes.arm_action_modes import (
    EndEffectorPoseViaPlanning,
    Scene,
)

from pyquaternion import Quaternion
from pyrep.const import ConfigurationPathAlgorithms as Algos
from pyrep.errors import ConfigurationPathError, IKError
from pyrep.const import ObjectType

from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.robot import Robot
from rlbench.backend.scene import Scene

def assert_action_shape(action: np.ndarray, expected_shape: tuple):
    if np.shape(action) != expected_shape:
        raise InvalidActionError(
            'Expected the action shape to be: %s, but was shape: %s' % (
                str(expected_shape), str(np.shape(action))))


def assert_unit_quaternion(quat):
    if not np.isclose(np.linalg.norm(quat), 1.0):
        raise InvalidActionError('Action contained non unit quaternion!')
    
def calculate_delta_pose(robot: Robot, action: np.ndarray):
    a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = action
    x, y, z, qx, qy, qz, qw = robot.arm.get_tip().get_pose()
    new_rot = Quaternion(
        a_qw, a_qx, a_qy, a_qz) * Quaternion(qw, qx, qy, qz)
    qw, qx, qy, qz = list(new_rot)
    pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]
    return pose

class EndEffectorPoseViaPlanning2(EndEffectorPoseViaPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def action(self, scene: Scene, action: np.ndarray):
        #self._collision_checking
        if action[0]>10:#需要进行避障处理
            action=copy.deepcopy(action)
            action[0]-=20
            ignore_collisions=False
            assert_action_shape(action, (7,))
            assert_unit_quaternion(action[3:])
            if not self._absolute_mode and self._frame != 'end effector':
                action = calculate_delta_pose(scene.robot, action)
            relative_to = None if self._frame == 'world' else scene.robot.arm.get_tip()
            self._quick_boundary_check(scene, action)

            colliding_shapes = []
            if not ignore_collisions:
                if self._robot_shapes is None:
                    self._robot_shapes = scene.robot.arm.get_objects_in_tree(
                        object_type=ObjectType.SHAPE)
                # First check if we are colliding with anything
                colliding = scene.robot.arm.check_arm_collision()
                if colliding:
                    # Disable collisions with the objects that we are colliding with
                    grasped_objects = scene.robot.gripper.get_grasped_objects()
                    colliding_shapes = [
                        s for s in scene.pyrep.get_objects_in_tree(
                            object_type=ObjectType.SHAPE) if (
                                s.is_collidable() and
                                s not in self._robot_shapes and
                                s not in grasped_objects and
                                scene.robot.arm.check_arm_collision(
                                    s))]
                    [s.set_collidable(False) for s in colliding_shapes]

            try:
                path = scene.robot.arm.get_path(
                    action[:3],
                    quaternion=action[3:],
                    ignore_collisions=not ignore_collisions,
                    relative_to=relative_to,
                    trials=100,
                    max_configs=10,
                    max_time_ms=10,
                    trials_per_goal=5,
                    algorithm=Algos.RRTConnect
                )
                [s.set_collidable(True) for s in colliding_shapes]
            except ConfigurationPathError as e:
                print("Could not find a path avoiding collisions, "
                    "trying to find one ignoring collisions.")
                try:
                    path = scene.robot.arm.get_path(
                        action[:3],
                        quaternion=action[3:],
                        ignore_collisions=True,
                        relative_to=relative_to,
                        trials=100,
                        max_configs=10,
                        max_time_ms=10,
                        trials_per_goal=5,
                        algorithm=Algos.RRTConnect
                    )
                    [s.set_collidable(True) for s in colliding_shapes]
                except ConfigurationPathError as e:
                    [s.set_collidable(True) for s in colliding_shapes]
                    raise InvalidActionError(
                        'A path could not be found. Most likely due to the target '
                        'being inaccessible or a collison was detected.') from e
            observations = []

            done = False
            while not done:
                done = path.step()
                scene.step()

                if self._callable_each_step is not None:
                    # Record observations
                    self._callable_each_step(scene.get_observation())

                # DEBUG
                # observations.append(scene.get_observation())

                success, terminate = scene.task.success()
                # If the task succeeds while traversing path, then break early
                if success and self._callable_each_step is None:
                    break
        else:
            super().action(scene, action)
