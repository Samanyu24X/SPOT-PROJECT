
from __future__ import print_function
import logging
import math
import signal
import sys
import threading
from sys import platform

import cv2
import numpy as np
from PIL import Image

import bosdyn.client
import bosdyn.client.util
from bosdyn import geometry
from bosdyn.api import geometry_pb2, image_pb2, trajectory_pb2, world_object_pb2
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.frame_helpers import (BODY_FRAME_NAME, VISION_FRAME_NAME, get_a_tform_b,
                                         get_vision_tform_body)
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.lease import LeaseClient
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.power import PowerClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_id import RobotIdClient, version_tuple
from bosdyn.client.robot_state import RobotStateClient

import argparse
import sys
import time

import bosdyn.client
import bosdyn.client.util
from bosdyn.api import geometry_pb2 as geom
from bosdyn.api import world_object_pb2
from bosdyn.client.frame_helpers import *
from bosdyn.client.world_object import (WorldObjectClient, make_add_world_object_req,
                                        make_change_world_object_req, make_delete_world_object_req)
from bosdyn.util import now_timestamp

BODY_LENGTH = 1.1


class PackageDelivery(object):

    def __init__(self, robot):
        self._robot = robot
        self._robot_id = robot.ensure_client(RobotIdClient.default_service_name).get_id(timeout=0.4)
        self._power_client = robot.ensure_client(PowerClient.default_service_name)
        self._image_client = robot.ensure_client(ImageClient.default_service_name)
        self._robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        self._robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        self._world_object_client = robot.ensure_client(WorldObjectClient.default_service_name)


        self._current_tag_world_pose = np.array([])
        self._angle_desired = None

        self._tag_offset = .2 + BODY_LENGTH / 2.0  # meters

        # Maximum speeds.
        self._max_x_vel = 0.5
        self._max_y_vel = 0.5
        self._max_ang_vel = 1.0



    def get_package_fiducial(self):
        """Detects nearby package fiducial from world object service and returns it."""

        # search for nearby fiducials for specified waiting time
        waiting_time = 10
        print("Looking for package fiducial...")

        for x in range (0, waiting_time):
            request_fiducials = [world_object_pb2.WORLD_OBJECT_APRILTAG]
            current_fiducials = self._world_object_client.list_world_objects(object_type=request_fiducials).world_objects

            # loop through all detected fiducials and return the specified bag fiducial, if found
            for x in current_fiducials:
                if x.name == "world_obj_apriltag_523":
                    return x
            time.sleep(1)
        return None
        
    
    def go_to_package(self, package):
        """Takes in a package fiducial and calls the go_to_tag() method passing in the position of the fiducial"""
        fiducial_rt_world = None
        vision_tform_fiducial = get_a_tform_b(
            package.transforms_snapshot, VISION_FRAME_NAME,
            package.apriltag_properties.frame_name_fiducial).to_proto()
        if vision_tform_fiducial is not None:
            fiducial_rt_world = vision_tform_fiducial.position
            print("Calling go_to_tag()...")
            self.go_to_tag(fiducial_rt_world)
            return 0
        return 1
        


    def go_to_tag(self, fiducial_rt_world):
        """Use the position of the april tag in vision world frame and command the robot."""
        # Compute the go-to point (offset by .5m from the fiducial position) and the heading at
        # this point.
        self._current_tag_world_pose, self._angle_desired = self.offset_tag_pose(
            fiducial_rt_world, self._tag_offset)

        #Command the robot to go to the tag in kinematic odometry frame
        mobility_params = self.set_mobility_params()
        tag_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=self._current_tag_world_pose[0], goal_y=self._current_tag_world_pose[1],
            goal_heading=self._angle_desired, frame_name=VISION_FRAME_NAME, params=mobility_params,
            body_height=0.0, locomotion_hint=spot_command_pb2.HINT_AUTO)
        end_time = 5.0

        # not using this if condition to issue command and feedback loop, since we aren't using these boolean attributes
        #if self._movement_on and self._powered_on:

        #Issue the command to the robot
        print("Issuing move command...")
        self._robot_command_client.robot_command(lease=None, command=tag_cmd,
                                                     end_time_secs=time.time() + end_time)
            # #Feedback to check and wait until the robot is in the desired position or timeout
        start_time = time.time()
        current_time = time.time()
        while (not self.final_state() and current_time - start_time < end_time):
            time.sleep(.25)
            current_time = time.time()
        return
    
    def get_desired_angle(self, xhat):
        """Compute heading based on the vector from robot to object."""
        zhat = [0.0, 0.0, 1.0]
        yhat = np.cross(zhat, xhat)
        mat = np.array([xhat, yhat, zhat]).transpose()
        return Quat.from_matrix(mat).to_yaw()

    def offset_tag_pose(self, object_rt_world, dist_margin=1.0):
        """Offset the go-to location of the fiducial and compute the desired heading."""
        robot_rt_world = get_vision_tform_body(self.robot_state.kinematic_state.transforms_snapshot)
        robot_to_object_ewrt_world = np.array(
            [object_rt_world.x - robot_rt_world.x, object_rt_world.y - robot_rt_world.y, 0])
        robot_to_object_ewrt_world_norm = robot_to_object_ewrt_world / np.linalg.norm(
            robot_to_object_ewrt_world)
        heading = self.get_desired_angle(robot_to_object_ewrt_world_norm)
        goto_rt_world = np.array([
            object_rt_world.x - robot_to_object_ewrt_world_norm[0] * dist_margin,
            object_rt_world.y - robot_to_object_ewrt_world_norm[1] * dist_margin
        ])
        return goto_rt_world, heading
    
    def set_mobility_params(self):
        """Set robot mobility params to disable obstacle avoidance."""

        # this obstacles list of params is used to specify certain obstacle behaviors.
        # since we are keeping all avoidance turned on, we might not need this line

        obstacles = spot_command_pb2.ObstacleParams(disable_vision_body_obstacle_avoidance=False,
                                                    disable_vision_foot_obstacle_avoidance=False,
                                                    disable_vision_foot_constraint_avoidance=False,
                                                    obstacle_avoidance_padding=.001)
        body_control = self.set_default_body_control()

        # original code from SDK method which sets params based on obstacle booleans and speed limit
        # we are straight up leaving all avoidances turned on so no need for this
        '''
        if self._limit_speed:
            speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(
                linear=Vec2(x=self._max_x_vel, y=self._max_y_vel), angular=self._max_ang_vel))
            if not self._avoid_obstacles:
                mobility_params = spot_command_pb2.MobilityParams(
                    obstacle_params=obstacles, vel_limit=speed_limit, body_control=body_control,
                    locomotion_hint=spot_command_pb2.HINT_AUTO)
            else:
                mobility_params = spot_command_pb2.MobilityParams(
                    vel_limit=speed_limit, body_control=body_control,
                    locomotion_hint=spot_command_pb2.HINT_AUTO)
        elif not self._avoid_obstacles:
            mobility_params = spot_command_pb2.MobilityParams(
                obstacle_params=obstacles, body_control=body_control,
                locomotion_hint=spot_command_pb2.HINT_AUTO)
        else:
            #When set to none, RobotCommandBuilder populates with good default values
            mobility_params = None
        '''

        # replacement code
        speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(linear=Vec2(x=self._max_x_vel, y=self._max_y_vel), angular=self._max_ang_vel))
        mobility_params = spot_command_pb2.MobilityParams(
                    vel_limit=speed_limit, body_control=body_control,
                    locomotion_hint=spot_command_pb2.HINT_AUTO)

        return mobility_params
    
    @staticmethod
    def set_default_body_control():
        """Set default body control params to current body position"""
        footprint_R_body = geometry.EulerZXY()
        position = geometry_pb2.Vec3(x=0.0, y=0.0, z=0.0)
        rotation = footprint_R_body.to_quaternion()
        pose = geometry_pb2.SE3Pose(position=position, rotation=rotation)
        point = trajectory_pb2.SE3TrajectoryPoint(pose=pose)
        traj = trajectory_pb2.SE3Trajectory(points=[point])
        return spot_command_pb2.BodyControlParams(base_offset_rt_footprint=traj)
    
    
    def final_state(self):
        """Check if the current robot state is within range of the fiducial position."""
        robot_state = get_vision_tform_body(self.robot_state.kinematic_state.transforms_snapshot)
        robot_angle = robot_state.rot.to_yaw()
        if self._current_tag_world_pose.size != 0:
            x_dist = abs(self._current_tag_world_pose[0] - robot_state.x)
            y_dist = abs(self._current_tag_world_pose[1] - robot_state.y)
            angle = abs(self._angle_desired - robot_angle)
            if ((x_dist < self._x_eps) and (y_dist < self._y_eps) and (angle < self._angle_eps)):
                return True
        return False

    def pickup_package(self):
        return

    def walk_to_destination(self):
        return

    def find_dropoff(self):
        return

    def deliver_package(self):
        return
    
    def start(self):

        print("Powering on...")
        self.power_on()
        blocking_stand(self._robot_command_client)
        time.sleep(1)

        package_fiducial = self.get_package_fiducial()
        if package_fiducial is None:
            print("Could not identify the package fiducial.")
            self.power_off()
        if self.go_to_package(package_fiducial) == 1:
            print("Could not get the position of the package fiducial.")
            self.power_off()

        return
    
    def power_on(self):
        """Power on the robot."""
        self._robot.power_on()
        #self._powered_on = True
        #print("Powered On " + str(self._robot.is_powered_on()))

    def power_off(self):
        """Power off the robot."""
        self._robot.power_off()
        #print("Powered Off " + str(not self._robot.is_powered_on()))

def main(argv):
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)

    # Create robot object with a package delivery client.
    sdk = bosdyn.client.create_standard_sdk('PackageDeliveryClient')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)

    # Time sync is necessary so that time-based filter requests can be converted.
    robot.time_sync.wait_for_sync()

    # create an instance of our package delivery robot
    package_delivery_robot = PackageDelivery(robot)
    time.sleep(.1)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        package_delivery_robot.start()


    return

if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)