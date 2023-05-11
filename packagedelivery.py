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
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
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

# new imports added when merging pickup.py code
import bosdyn.client.estop
import bosdyn.client.lease
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, math_helpers
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient

from bosdyn.client.robot import Robot

BODY_LENGTH = 1.1

g_image_click = None
g_image_display = None


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

        self._tag_offset = 0.3 + BODY_LENGTH / 2.0  # meters

        # Maximum speeds.
        self._max_x_vel = 0.5
        self._max_y_vel = 0.5
        self._max_ang_vel = 1.0

        # Epsilon distance between robot and desired go-to point.
        self._x_eps = .05
        self._y_eps = .05
        self._angle_eps = .075


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
                if x.name == "world_obj_apriltag_544":
                    return x
            time.sleep(1)
        return None
    
    def get_delivery_fiducial(self):
        """Detects nearby package fiducial from world object service and returns it."""

        # search for nearby fiducials for specified waiting time
        waiting_time = 10
        print("Looking for delivery fiducial...")

        for x in range (0, waiting_time):
            request_fiducials = [world_object_pb2.WORLD_OBJECT_APRILTAG]
            current_fiducials = self._world_object_client.list_world_objects(object_type=request_fiducials).world_objects

            # loop through all detected fiducials and return the specified delivery fiducial, if found
            for x in current_fiducials:
                if x.name == "world_obj_apriltag_545":
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
    
    def go_to_delivery(self, delivery):
        """Takes in a package fiducial and calls the go_to_tag() method passing in the position of the fiducial"""
        fiducial_rt_world = None
        vision_tform_fiducial = get_a_tform_b(
            delivery.transforms_snapshot, VISION_FRAME_NAME,
            delivery.apriltag_properties.frame_name_fiducial).to_proto()
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
        print("Move command was issued")
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
    
    @property
    def robot_state(self):
        """Get latest robot state proto."""
        return self._robot_state_client.get_robot_state()
    
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

    def pickup_package(self, package_fiducial):

        # to avoid having to go back to the command line, hardcode the options string
        # this argstring effectively replaces "argv" used in the pickup.py file
        argstring = ['192.168.80.3','-t']
        # adjusted code from pickup.py main() that configures the options
        parser = argparse.ArgumentParser()
        bosdyn.client.util.add_base_arguments(parser)
        img_src=str(package_fiducial.image_properties.camera_source)+"_image"
        print(img_src)
        #switched it to default to frontleft since that is what the fiducial uses and we are basing coord off of
        parser.add_argument('-i', '--image-source', help='Get image from source',
                        default=img_src)
        parser.add_argument('-t', '--force-top-down-grasp',
                        help='Force the robot to use a top-down grasp (vector_alignment demo)',
                        action='store_true')
        options = parser.parse_args(argstring)
        
        '''
        # i commented it out since we are only using one grasp, no need to have this still check for erors
        #Keep to test if SPOT will continue to utilize other grasps
        num = 0
        if options.force_top_down_grasp:
            num += 1
        if options.force_horizontal_grasp:
            num += 1
        if options.force_45_angle_grasp:
            num += 1
        if options.force_squeeze_grasp:
            num += 1
        if num > 1:
            print("Error: cannot force other grasp besides top down.  Choose only one.")
            sys.exit(1)
        '''
        # boolean that chooses if we use the click pickup or automated pickup
        # True = click pickup (Neil), false = automated (Francisco)
        manual = False


        '''
        potnetial method we can use: bosdyn.client.image.pixel_to_camera_space(image_proto, pixel_x, pixel_y, depth=1.0)
            Using the camera intrinsics, determine the (x,y,z) point in the camera frame for the (u,v) pixel coordinates.
        
            Note that the front left and front right cameras on Spot are rotated 90 degrees counterclockwise from upright, 
            and the right camera on Spot is rotated 180 degrees from upright. As a result, the corresponding images aren’t 
            initially saved with the same orientation as is seen on the tablet. By adding the command line argument --auto-rotate, 
            this example code automatically rotates all images from Spot to be saved in the orientation they are seen on the tablet screen
        '''

        try:
            if manual:

                arm_object_grasp(options, self._robot, package_fiducial) # returns True once the operation was completed
            else:
                print(package_fiducial)
                image_client=self._image_client
                image_responses = image_client.get_image_from_sources([options.image_source])
                print(image_responses[0])
                arm_object_grasp_with_coordinates(options, package_fiducial,  self._robot, image_responses)
            return True
        except Exception as exc:  # pylint: disable=broad-except
            logger = bosdyn.client.util.get_logger()
            logger.exception("Threw an exception")
            return False # returns false if there was an exception


    def walk_to_destination(self):
        return

    def stow_package(self):
        stow = RobotCommandBuilder.arm_stow_command()

        # Issue the command via the RobotCommandClient
        command_client = self._robot.ensure_client(RobotCommandClient.default_service_name)
        print("issuing stow command")
        stow_command_id = command_client.robot_command(stow)

        self._robot.logger.info("Stow command issued.")
        #block_until_arm_arrives(command_client, stow_command_id, 3.0)
    
    def find_dropoff(self):
        return

    def deliver_package(self):
        return
    
    def start(self):

        # power on and stand
        print("Powering on...")
        self.power_on()
        blocking_stand(self._robot_command_client)
        time.sleep(1)

        # go to package
        package_fiducial = self.get_package_fiducial()
        if package_fiducial is None:
            print("Could not identify the package fiducial.")
            self.power_off()
        if self.go_to_package(package_fiducial) == 1:
            print("Could not get the position of the package fiducial.")
            self.power_off()
        print("Moved to package fiducial")

        # get package fiducial again now that it's right in front of SPOT
        package_fiducial = self.get_package_fiducial()

        # pickup package
        # get package fiducial again now that it's right in front of SPOT
        package_fiducial = self.get_package_fiducial()
        print(package_fiducial)
        if self.pickup_package(package_fiducial) is True:
            print("Completed arm grasp")
            self.stow_package()
            time.sleep(4.0)



        # go to delivery
        delivery_fiducial = self.get_delivery_fiducial()
        if delivery_fiducial is None:
            print("Could not identify the delivery fiducial.")
            self.power_off()
        if self.go_to_delivery(delivery_fiducial) == 1:
            print("Could not get the position of the delivery fiducial.")
            self.power_off()
        print("Moved to delivery fiducial")

        # get delivery fiducial again
        delivery_fiducial = self.get_delivery_fiducial()

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



# START new pickup.py code

def verify_estop(robot):
    """Verify the robot is not estopped"""

    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = "Robot is estopped. Please use an external E-Stop client, such as the" \
        " estop SDK example, to configure E-Stop."
        robot.logger.error(error_message)
        raise Exception(error_message)
    
def getCoordinates(worldObj):
    

    image_info = worldObj.image_properties

    vertices = []

    for vertex in image_info.coordinates.vertexes:
        v = []
        v.append(vertex.x)
        v.append(vertex.y)
        vertices.append(v)

    vertex1 = vertices[-1]
    vertex2 = vertices[-2]

    vertex3=vertices[0]
    vertex4=vertices[1]


    smaller=min(vertex1[0], vertex3[0])

    x = int(abs(vertex1[0] - vertex3[0]))
    y = int(vertex2[1] - 5)
    return x,y

def arm_object_grasp(config, robot,fiducial):
    """A simple example of using the Boston Dynamics API to command Spot's arm."""
    x_coor,y_coor= getCoordinates(worldObj=fiducial)
    print("coordiantes detected by the fiducial (x,y)",x_coor, y_coor)
    print(fiducial.image_properties.coordinates)


    # See hello_spot.py for an explanation of these lines.
    bosdyn.client.util.setup_logging(config.verbose)

    sdk = bosdyn.client.create_standard_sdk('ArmObjectGraspClient')
    #robot = sdk.create_robot(config.hostname)
    #bosdyn.client.util.authenticate(robot)
    #robot.time_sync.wait_for_sync()

    assert robot.has_arm(), "Robot requires an arm to run this example."

    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    verify_estop(robot)

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)

    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=False):
        # Now, we are ready to power on the robot. This call will block until the power
        # is on. Commands would fail if this did not happen. We can also check that the robot is
        # powered at any point.
        robot.logger.info("Powering on robot... This may take a several seconds.")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."
        robot.logger.info("Robot powered on.")

        # Tell the robot to stand up. The command service is used to issue commands to a robot.
        # The set of valid commands for a robot depends on hardware configuration. See
        # RobotCommandBuilder for more detailed examples on command building. The robot
        # command service requires timesync between the robot and the client.
        robot.logger.info("Commanding robot to stand...")
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")

        # Take a picture with a camera
        robot.logger.info('Getting an image from: ' + config.image_source)
        image_responses = image_client.get_image_from_sources([config.image_source])

        if len(image_responses) != 1:
            print('Got invalid number of images: ' + str(len(image_responses)))
            print(image_responses)
            assert False

        image = image_responses[0]
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            dtype = np.uint16
        else:
            dtype = np.uint8
        #this manipulation might give me something i dont entirely want 
        img = np.fromstring(image.shot.image.data, dtype=dtype)
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            img = img.reshape(image.shot.image.rows, image.shot.image.cols)
        else:
            img = cv2.imdecode(img, -1)

        # Show the image to the user and wait for them to click on a pixel
        robot.logger.info('Click on an object to start grasping...')
        image_title = 'Click to grasp'
        cv2.namedWindow(image_title)
        cv2.setMouseCallback(image_title, cv_mouse_callback)

        global g_image_click, g_image_display
        g_image_display = img
        cv2.imshow(image_title, g_image_display)
        while g_image_click is None:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                # Quit
                print('"q" pressed, exiting.')
                exit(0)

        robot.logger.info('Picking object at image location (' + str(g_image_click[0]) + ', ' +
                          str(g_image_click[1]) + ')')

        pick_vec = geometry_pb2.Vec2(x=g_image_click[0], y=g_image_click[1])
        print("the vector that gets produced by the click: ",pick_vec)

        # Build the proto
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec, transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=image.source.pinhole)

        # Optionally add a grasp constraint.  This lets you tell the robot you only want top-down grasps or side-on grasps.
        add_grasp_constraint(config, grasp, robot_state_client)

        # Ask the robot to pick up the object
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

        # Send the request
        cmd_response = manipulation_api_client.manipulation_api_command(
            manipulation_api_request=grasp_request)

        # Get feedback from the robot
        while True:
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)

            # Send the request
            response = manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)

            print('Current state: ',
                  manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state))

            if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED or response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                break

            time.sleep(0.25)

        robot.logger.info('Finished grasp.')
        time.sleep(4.0)

        #robot.logger.info('Sitting down and turning off.')

        # Power the robot off. By specifying "cut_immediately=False", a safe power off command
        # is issued to the robot. This will attempt to sit the robot before powering off.
        #robot.power_off(cut_immediately=False, timeout_sec=20)
        #assert not robot.is_powered_on(), "Robot power off failed."
        #robot.logger.info("Robot safely powered off.")

def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        # Draw some lines on the image.
        #print('mouse', x, y)
        color = (30, 30, 30)
        thickness = 2
        image_title = 'Click to grasp'
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, clone)

def add_grasp_constraint(config, grasp, robot_state_client):
    #Reduced options to top_down_grasp due to bag handle position
    #Current Model will only use said grasp to obtain packages

    # For these options, we'll use a vector alignment constraint.
    use_vector_constraint = config.force_top_down_grasp or config.force_horizontal_grasp

    # Specify the frame we're using.
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

    if use_vector_constraint:
        if config.force_top_down_grasp:
            # Add a constraint that requests that the x-axis of the gripper is pointing in the
            # negative-z direction in the vision frame.

            # The axis on the gripper is the x-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

            # The axis in the vision frame is the negative z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)

        # lines that Neil deleted
        # Add the vector constraint to our proto.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
            axis_on_gripper_ewrt_gripper)
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
            axis_to_align_with_ewrt_vo)

        # We'll take anything within about 10 degrees for top-down or horizontal grasps.
        constraint.vector_alignment_with_tolerance.threshold_radians = 0.17

def arm_object_grasp_with_coordinates(config, worldObj, bot:Robot, imageResponses):
    """A simple example of using the Boston Dynamics API to command Spot's arm."""
    #bosdyn.client.util.setup_logging(config.verbose)
    print(imageResponses)
    y_coor, x_coor= getCoordinates(worldObj=worldObj)
    robot = bot
    #bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), "Robot requires an arm to run this example."

    verify_estop(robot)

    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

    
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    #we would need the image client and the image response to be initated prior to this part of the code 
    image_responses = imageResponses
    assert robot.is_powered_on(), "Robot power on failed."
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    blocking_stand(command_client, timeout_sec=10)
    if len(image_responses) != 1:
        print('Got invalid number of images: ' + str(len(image_responses)))
        print(image_responses)
        assert False

    image = image_responses[0]

    pick_vec = geometry_pb2.Vec2(x=x_coor, y=y_coor)
    print(pick_vec)

    # Build the proto
    grasp = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=pick_vec, transforms_snapshot_for_camera=image.shot.transforms_snapshot,
        frame_name_image_sensor=image.shot.frame_name_image_sensor,
        camera_model=image.source.pinhole)

    # Optionally add a grasp constraint.  This lets you tell the robot you only want top-down grasps or side-on grasps.
    add_grasp_constraint(config, grasp, robot_state_client)

    # Ask the robot to pick up the object
    grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

    # Send the request
    cmd_response = manipulation_api_client.manipulation_api_command(
        manipulation_api_request=grasp_request)

    # Get feedback from the robot
    while True:
        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_response.manipulation_cmd_id)

        # Send the request
        response = manipulation_api_client.manipulation_api_feedback_command(
            manipulation_api_feedback_request=feedback_request)

        print('Current state: ',
                manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state))

        if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED or response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
            break

        time.sleep(0.25)

    robot.logger.info('Finished grasp.')
    time.sleep(4.0)

        #robot.logger.info('Sitting down and turning off.')

        # Power the robot off. By specifying "cut_immediately=False", a safe power off command
        # is issued to the robot. This will attempt to sit the robot before powering off.
        #robot.power_off(cut_immediately=False, timeout_sec=20)
        #assert not robot.is_powered_on(), "Robot power off failed."
        #robot.logger.info("Robot safely powered off.")
    return 


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