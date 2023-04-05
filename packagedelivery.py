
from __future__ import print_function

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


def find_package(world_object_client):
    """ Detects nearby package fiducial and navigates to it."""

    request_fiducials = [world_object_pb2.WORLD_OBJECT_APRILTAG]
    current_fiducials = world_object_client.list_world_objects(object_type=request_fiducials).world_objects
    print(current_fiducials)
    return

def pickup_package():
    return

def walk_to_destination():
    return

def find_dropoff():
    return

def deliver_package():
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

    # Create the world object client.
    world_object_client = robot.ensure_client(WorldObjectClient.default_service_name)
    find_package(world_object_client)

    return

if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)