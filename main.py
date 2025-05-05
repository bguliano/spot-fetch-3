import time

import bosdyn.client
import bosdyn.client.util
import numpy as np
from bosdyn.api import geometry_pb2, manipulation_api_pb2
from bosdyn.client import frame_helpers, math_helpers
from bosdyn.client.lease import LeaseClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.math_helpers import SE3Pose
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient, block_for_trajectory_cmd,
                                         block_until_arm_arrives)
from bosdyn.client.robot_state import RobotStateClient
from spot_tools.common import get_walking_params
from spot_tools.network_compute_client import NetworkComputeClient, InferenceObject
from spot_tools.network_compute_server import NetworkComputeServer, DirectoryServiceRegistration
from spot_tools.spot import Spot

PULL_THRESHOLD = 0.003


class FetchApp:
    def __init__(self, spot: Spot):
        self.spot = spot

        # configure ML service and client
        service_name = 'bg-spot-fetch-3'
        self._ncs = NetworkComputeServer(
            spot=spot,
            registration=DirectoryServiceRegistration(
                name=service_name
            ),
            models_path='models'
        )
        self._ncs.wait_for_initial_connection()
        self._ncc = NetworkComputeClient(
            spot=spot,
            service_name=service_name
        )

        # allow ncc to show annotated images
        self._ncc.enable_showing_annotated_images('Fetch 3')

    def get_hand_pose(self) -> SE3Pose:
        robot_state = self.spot.clients[RobotStateClient].get_robot_state()
        hand_pos = frame_helpers.get_a_tform_b(
            robot_state.kinematic_state.transforms_snapshot,
            frame_helpers.VISION_FRAME_NAME,
            'hand'
        )
        return hand_pos

    def wait_for_hand_detection(self, time_between_checks: float = 0.1):
        original_hand_pose = self.get_hand_pose()

        while True:
            hand_pose = self.get_hand_pose()
            print(f'Hand height: {hand_pose.z - original_hand_pose.z}')
            if hand_pose.z - original_hand_pose.z < -PULL_THRESHOLD:  # pulled
                break
            time.sleep(time_between_checks)

    def wait_for_grasp(self, grasp_request: manipulation_api_pb2.ManipulationApiRequest) -> bool:
        # grab necessary clients
        manipulation_api_client = self.spot.clients[ManipulationApiClient]

        # first, send the request
        print('Sending grasp request...')
        cmd_response = manipulation_api_client.manipulation_api_command(
            manipulation_api_request=grasp_request
        )

        # then, poll the status of the request and wait for completion
        start_time = time.time()
        while True:
            # Send a request for feedback
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id
            )
            response = manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request
            )

            current_state = response.current_state
            current_state_name = manipulation_api_pb2.ManipulationFeedbackState.Name(current_state)
            print(f'Current state ({time.time() - start_time:.1f} sec): {current_state_name}', end='\r', flush=True)

            failed_states = [
                manipulation_api_pb2.MANIP_STATE_GRASP_FAILED,
                manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION,
                manipulation_api_pb2.MANIP_STATE_GRASP_FAILED_TO_RAYCAST_INTO_MAP,
                manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_WAITING_DATA_AT_EDGE
            ]

            no_timeout_states = [
                manipulation_api_pb2.MANIP_STATE_MOVING_TO_GRASP,
                manipulation_api_pb2.MANIP_STATE_ATTEMPTING_RAYCASTING
            ]

            # check success
            if current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
                print('\nGrasp success')
                return True

            # check failures
            if current_state in failed_states:
                print('\nGrasp failed - general failure')
                return False
            elif time.time() - start_time > 5.0 and current_state == manipulation_api_pb2.MANIP_STATE_SEARCHING_FOR_GRASP:
                print('\nGrasp failed - timeout while searching')
                return False
            elif time.time() - start_time > 10.0 and current_state not in no_timeout_states:
                print('\nGrasp failed - primary timeout')
                return False
            elif time.time() - start_time > 15.0:
                print('\nGrasp failed - secondary timeout')
                return False

            # no checks passed, continue looping
            time.sleep(0.1)

    def is_holding_object(self, minimum_grip: float) -> bool:
        # grab necessary clients
        robot_state_client = self.spot.clients[RobotStateClient]

        # request robot state and look in the manipulator_state section
        robot_state = robot_state_client.get_robot_state()
        grip_amount = robot_state.manipulator_state.gripper_open_percentage / 100
        print(f'Gripper at {grip_amount:.2f}, {minimum_grip=}')

        # True if the gripper is more open than the input minimum
        return grip_amount >= minimum_grip

    def is_holding(self) -> bool:
        # grab necessary clients
        robot_state_client = self.spot.clients[RobotStateClient]

        # get robot state and look in the manipulator_state section
        robot_state = robot_state_client.get_robot_state()
        return robot_state.manipulator_state.is_gripper_holding_item

    def request_arm_stow(self, blocking: bool = True):
        # grab necessary clients
        command_client = self.spot.clients[RobotCommandClient]

        if blocking:
            print('Blocking stow requested...', end='', flush=True)
        else:
            print('Non-blocking stow requested...', end='', flush=True)

        # if robot is holding something, it must drop it before it can stow its arm
        if self.is_holding():
            print('Releasing object...', end='', flush=True)
            open_gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
            command_client.robot_command(open_gripper_command)
            time.sleep(0.8)

        # create actual command
        stow_cmd = RobotCommandBuilder.arm_stow_command()
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(0.0)
        synchro_command = RobotCommandBuilder.build_synchro_command(gripper_command, stow_cmd)

        # execute command, blocking if requested
        cmd_id = command_client.robot_command(synchro_command)
        if blocking:
            block_until_arm_arrives(command_client, cmd_id)
        print('Finished')

    def compute_stand_location_and_yaw(self, vision_tform_target: SE3Pose, distance_margin: float):
        # grab necessary clients
        robot_state_client = self.spot.clients[RobotStateClient]

        # Compute drop-off location:
        #   Draw a line from Spot to the person
        #   Back up 2.0 meters on that line
        vision_tform_robot = frame_helpers.get_a_tform_b(
            robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
            frame_helpers.VISION_FRAME_NAME,
            frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME
        )

        # Compute vector between robot and person
        robot_rt_person_ewrt_vision = [
            vision_tform_robot.x - vision_tform_target.x,
            vision_tform_robot.y - vision_tform_target.y,
            vision_tform_robot.z - vision_tform_target.z
        ]

        # Compute the unit vector.
        if np.linalg.norm(robot_rt_person_ewrt_vision) < 0.01:
            robot_rt_person_ewrt_vision_hat = vision_tform_robot.transform_point(1, 0, 0)
        else:
            robot_rt_person_ewrt_vision_hat = robot_rt_person_ewrt_vision / np.linalg.norm(robot_rt_person_ewrt_vision)

        # Starting at the person, back up meters along the unit vector.
        drop_position_rt_vision = [
            vision_tform_target.x + robot_rt_person_ewrt_vision_hat[0] * distance_margin,
            vision_tform_target.y + robot_rt_person_ewrt_vision_hat[1] * distance_margin,
            vision_tform_target.z + robot_rt_person_ewrt_vision_hat[2] * distance_margin
        ]

        # We also want to compute a rotation (yaw) so that we will face the person when dropping.
        # We'll do this by computing a rotation matrix with X along
        #   -robot_rt_person_ewrt_vision_hat (pointing from the robot to the person) and Z straight up:
        xhat = -robot_rt_person_ewrt_vision_hat
        zhat = [0.0, 0.0, 1.0]
        yhat = np.cross(zhat, xhat)
        mat = np.matrix([xhat, yhat, zhat]).transpose()
        heading_rt_vision = math_helpers.Quat.from_matrix(mat).to_yaw()

        return drop_position_rt_vision, heading_rt_vision

    def run(self):
        robot_state_client = self.spot.clients[RobotStateClient]
        command_client = self.spot.clients[RobotCommandClient]
        lease_client = self.spot.clients[LeaseClient]

        # This script assumes the robot is already standing via the tablet.  We'll take over from the tablet.
        lease_client.take()
        with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            # Stow the arm in case it is deployed
            self.request_arm_stow()

            # main loop
            while True:
                # find and pick up object loop
                num_grasp_failures = 0
                print(f'Reset num_grasp_failures, {num_grasp_failures=}')
                while True:
                    # if num_grasp_failures is too high, attempt to reset the arm before trying again
                    if num_grasp_failures >= 2:
                        print(f'{num_grasp_failures=}, resetting arm')
                        self.request_arm_stow()

                    # run 360 inspection (capture from all cameras)
                    paper_inference_results = self._ncc.perform_360_inspection(
                        'spot-fetch-2-4',
                        color_image=False,
                        whitelist_labels=['paper']
                    )

                    # analyze results to find the closest paper
                    if paper := paper_inference_results.get_first('paper') is None:
                        continue

                    # Got a paper.  Request pick up.
                    print('Found paper...')

                    # Request Pick Up on that pixel.
                    grasp = manipulation_api_pb2.PickObjectInImage(
                        pixel_xy=paper.bounding_box.center.as_vec2(),
                        transforms_snapshot_for_camera=paper.image_response.shot.transforms_snapshot,
                        frame_name_image_sensor=paper.image_response.shot.frame_name_image_sensor,
                        camera_model=paper.image_response.source.pinhole
                    )

                    # We can specify where in the gripper we want to grasp. About halfway is generally good for
                    # small objects like this. For a bigger object like a shoe, 0 is better (use the entire
                    # gripper)
                    grasp.grasp_params.grasp_palm_to_fingertip = 0.7

                    # Tell the grasping system that we want a top-down grasp.

                    # Add a constraint that requests that the x-axis of the gripper is pointing in the
                    # negative-z direction in the vision frame.

                    # The axis on the gripper is the x-axis.
                    axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

                    # The axis in the vision frame is the negative z-axis
                    axis_to_align_with_ewrt_vision = geometry_pb2.Vec3(x=0, y=0, z=-1)

                    # Add the vector constraint to our proto.
                    constraint = grasp.grasp_params.allowable_orientation.add()
                    constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
                        axis_on_gripper_ewrt_gripper)
                    constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
                        axis_to_align_with_ewrt_vision)

                    # We'll take anything within about 15 degrees for top-down or horizontal grasps.
                    constraint.vector_alignment_with_tolerance.threshold_radians = 0.25

                    # Specify the frame we're using.
                    grasp.grasp_params.grasp_params_frame_name = frame_helpers.VISION_FRAME_NAME

                    # Build the proto
                    grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

                    # attempt grasp
                    grasp_success = self.wait_for_grasp(grasp_request)
                    if not grasp_success:
                        num_grasp_failures += 1
                        continue

                    # if successful, then make sure the object is actually in the gripper
                    holding_success = self.is_holding_object(0.03)
                    if not holding_success:
                        # since we're having the robot find the object again, make sure the arm is out of the way
                        num_grasp_failures += 5
                        continue

                    # if both checks are passed, break from loop to continue fetch sequence
                    break

                # Move the arm to a carry position.
                print('Grasp finished, search for a person...')
                carry_cmd = RobotCommandBuilder.arm_carry_command()
                command_client.robot_command(carry_cmd)

                # Wait for the carry command to finish
                time.sleep(0.75)

                person: InferenceObject | None = None
                while not person:
                    # run 360 inspection (capture from all cameras)
                    person_inference_results = self._ncc.perform_360_inspection(
                        'yolov8n',
                        whitelist_labels=['person']
                    )

                    # find the closest person to Spot
                    person = person_inference_results.get_closest('person')

                # We now have found a person to drop the toy off near.
                drop_position_rt_vision, heading_rt_vision = self.compute_stand_location_and_yaw(
                    person.vision_tform_obj,
                    distance_margin=1.0
                )

                wait_position_rt_vision, wait_heading_rt_vision = self.compute_stand_location_and_yaw(
                    person.vision_tform_obj,
                    distance_margin=2.0
                )

                # Tell the robot to go there
                # Limit the speed so we don't charge at the person.
                se2_pose = geometry_pb2.SE2Pose(
                    position=geometry_pb2.Vec2(
                        x=drop_position_rt_vision[0],
                        y=drop_position_rt_vision[1]
                    ),
                    angle=heading_rt_vision
                )
                move_cmd = RobotCommandBuilder.synchro_se2_trajectory_command(
                    se2_pose,
                    frame_name=frame_helpers.VISION_FRAME_NAME,
                    params=get_walking_params(0.5, 0.5)
                )
                end_time = 5.0
                cmd_id = command_client.robot_command(
                    command=move_cmd,
                    end_time_secs=time.time() + end_time
                )

                # Wait until the robot reports that it is at the goal.
                block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=6)

                print('Arrived at goal, dropping object...')

                # Do an arm-move to gently put the object down.
                # Build a position to move the arm to
                # (in meters, relative to and expressed in the gravity aligned body frame)
                x = 0.75
                y = 0
                z = 0.4
                hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)

                # Point the hand straight down with a quaternion.
                qw = 0.707
                qx = 0
                qy = 0.707
                qz = 0
                flat_body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

                flat_body_tform_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body,
                                                            rotation=flat_body_Q_hand)

                robot_state = robot_state_client.get_robot_state()
                vision_tform_flat_body = frame_helpers.get_a_tform_b(
                    robot_state.kinematic_state.transforms_snapshot, frame_helpers.VISION_FRAME_NAME,
                    frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME)

                vision_tform_hand_at_drop = vision_tform_flat_body * math_helpers.SE3Pose.from_proto(
                    flat_body_tform_hand)

                # duration in seconds
                seconds = 1

                arm_command = RobotCommandBuilder.arm_pose_command(
                    vision_tform_hand_at_drop.x, vision_tform_hand_at_drop.y,
                    vision_tform_hand_at_drop.z, vision_tform_hand_at_drop.rot.w,
                    vision_tform_hand_at_drop.rot.x, vision_tform_hand_at_drop.rot.y,
                    vision_tform_hand_at_drop.rot.z, frame_helpers.VISION_FRAME_NAME, seconds)

                # Keep the gripper closed.
                gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(0.0)

                # Combine the arm and gripper commands into one RobotCommand
                command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)

                # Send the request
                cmd_id = command_client.robot_command(command)

                # Wait until the arm arrives at the goal.
                block_until_arm_arrives(command_client, cmd_id)

                # wait until spot detects a hand underneath its gripper to release the paper
                self.wait_for_hand_detection()

                # Open the gripper
                gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
                command = RobotCommandBuilder.build_synchro_command(gripper_command)
                command_client.robot_command(command)

                # Wait for the paper to fall out
                time.sleep(1.5)

                # Stow the arm.
                # non-blocking allows spot to move to next location while still stowing the arm
                self.request_arm_stow(blocking=False)

                time.sleep(1)

                print('Backing up and waiting...')

                # Back up one meter and wait for the person to throw the object again.
                se2_pose = geometry_pb2.SE2Pose(
                    position=geometry_pb2.Vec2(x=wait_position_rt_vision[0],
                                               y=wait_position_rt_vision[1]),
                    angle=wait_heading_rt_vision)
                move_cmd = RobotCommandBuilder.synchro_se2_trajectory_command(
                    se2_pose, frame_name=frame_helpers.VISION_FRAME_NAME,
                    params=get_walking_params(0.5, 0.5))
                end_time = 5.0
                cmd_id = command_client.robot_command(command=move_cmd,
                                                      end_time_secs=time.time() + end_time)

                # Wait until the robot reports that it is at the goal.
                block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=5)


if __name__ == '__main__':
    spot = Spot(authentication_file='authentication.json')
    fetch_app = FetchApp(spot)
    fetch_app.run()
