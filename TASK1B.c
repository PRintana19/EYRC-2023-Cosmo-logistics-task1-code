#!/usr/bin/env python3

from threading import Thread
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from pymoveit2 import MoveIt2
from pymoveit2.robots import ur5


def main():
    rclpy.init()

    # Create node for this example
    node = Node("move_arm_node")

    # Declare parameters for positions and orientations
    positions = {
        "P1": {
            "position": [0.35, 0.1, 0.68],
            "quat_xyzw": [0.5, 0.5, 0.5, 0.5]
        },
        "D": {
            "position": [-0.37, 0.12, 0.397],
            "quat_xyzw": [0.5, -0.5, -0.5, 0.5]
        },
        "P2": {
            "position": [0.194, -0.43, 0.701],
            "quat_xyzw": [0.7071, 0.0, 0.0, 0.7071]
        },
    }

    # Create callback group that allows execution of callbacks in parallel without restrictions
    callback_group = ReentrantCallbackGroup()

    # Create MoveIt 2 interface
    moveit2 = MoveIt2(
        node=node,
        joint_names=ur5.joint_names(),
        base_link_name=ur5.base_link_name(),
        end_effector_name=ur5.end_effector_name(),
        group_name=ur5.MOVE_GROUP_ARM,
        callback_group=callback_group,
    )

    # Spin the node in background thread(s)
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # Move the arm in the specified order
    for target, info in positions.items():
        position = info["position"]
        quat_xyzw = info["quat_xyzw"]
        node.get_logger().info(f"Moving to {target} position: {position}, orientation: {quat_xyzw}")
        moveit2.move_to_pose(position=position, quat_xyzw=quat_xyzw, cartesian=True)
        moveit2.wait_until_executed()

    # Move to drop position again
    node.get_logger().info(f"Moving to D position again: {positions['D']['position']}")
    moveit2.move_to_pose(position=positions['D']['position'], quat_xyzw=positions['D']['quat_xyzw'], cartesian=True)
    moveit2.wait_until_executed()

    rclpy.shutdown()
    exit(0)


if __name__ == "__main__":
    main()

