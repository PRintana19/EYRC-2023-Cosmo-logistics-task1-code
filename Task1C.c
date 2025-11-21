#! /usr/bin/env python3

import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

def main():
    rclpy.init()

    nav = BasicNavigator()

    poses = [
        [1.8, 1.5, 1.57],
        [2.0, -7.0, -1.57],
        [-3.0, 2.5, 1.57]
    ]

    for pose in poses:
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.position.x = pose[0]
        goal_pose.pose.position.y = pose[1]

        quaternion_goal = Quaternion()
        quaternion_goal.x = 0.0
        quaternion_goal.y = 0.0
        quaternion_goal.z = pose[2]
        quaternion_goal.w = 0.0

        nav.goToPose(goal_pose)

        while not nav.isTaskComplete():
            feedback = nav.getFeedback()

        result = nav.getResult()
        if result == TaskResult.SUCCEEDED:
            print('Goal succeeded!')
        elif result == TaskResult.CANCELED:
            print('Goal was canceled!')
        elif result == TaskResult.FAILED:
            print('Goal failed!')

    nav.lifecycleShutdown()
    exit(0)

if __name__ == '__main__':
    main()

