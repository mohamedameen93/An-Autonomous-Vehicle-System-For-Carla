#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import numpy as np
from scipy.spatial import KDTree

import math


'''
This node will publish waypoints from the car's current position to some `x`
distance ahead.

As mentioned in the doc, you should ideally first implement a version which
does not care about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of
traffic lights too.

Please note that our simulator also provides the exact location of traffic lights
and their current status in `/vehicle/traffic_lights` message. You can use this
message to build this node as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

# Number of waypoints we will publish. You can change this number
# NOTE: it GREATLY impacts the performance
LOOKAHEAD_WPS = 20

# TODO(MD): how to choose the best MAX_DECEL?
MAX_DECEL = 0.5


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_waypoint_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Add other member variables you need below
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopline_wp_idx = -1

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_lane and self.waypoint_tree:
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        # Equation for hyperplane through `closest_coords`
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def publish_waypoints(self, closest_waypoint_idx):
        final_lane = self._gen_lane(closest_waypoint_idx)
        self.final_waypoints_pub.publish(final_lane)

    def _gen_lane(self, closest_idx):
        lane = Lane()
        diff = len(self.waypoints_2d)
        lane.header = self.base_waypoints.header

        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]

        if farthest_idx < diff:
            base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]
        else:
            base_waypoints = self.base_lane.waypoints[closest_idx:int(diff)] + self.base_lane.waypoints[0:int(farthest_idx % diff)]

        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self._decelerate_waypoints(base_waypoints, closest_idx)

        return lane

    def _decelerate_waypoints(self, base_waypoints, closest_idx):
        temp = []
        for i, wp in enumerate(base_waypoints):
            p = Waypoint()
            p.pose = wp.pose

            # Two base_waypoints back from line so front of car stops at line
            # (-2 is from the center of the car)
            stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0)
            dist = self.distance(base_waypoints, i, stop_idx)
            # TODO: consider a different function than sqare root
            vel = np.power(MAX_DECEL * dist * 7, 0.33) * 1.7
            if vel < 1.0:
                vel = 0

            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)

        return temp

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        self.base_lane = waypoints
        if not self.waypoints_2d:  # Check this in order to avoid a race condition
            self.waypoints_2d = [
                [waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
                for waypoint in waypoints.waypoints
            ]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_waypoint_cb(self, msg):
        self.stopline_wp_idx = msg.data

    def get_waypoint_velocity(self, waypoint):
        """
        Gets the linear velocity (x-direction) for a single `waypoint`.
        """
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        """
        Sets the linear velocity (x-direction) for a single waypoint in a list
        of waypoints.

        Here, `waypoints` is a list of waypoints, `waypoint` is a waypoint index
        in the list, and velocity is the desired velocity.
        """
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        """
        Computes the distance between two waypoints in a list along the piecewise
        linear arc connecting all waypoints between the two.

        Here, `waypoints` is a list of waypoints, and `wp1` and `wp2` are the
        indices of two waypoints in the list.

        This method may be helpful in determining the velocities for a sequence
        of waypoints leading up to a red light (the velocities should gradually
        decrease to zero starting some distance from the light).
        """
        dist = 0
        for i in range(wp1+1, wp2+1):
            dist += self.dist_fn(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    @staticmethod
    def dist_fn(pos_a, pos_b):
        return math.sqrt((pos_a.x-pos_b.x)**2 + (pos_a.y-pos_b.y)**2 + (pos_a.z-pos_b.z)**2)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
