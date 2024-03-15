#!/usr/bin/env python3
import time

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest

mav_state = State()


def state_cb(msg):
  """ MAV  State Callback """
  global mav_state
  mav_state = msg


if __name__ == "__main__":
  rospy.init_node("offb_node_py")

  sub_state = rospy.Subscriber("mavros/state", State, callback=state_cb)
  pub_pos = rospy.Publisher("/mavros/setpoint_position/local",
                            PoseStamped,
                            queue_size=10)

  arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
  set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
  rospy.wait_for_service("/mavros/cmd/arming")
  rospy.wait_for_service("/mavros/set_mode")

  # Setpoint publishing MUST be faster than 2Hz
  rate = rospy.Rate(20)

  # Wait for Flight Controller connection
  while rospy.is_shutdown() is False and mav_state.connected is False:
    rate.sleep()
  rospy.loginfo("Connected!")

  # Send a few setpoints before starting
  pose = PoseStamped()
  pose.pose.position.x = 0
  pose.pose.position.y = 0
  pose.pose.position.z = 2
  for i in range(10):
    if rospy.is_shutdown():
      break
    pub_pos.publish(pose)
    rate.sleep()

  # Offboard mode
  offb_set_mode = SetModeRequest()
  offb_set_mode.custom_mode = 'OFFBOARD'
  set_mode_client.call(offb_set_mode)
  rospy.loginfo("Sent offboard mode!")
  rate.sleep()

  # Arm Command
  arm_cmd = CommandBoolRequest()
  arm_cmd.value = True
  arming_client.call(arm_cmd)
  rospy.loginfo("Sent ARM command!")
  rate.sleep()

  # Keep publishing pose setpoint
  rospy.loginfo("Sending position setpoint!")
  while (not rospy.is_shutdown()):
    pub_pos.publish(pose)
    rate.sleep()
