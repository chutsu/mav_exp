#include <map>
#include <string>
#include <csignal>

#include <ros/ros.h>
#include <ros/console.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/transform_broadcaster.h>

#include "DataStreamClient.h"

namespace vicon = ViconDataStreamSDK::CPP;
using PoseStamped = geometry_msgs::PoseStamped;
using TF2Stamped = geometry_msgs::TransformStamped;

struct ViconNode {
  vicon::Client client;
  ros::NodeHandle nh;

  std::string hostname = "10.0.5.127";
  size_t buffer_size = 200;
  std::string ns_name = "vicon";
  std::map<std::string, ros::Publisher> pub_pose_map;
  tf2_ros::TransformBroadcaster tf_broadcaster;

  ViconNode() { std::signal(SIGINT, &ViconNode::signal_handler); }

  static void signal_handler(int signum) { ros::shutdown(); }

  bool connect() {
    // Connect to server
    ROS_INFO("Connecting to [%s] ...", hostname.c_str());
    const int num_retries = 5;
    for (int i = 0; i < num_retries; i++) {
      if (client.IsConnected().Connected) {
        break;
      }
      if (client.Connect(hostname).Result != vicon::Result::Success) {
        ROS_WARN("Failed to connect, retrying ...");
        sleep(1);
      }
    }

    // Check connection
    if (client.Connect(hostname).Result != vicon::Result::Success) {
      ROS_ERROR("Failed to connect to vicon");
      return false;
    } else {
      ROS_INFO("Connected!");
    }

    // Perform further initialization
    client.EnableSegmentData();
    client.EnableMarkerData();
    client.EnableUnlabeledMarkerData();
    client.EnableMarkerRayData();
    client.EnableDeviceData();
    client.EnableDebugData();
    client.SetStreamMode(vicon::StreamMode::ClientPull);
    client.SetBufferSize(buffer_size);

    return true;
  }

  bool disconnect() {
    if (!client.IsConnected().Connected) {
      return true;
    }

    sleep(1);
    client.DisableSegmentData();
    client.DisableMarkerData();
    client.DisableUnlabeledMarkerData();
    client.DisableDeviceData();
    client.DisableCentroidData();

    ROS_INFO("Disconnecting from [%s] ...", hostname.c_str());
    client.Disconnect();

    if (!client.IsConnected().Connected) {
      ROS_INFO("Successfully disconnected!");
      return true;
    }

    ROS_ERROR("Failed to disconnect!");
    return false;
  }

  void get_frame() {
    client.GetFrame();
    const size_t sub_count = client.GetSubjectCount().SubjectCount;

    for (size_t sub_index = 0; sub_index < sub_count; ++sub_index) {
      const std::string sub_name = client.GetSubjectName(sub_index).SubjectName;
      const auto seg_count = client.GetSegmentCount(sub_name).SegmentCount;

      for (size_t seg_index = 0; seg_index < seg_count; ++seg_index) {
        const std::string seg_name =
            client.GetSegmentName(sub_name, seg_index).SegmentName;
        publish(sub_name, seg_name);
      }
    }
  }

  void publish(const std::string &sub_name, const std::string &seg_name) {
    const std::string topic_name = sub_name + "/" + seg_name;
    if (pub_pose_map.count(topic_name)) {
      // clang-format off
      const auto timestamp = ros::Time::now();
      const auto trans = client.GetSegmentGlobalTranslation(sub_name, seg_name);
      const auto pos_x = trans.Translation[0] * 1e-3;  // Convert mm to m
      const auto pos_y = trans.Translation[1] * 1e-3;  // Convert mm to m
      const auto pos_z = trans.Translation[2] * 1e-3;  // Convert mm to m
      const auto rot = client.GetSegmentGlobalRotationQuaternion(sub_name, seg_name);
      const auto qx = rot.Rotation[0];
      const auto qy = rot.Rotation[1];
      const auto qz = rot.Rotation[2];
      const auto qw = rot.Rotation[3];
      // clang-format on

      // Publish pose stamped message
      PoseStamped msg;
      msg.header.frame_id = sub_name + "/" + seg_name;
      msg.header.stamp = timestamp;
      msg.pose.position.x = pos_x;
      msg.pose.position.y = pos_y;
      msg.pose.position.z = pos_z;
      msg.pose.orientation.x = qx;
      msg.pose.orientation.y = qy;
      msg.pose.orientation.z = qz;
      msg.pose.orientation.w = qw;
      pub_pose_map[topic_name].publish(msg);

      // Publish pose stamped message
      TF2Stamped tf2_msg;
      tf2_msg.header.frame_id = "map";
      tf2_msg.child_frame_id = sub_name + "/" + seg_name;
      tf2_msg.header.stamp = timestamp;
      tf2_msg.transform.translation.x = pos_x;
      tf2_msg.transform.translation.y = pos_y;
      tf2_msg.transform.translation.z = pos_z;
      tf2_msg.transform.rotation.x = qx;
      tf2_msg.transform.rotation.y = qy;
      tf2_msg.transform.rotation.z = qz;
      tf2_msg.transform.rotation.w = qw;
      tf_broadcaster.sendTransform(tf2_msg);

    } else if (pub_pose_map.count(topic_name) == 0) {
      const std::string topic_name = ns_name + "/" + sub_name + "/" + seg_name;
      const std::string key = sub_name + "/" + seg_name;
      ROS_INFO("Creating publisher [%s]", key.c_str());
      pub_pose_map[key] = nh.advertise<PoseStamped>(topic_name, 1);
    }
  }
};

int main(int argc, char **argv) {
  // Initialize ROS Node
  ros::init(argc, argv, "vicon_node");

  // Connect and loop Vicon Node
  ViconNode node;
  node.connect();
  while (ros::ok()) {
    node.get_frame();
  }

  // Disconnect
  node.disconnect();

  return 0;
}
