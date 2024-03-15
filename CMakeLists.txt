CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(mav_exp)

# DEPENDENCIES
FIND_PACKAGE(catkin REQUIRED COMPONENTS roscpp rospy tf2 tf2_ros std_msgs geometry_msgs)

# VICON SDK
INCLUDE_DIRECTORIES(vicon_datastream_sdk/Release/Linux64)
LINK_DIRECTORIES(vicon_datastream_sdk/Release/Linux64)

# Set executable rpath to point to <colcon_ws>/install/lib/ros2_vicon for the dlls
# This approach avoids having to install the library to /usr/local/lib or /usr/lib
SET(VICON_LIBS
  vicon_datastream_sdk/Release/Linux64/libboost_atomic-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_chrono-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_container-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_context-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_contract-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_coroutine-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_date_time-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_fiber-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_filesystem-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_graph-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_json-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_locale-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_log-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_log_setup-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_math_c99-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_math_c99f-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_math_c99l-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_math_tr1-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_math_tr1f-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_math_tr1l-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_nowide-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_prg_exec_monitor-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_program_options-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_random-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_regex-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_serialization-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_stacktrace_addr2line-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_stacktrace_backtrace-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_stacktrace_basic-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_stacktrace_noop-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_system-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_thread-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_timer-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_type_erasure-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_unit_test_framework-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_wave-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libboost_wserialization-mt-x64.so.1.75.0
  vicon_datastream_sdk/Release/Linux64/libViconDataStreamSDK_C.so
  vicon_datastream_sdk/Release/Linux64/libViconDataStreamSDK_CPP.so
  vicon_datastream_sdk/Release/Linux64/libboost_atomic-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_chrono-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_container-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_context-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_contract-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_coroutine-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_date_time-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_fiber-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_filesystem-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_graph-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_json-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_locale-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_log-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_log_setup-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_math_c99-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_math_c99f-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_math_c99l-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_math_tr1-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_math_tr1f-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_math_tr1l-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_nowide-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_prg_exec_monitor-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_program_options-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_random-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_regex-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_serialization-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_stacktrace_addr2line-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_stacktrace_backtrace-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_stacktrace_basic-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_stacktrace_noop-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_system-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_thread-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_timer-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_type_erasure-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_unit_test_framework-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_wave-mt-x64.so
  vicon_datastream_sdk/Release/Linux64/libboost_wserialization-mt-x64.so
)
FILE(COPY ${VICON_LIBS} DESTINATION lib/${PROJECT_NAME})
SET(CMAKE_EXE_LINKER_FLAGS
  "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,${CMAKE_BINARY_DIR}/lib/${PROJECT_NAME}")

# CATKIN
CATKIN_PACKAGE(CATKIN_DEPENDS roscpp geometry_msgs)
CATKIN_INSTALL_PYTHON(
  PROGRAMS src/mav_node.py src/offb_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# VICON NODE
INCLUDE_DIRECTORIES(${catkin_INCLUDE_DIRS})
ADD_EXECUTABLE(vicon_node src/vicon_node.cpp)
TARGET_LINK_LIBRARIES(vicon_node ${catkin_LIBRARIES} ViconDataStreamSDK_CPP)
INSTALL(TARGETS vicon_node DESTINATION lib/${PROJECT_NAME})

# INSTALL
INSTALL(
  DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
INSTALL(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
