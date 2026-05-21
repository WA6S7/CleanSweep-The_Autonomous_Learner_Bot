cmake_minimum_required(VERSION 3.8)
project(clean_sweep_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

install(
  PROGRAMS
    scripts/random_obstacles_spawner_gz_with_robot.py
    scripts/coverage_executor_gazebo.py
    scripts/coverage_map_visualiser.py
    scripts/coverage_heatmap_visualiser.py
    scripts/ekf_particle_visualiser.py
    scripts/ground_truth_publisher.py
    scripts/coverage_monitor_hw.py
    scripts/cmd_vel_odom_node.py
    scripts/aruco_localizer_node.py
    scripts/coverage_executor_gazebo_ppo.py
    scripts/serial_bridge_node.py
    scripts/set_initial_pose.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config worlds
  DESTINATION share/${PROJECT_NAME}/
)


ament_package()
