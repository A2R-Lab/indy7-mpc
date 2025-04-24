cmake_minimum_required(VERSION 3.10)
project(mujoco_sim)

# Set OpenGL policy to prefer GLVND
cmake_policy(SET CMP0072 NEW)
set(OpenGL_GL_PREFERENCE GLVND)

# # Option to enable/disable visualization
# option(ENABLE_VISUALIZATION "Enable MuJoCo visualization with GLFW" ON)

# # Set environment variable for package.xml conditions
# set(ENV{ENABLE_VISUALIZATION} ${ENABLE_VISUALIZATION})

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(mujoco REQUIRED HINTS /usr/local)
find_package(glfw3 REQUIRED)


add_executable(sim_node src/sim_node.cpp)
target_compile_features(sim_node PUBLIC c_std_99 cxx_std_17) 

target_include_directories(sim_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(sim_node
  mujoco::mujoco
  glfw
)

ament_target_dependencies(sim_node
  rclcpp
  sensor_msgs
  geometry_msgs
)

install(TARGETS sim_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
