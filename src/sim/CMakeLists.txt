cmake_minimum_required(VERSION 3.8)
project(sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

cmake_policy(SET CMP0072 NEW)
#set(OpenGL_GL_PREFERENCE GLVND)
find_package(ament_cmake REQUIRED)
find_package(mujoco REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
option(ENABLE_VISUALIZATION "Enable MuJoCo visualization with GLFW" ON)
set(ENV{ENABLE_VISUALIZATION} ${ENABLE_VISUALIZATION})

if(ENABLE_VISUALIZATION)
  find_package(glfw3 REQUIRED)
  add_definitions(-DENABLE_VISUALIZATION)
endif()

add_executable(sim_node sim/sim_node.cpp)
target_compile_features(sim_node PUBLIC c_std_99 cxx_std_17) 

target_include_directories(sim_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(sim_node
  mujoco::mujoco
)

# Only link GLFW if visualization is enabled
if(ENABLE_VISUALIZATION)
  target_link_libraries(
    sim_node
    glfw
  )
endif()

ament_target_dependencies(
  sim_node
  rclcpp
  sensor_msgs
)

install(TARGETS sim_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
