#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <stdexcept>
#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/wrench.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "viz.h" // Use correct relative path if needed, assuming include dir is standard

class Sim_node : public rclcpp::Node {
public:
    Sim_node(std::string model_path, double sim_timestep, bool enable_visualization) 
        : Node("sim"), 
          sim_timestep_(sim_timestep),
          enable_visualization_(enable_visualization) {
        
        if (enable_visualization_) { initializeGLFW(); }
        
        initializeMujoco(model_path);
        initializeSimulation();
        initializeROSInterfaces();
    }

    ~Sim_node() {
        cleanup();
    }

private:
    mjModel* model_{nullptr};
    mjData* data_{nullptr};
    int ee_body_id_;
    std::mutex mutex_;
 
    mjvScene scene_;
    mjvCamera camera_;
    mjvOption options_;
    mjrContext context_;
    
    GLFWwindow* window_{nullptr};
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr ee_pos_pub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr controller_sub_;
    rclcpp::TimerBase::SharedPtr sim_timer_;

    std::unique_ptr<sensor_msgs::msg::JointState> state_msg_;
    std::unique_ptr<geometry_msgs::msg::PointStamped> ee_pos_msg_;
    
    double sim_timestep_;
    rclcpp::Time current_sim_time_;
    rclcpp::Clock::SharedPtr sim_clock_;
    
    std::vector<std::string> joint_names_{"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
    
    // External force parameters
    bool apply_external_force_{false};
    std::vector<double> external_force_{0.0, 0.0, 0.0};
    rclcpp::Subscription<geometry_msgs::msg::Wrench>::SharedPtr external_force_sub_;
    
    // Visualization flag
    bool enable_visualization_;
    
    // Flag to track if a command has been received
    bool command_received_{false};

    void initializeGLFW() {
        RCLCPP_INFO(get_logger(), "Initializing GLFW");
        if (!glfwInit()) { throw std::runtime_error("Failed to initialize GLFW"); }
        glfwSetErrorCallback(VisualizationHelpers::glfw_error_callback);
        try {
            window_ = glfwCreateWindow(800, 600, "MuJoCo Simulator", nullptr, nullptr);
            if (!window_) { throw std::runtime_error("Failed to create GLFW window"); }
            glfwMakeContextCurrent(window_);
            glfwSwapInterval(1);
            glfwSetMouseButtonCallback(window_, VisualizationHelpers::mouse_button_callback);
            glfwSetCursorPosCallback(window_, VisualizationHelpers::cursor_position_callback);
            glfwSetKeyCallback(window_, VisualizationHelpers::keyboard_callback);
            glfwSetScrollCallback(window_, VisualizationHelpers::scroll_callback);
        } catch (const std::runtime_error& e) {
            glfwTerminate();
            throw;
        }
    }

    void initializeMujoco(std::string model_path) {
        RCLCPP_INFO(get_logger(), "loading MuJoCo model from %s", model_path.c_str());
        char error[1000];
        model_ = mj_loadXML(model_path.c_str(), nullptr, error, 1000);
        if (!model_) {
            RCLCPP_ERROR(get_logger(), "Failed to load model: %s", error);
            throw std::runtime_error("Model loading failed");
        }
        // disable gravity
        // model_->opt.gravity[0], model_->opt.gravity[1], model_->opt.gravity[2] = 0, 0, 0;
        data_ = mj_makeData(model_);

        std::vector<double> initial_positions = {1.57994932,  0.06313131, -1.18071498,  1.09272369, -0.6255293,  -0.01895007}; // TODO: make a parameter
        for (size_t i = 0; i < initial_positions.size(); ++i) {
            data_->qpos[i] = initial_positions[i];
        }
        mj_forward(model_, data_);
        ee_body_id_ = model_->nbody - 1;
    }

    void initializeSimulation() {
        mjv_defaultScene(&scene_);
        mjv_defaultOption(&options_);
        mjr_defaultContext(&context_);
        
        if (enable_visualization_) {
            mjv_makeScene(model_, &scene_, 2000);
            mjr_makeContext(model_, &context_, mjFONTSCALE_150);
            mjv_defaultCamera(&camera_);
            camera_.type = mjCAMERA_FREE;
            camera_.distance = VisualizationHelpers::CameraParams::distance;
            camera_.azimuth = VisualizationHelpers::CameraParams::azimuth;
            camera_.elevation = VisualizationHelpers::CameraParams::elevation;
        }
        
        model_->opt.timestep = sim_timestep_;
        sim_clock_ = std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME);
        current_sim_time_ = sim_clock_->now();
    }

    void initializeROSInterfaces() {
        state_msg_ = std::make_unique<sensor_msgs::msg::JointState>();
        state_msg_->name = joint_names_;
        state_msg_->position.resize(model_->nv);
        state_msg_->velocity.resize(model_->nv);
        state_msg_->effort.resize(model_->nv);

        ee_pos_msg_ = std::make_unique<geometry_msgs::msg::PointStamped>();
        ee_pos_msg_->header.frame_id = "world";

        // State//ee_pos publishers
        joint_state_pub_ = create_publisher<sensor_msgs::msg::JointState>("joint_states", 1);
        ee_pos_pub_ = create_publisher<geometry_msgs::msg::PointStamped>("ee_position", 1);

        // Controls subscription
        controller_sub_ = create_subscription<sensor_msgs::msg::JointState>("joint_commands", 1,
            std::bind(&Sim_node::controllerCallback, this, std::placeholders::_1)
        );
        
        // External force subscription
        external_force_sub_ = create_subscription<geometry_msgs::msg::Wrench>(
            "external_force", 1,
            std::bind(&Sim_node::externalForceCallback, this, std::placeholders::_1)
        );

        // Simulation step timer
        sim_timer_ = create_wall_timer(
            std::chrono::duration<double>(sim_timestep_),
            std::bind(&Sim_node::simulationStepCallback, this)
        );

        this->set_parameter(rclcpp::Parameter("use_sim_time", true));
        RCLCPP_INFO(get_logger(), "MuJoCo simulation node initialized with timestep: %f", sim_timestep_);

    }

    void cleanup() {
        if (data_) mj_deleteData(data_);
        if (model_) mj_deleteModel(model_);
        
        if (enable_visualization_) {
            mjv_freeScene(&scene_);
            mjr_freeContext(&context_);
            if (window_) glfwDestroyWindow(window_);
            glfwTerminate();
        }
    }

    void simulationStepCallback() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Only step the simulation if a command has been received
        if (command_received_) {
            mj_step(model_, data_);
        } else { // If no command received yet, just forward the model to update visualization
            mj_forward(model_, data_);
        }
        
        publish();  // publish joint states and end effector position
        
        if (enable_visualization_) {
            updateVisualization();
        }
    }

    // get new commands from controller
    void controllerCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (msg->effort.size() != 6) {
            RCLCPP_WARN(get_logger(), "Received incorrect control command, expected 6 joint torques");
            return;
        }
        std::copy_n(msg->effort.begin(), msg->effort.size(), data_->ctrl);

        // RCLCPP_INFO(get_logger(), "Received controls: %.4f %.4f %.4f %.4f %.4f %.4f",
        //             msg->effort[0], msg->effort[1], msg->effort[2], msg->effort[3], msg->effort[4], msg->effort[5]);

        command_received_ = true;
    }


    // update external force
    void externalForceCallback(const geometry_msgs::msg::Wrench::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        external_force_[0] = msg->force.x;
        external_force_[1] = msg->force.y;
        external_force_[2] = msg->force.z;
        apply_external_force_ = true;
        
        RCLCPP_INFO(rclcpp::get_logger("sim"), "Received external force: [%f, %f, %f]",
                    external_force_[0], external_force_[1], external_force_[2]);

        applyExternalForce();
    }


    // Runs every simulation step to publish joint states and end effector position
    void publish() {
        // Update timestamps
        current_sim_time_ += rclcpp::Duration::from_seconds(sim_timestep_);
        state_msg_->header.stamp = ee_pos_msg_->header.stamp = current_sim_time_;

        // Copy joint state data
        for (unsigned int i = 0; i < model_->nv; i++) {
            state_msg_->position[i] = data_->qpos[i];
            state_msg_->velocity[i] = data_->qvel[i];
            state_msg_->effort[i] = data_->ctrl[i];
        }
        
        // Copy end effector position
        const double* ee_pos = data_->xpos + 3 * ee_body_id_;
        ee_pos_msg_->point.x = ee_pos[0];
        ee_pos_msg_->point.y = ee_pos[1];
        ee_pos_msg_->point.z = ee_pos[2];
        
        // Publish messages
        joint_state_pub_->publish(*state_msg_);
        ee_pos_pub_->publish(*ee_pos_msg_);
    }

    void applyExternalForce() {
        // Get end effector position in world frame
        mjtNum pos[3];
        mju_copy3(pos, data_->xpos + 3 * ee_body_id_);
        
        // Convert force from world frame to local frame
        mjtNum force[3] = {external_force_[0], external_force_[1], external_force_[2]};
        mjtNum force_local[3];
        mju_mulMatVec(force_local, data_->xmat + 9 * ee_body_id_, force, 3, 3);
        
        mjtNum torque[3] = {0, 0, 0}; // No torque applied
        mj_applyFT(model_, data_, force_local, torque, pos, ee_body_id_, data_->qfrc_applied);

        RCLCPP_INFO(rclcpp::get_logger("sim"), "Applied force: %.4f %.4f %.4f on end effector!", force[0], force[1], force[2]);
    }


    void updateVisualization() {
        camera_.azimuth = VisualizationHelpers::CameraParams::azimuth;
        camera_.elevation = VisualizationHelpers::CameraParams::elevation;
        camera_.distance = VisualizationHelpers::CameraParams::distance;

        int width, height;
        glfwGetFramebufferSize(window_, &width, &height);

        mjrRect viewport = {0, 0, width, height};
        mjv_updateScene(model_, data_, &options_, nullptr, &camera_, mjCAT_ALL, &scene_);
        mjr_render(viewport, &scene_, &context_);

        glfwSwapBuffers(window_);
        glfwPollEvents();
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    
    // Parse command line arguments
    if (argc < 3) {
        RCLCPP_ERROR(rclcpp::get_logger("sim"), 
                     "Usage: %s <model_path> <sim_timestep> [enable_visualization]", argv[0]);
        return 1;
    }
    
    std::string model_path = argv[1];
    double sim_timestep = std::stod(argv[2]);
    bool enable_visualization = true; // Default to visualization enabled
    
    // Check if visualization flag is provided
    if (argc > 3) {
        std::string vis_flag = argv[3];
        enable_visualization = (vis_flag == "true" || vis_flag == "1");
    }
    
    RCLCPP_INFO(rclcpp::get_logger("sim"), 
                "Starting MuJoCo simulation with visualization: %s", 
                enable_visualization ? "enabled" : "disabled");
    
    auto node = std::make_shared<Sim_node>(model_path, sim_timestep, enable_visualization);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
