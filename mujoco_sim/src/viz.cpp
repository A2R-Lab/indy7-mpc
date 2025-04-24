#include "viz.h"
#include <iostream> // For std::cerr
#include <algorithm> // For std::clamp

namespace viz {

    void glfw_error_callback(int error, const char* description) {
        std::cerr << "GLFW Error " << error << ": " << description << std::endl;
    }

    void keyboard_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
    }

    void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            MouseState::drag = (action == GLFW_PRESS);
        }
    }

    void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
        if (MouseState::drag) {
            CameraParams::azimuth -= (xpos - MouseState::last_x) * CameraParams::sensitivity;
            CameraParams::elevation -= (ypos - MouseState::last_y) * CameraParams::sensitivity;

            // Clamp elevation to prevent camera flipping
            CameraParams::elevation = std::clamp(CameraParams::elevation, -90.0f, 90.0f);
        }
        MouseState::last_x = xpos;
        MouseState::last_y = ypos;
    }

    void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
        CameraParams::distance *= 1.0f - yoffset * 0.1f;
    }

} // namespace viz 