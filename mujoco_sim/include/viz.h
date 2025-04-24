#pragma once
#include <GLFW/glfw3.h>

namespace viz {

    struct CameraParams {
        static inline float azimuth = 60.0f;
        static inline float elevation = -45.0f;
        static inline float distance = 2.5f;
        static inline const float sensitivity = 0.5f;
    };

    struct MouseState {
        static inline bool drag = false;
        static inline double last_x = 0.0;
        static inline double last_y = 0.0;
    };

    // GLFW callback functions
    void glfw_error_callback(int error, const char* description);
    void keyboard_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
    void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

} // namespace viz