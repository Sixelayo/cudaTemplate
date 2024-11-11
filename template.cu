#include "util.hpp"
#define TITLE "template"

/* FIXME add readme here*/

struct Param {
    float scale;
    float mx, my;
};
Param h_params;
/*
 You can send params to GPU with :
 __constant__ Param d_params;
 memory if you need to have per-frame constant parameter common to every pixel instead of passing them as function argument. 
 You'll need to copy h_params to d_params each frame
 Be mindful whith memory access to d_params in device functions !
 */


namespace cpu{
    //forward declaration ...
    void example();

    void init(){
        gbl::pixels = (float4*)malloc(gbl::SCREEN_X*gbl::SCREEN_Y*sizeof(float4));
        gbl::display = example;
    }
    void clean(){
        free(gbl::pixels);
    }
    void reinit(){
        gbl::pixels = (float4*)realloc(gbl::pixels, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(float4));
    }

    void example(){
        int i, j;
        for (i = 0; i<gbl::SCREEN_Y; i++){
            for (j = 0; j<gbl::SCREEN_X; j++){
                float x = (float)(h_params.scale*(j - gbl::SCREEN_X / 2));
                float y = (float)(h_params.scale*(i - gbl::SCREEN_Y / 2));
                float4* p = gbl::pixels + (i*gbl::SCREEN_X + j);
                // default: black
                p->x = 0.0f;
                p->y = 0.0f;
                p->z = 0.0f;
                p->w = 1.0f;
                if (sqrt((x - h_params.mx)*(x - h_params.mx) + (y - h_params.my)*(y - h_params.my))<0.01)
                    p->x = 1.0f;
                else if ((i == gbl::SCREEN_Y / 2) || (j == gbl::SCREEN_X / 2))
                {
                    p->x = 1.0f;
                    p->y = 1.0f;
                    p->z = 1.0f;
                }
            }
        }
    }

}//end namespace cpu

namespace gpu{ //replace with your own code
    //forward declaration ...
    void imp_Julia();

    void init(){
        gbl::pixels = (float4*)malloc(gbl::SCREEN_X*gbl::SCREEN_Y*sizeof(float4));
        gbl::display = cpu::example;
    }
    void reinit(){
        gbl::pixels = (float4*)realloc(gbl::pixels, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(float4));
    }
    void clean(){
        free(gbl::pixels);
    }
    
}//end namespace gpu

void clean(){
	switch (gbl::mode){
        case CPU_MODE: gpu::clean(); break;
        case GPU_MODE: cpu::clean(); break;
	}
}
void init(){
	switch (gbl::mode){
        case CPU_MODE: cpu::init(); break;
        case GPU_MODE: gpu::init(); break;
	}
}
void reinit(){
	switch (gbl::mode){
        case CPU_MODE: cpu::reinit(); break;
        case GPU_MODE: gpu::reinit(); break;
	}
}



namespace cbk{ 
    /*various callback
    You must ALWAYS forward the event to ImGui before processing it (except window resizing)
    You can find relevant ImGui callback in ./imgui/imgui_impl_glfw.cpp line 536 in function ImGui_ImplGlfw_InstallCallbacks
    */

   void key(GLFWwindow* window, int key, int scancode, int action, int mods){
        // Forward the event to ImGui
        ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);


        //if ImGui doesn't want the event, process it
        ImGuiIO& io = ImGui::GetIO();
        if(!io.WantCaptureKeyboard){
            /* uses US keyboard layout ! https://www.glfw.org/docs/latest/group__keys.html
            use charCallback if you want to avoid translation qwerty->azerty*/
            if (key == GLFW_KEY_Z && action == GLFW_PRESS){ //match W in azerty
                gbl::otherWindow = !gbl::otherWindow;
            }
        }
    }

    inline void updt_mpos(double xpos, double ypos){
        h_params.mx = (float)(h_params.scale*(xpos - gbl::SCREEN_X / 2));
        h_params.my = - (float)(h_params.scale*(ypos - gbl::SCREEN_Y / 2));
    }

    void mouse_button(GLFWwindow* window, int button, int action, int mods){
        // Forward the event to ImGui
        ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
        
        //if ImGui doesn't want the event, process it
        ImGuiIO& io = ImGui::GetIO();
        if(!io.WantCaptureMouse){
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            if(button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS){
                updt_mpos(xpos, ypos);
            }
            if(button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS){
                //...
            }
        }
    }

    void cursor_position(GLFWwindow* window, double xpos, double ypos){
        //forward the event to ImGui
        ImGui_ImplGlfw_CursorPosCallback(window, xpos, ypos);

        //if ImGui doesn't want the event, process i
        ImGuiIO& io = ImGui::GetIO();
        if(!io.WantCaptureMouse){
            int leftState = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
            if(leftState == GLFW_PRESS){
                h_params.mx = (float)(h_params.scale*(xpos - gbl::SCREEN_X / 2));
                h_params.my = -(float)(h_params.scale*(ypos - gbl::SCREEN_Y / 2));
            }
        }
    }

    void scroll(GLFWwindow* window, double xoffset, double yoffset){
        // Forward the event to ImGui
        ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
        
        //if ImGui doesn't want the event, process it
        ImGuiIO& io = ImGui::GetIO();
        if(!io.WantCaptureMouse){
            if (yoffset >0) h_params.scale /= 1.05f;
	        else h_params.scale *= 1.05f;
        }
    }

    void window_size(GLFWwindow* window, int width, int height){
        //reszing logic handled in gbl::resizePixelsBuffer() called from gbl::calculate
        gbl::needResize = true;
        gbl::paused = true;
    }

}//end namespace cbk

int main(void){
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(gbl::SCREEN_X, gbl::SCREEN_Y, TITLE, NULL, NULL);
    if (!window){
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);
    utl::initImGui(window);

    /* malloc render specific arrays... */
    init();
    {/* set up parameter*/
        h_params.scale = 0.003f;
        h_params.mx = 0.0f;
        h_params.my = 0.0f;

        //add your own parameters ...
    }

    /* Initialize callback*/
    glfwSetKeyCallback(window, cbk::key);
    glfwSetMouseButtonCallback(window, cbk::mouse_button);
    glfwSetCursorPosCallback(window, cbk::cursor_position);
    glfwSetScrollCallback(window, cbk::scroll);
    glfwSetWindowSizeCallback(window, cbk::window_size);


    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Poll for and process events */
        glfwPollEvents();

        /* Interface*/
        utl::newframeImGui();
        if(gbl::otherWindow) utl::wdw_info(gbl::mode, gbl::SCREEN_X,gbl::SCREEN_Y,gbl::currentFPS);
        
        /* Render here */
        gbl::calculate(window);
        gbl::display();
        if(!gbl::paused) glDrawPixels(gbl::SCREEN_X, gbl::SCREEN_Y, GL_RGBA, GL_FLOAT, gbl::pixels);
        
        /* Interface */
        utl::endframeImGui();
        utl::multiViewportImGui(window);
        
        /* Swap front and back buffers */
        glfwSwapBuffers(window);
    }

    utl::shutdownImGui();
    glfwTerminate();
    return 0;
}