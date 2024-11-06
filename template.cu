#include "util.hpp"
#define TITLE "template"

/* use task to compile or run command*/


namespace gbl{
    int SCREEN_X=640;
    int SCREEN_Y=480;
    float4* pixels;

    int mode = CPU_MODE;

    float scale = 0.003f;
    //mouse coordinate
    float mx, my;

}//end namespace gbl


namespace cpu{
    void init(){
        gbl::pixels = (float4*)malloc(gbl::SCREEN_X*gbl::SCREEN_Y*sizeof(float4));
    }
    void clean(){
        free(gbl::pixels);
    }

    void example(){
        int i, j;
        for (i = 0; i<gbl::SCREEN_Y; i++){
            for (j = 0; j<gbl::SCREEN_X; j++){
                float x = (float)(gbl::scale*(j - gbl::SCREEN_X / 2));
                float y = (float)(gbl::scale*(i - gbl::SCREEN_Y / 2));
                float4* p = gbl::pixels + (i*gbl::SCREEN_X + j);
                // default: black
                p->x = 0.0f;
                p->y = 0.0f;
                p->z = 0.0f;
                p->w = 1.0f;
                if (sqrt((x - gbl::mx)*(x - gbl::mx) + (y - gbl::my)*(y - gbl::my))<0.01)
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

namespace gpu{
    void init(){
        gbl::pixels = (float4*)malloc(gbl::SCREEN_X*gbl::SCREEN_Y*sizeof(float4));
    }
    void clean(){
        free(gbl::pixels);
    }
    
}//end namespace gpu

void clean(){
	switch (gbl::mode){
        case CPU_MODE: cpu::clean(); break;
        case GPU_MODE: gpu::clean(); break;
	}
}
void init(){
	switch (gbl::mode){
        case CPU_MODE: cpu::init(); break;
        case GPU_MODE: gpu::init(); break;
	}
}
void toggleMode(int m){
	clean();
	gbl::mode = (gbl::mode+1)%NB_MODE;
	init();
}

namespace cbk{ 
    /*various callback
    You must ALWAYS forward the event to ImGui before processing it
    You can find relevant ImGui callback in ./imgui/imgui_impl_glfw.cpp line 536 in function ImGui_ImplGlfw_InstallCallbacks
    */

    void mouse_button(GLFWwindow* window, int button, int action, int mods){
        // Forward the event to ImGui
        ImGuiIO& io = ImGui::GetIO();
        ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
        
        //if ImGui doesn't want the event, process it
        if(!io.WantCaptureMouse){
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS){
                gbl::mx = (float)(gbl::scale*(xpos - gbl::SCREEN_X / 2));
                gbl::my = -(float)(gbl::scale*(ypos - gbl::SCREEN_Y / 2));
            }
        }
    }

    void scroll(GLFWwindow* window, double xoffset, double yoffset){
        // Forward the event to ImGui
        ImGuiIO& io = ImGui::GetIO();
        ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
        
        //if ImGui doesn't want the event, process it
        if(!io.WantCaptureMouse){
            if (yoffset >0) gbl::scale /= 1.05f;
	        else gbl::scale *= 1.05f;
        }
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

    /* init render specific values*/
    init();

    /* Initialize callback*/
    glfwSetMouseButtonCallback(window, cbk::mouse_button);
    glfwSetScrollCallback(window, cbk::scroll);

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Poll for and process events */
        glfwPollEvents();

        utl::newframeImGui();
        utl::wdw_info();
        
        /* Render here */
        //glClear(GL_COLOR_BUFFER_BIT);
        cpu::example(); //calcule colors
        glDrawPixels(gbl::SCREEN_X, gbl::SCREEN_Y, GL_RGBA, GL_FLOAT, gbl::pixels);
        
        /* end frame for imgui*/
        utl::endframeImGui();
        utl::multiViewportImGui(window);
        

        /* Swap front and back buffers */
        glfwSwapBuffers(window);
    }

    utl::shutdownImGui();
    glfwTerminate();
    return 0;
}