#include "util.hpp"
#define TITLE "template"

/* use task to compile or run command*/


namespace gbl{
    int SCREEN_X=640;
    int SCREEN_Y=480;
    float4* pixels;

}//end namespace gbl


namespace cpu{
    void init(){
        gbl::pixels = (float4*)malloc(gbl::SCREEN_X*gbl::SCREEN_Y*sizeof(float4));
    }
    void clean(){
        free(gbl::pixels);
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

int main(void){
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(gbl::SCREEN_X, gbl::SCREEN_Y, "Hello World", NULL, NULL);
    if (!window){
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);
    utl::initImGui(window);

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Poll for and process events */
        glfwPollEvents();

        utl::newframeImGui();
        
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);
        
        utl::endframeImGui();
        utl::multiViewportImGui(window);

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

    }

    utl::shutdownImGui();
    glfwTerminate();
    return 0;
}