//glfw includes
#include <GLFW/glfw3.h>
#include <iostream>

//imgui includes
#include "./imgui/imgui.h"
#include "./imgui/imgui_impl_glfw.h"
#include "./imgui/imgui_impl_opengl3.h"


#ifndef NDEBUG
    #define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
    inline void __checkCudaErrors(cudaError err, const char* file, const int line) {
        if (err != cudaSuccess)
        {
            fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
            system("pause");
            exit(1);
        }
    }
    inline void checkKernelErrors() {
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
    }
#else //disable GPU debuging
    #define checkCudaErrors(err) err
    inline void checkKernelErrors() {}
#endif

#define CPU_MODE 1
#define GPU_MODE 2

#define FPS_UPDATE_DELAY 0.5


//mandatory forward declaration
void reinit();

//global variables
namespace gbl{
    int SCREEN_X=640;
    int SCREEN_Y=480;
    float4* pixels;
    float4* d_pixels;

    //display callback for drawing pixels
    void (*display)();

    int mode = CPU_MODE;
    bool paused = false;        //stop displaying pixel (prevent unallowed memory access when processing callback)
    bool needResize = false;    //if pixel buffer needs to be reallocated (prevent too many calls to  free/malloc)
    bool otherWindow = true;    //display or not other window

    int frameAcc = 0; //number of frame since last FPS calculation
    double prevUpdt = 0.0; //time at previous FPS evaluation
    int currentFPS = 0.0f;


    void resizePixelsBuffer(){
        reinit();
    }

    //handle fps computation, and reallocating buffer if needed(to avoid too many call to mallloc/free)
    void calculate(GLFWwindow* window){
        frameAcc++;
        double timeCurr  = glfwGetTime();
        float elapsedTime = timeCurr-prevUpdt;
        if(elapsedTime>FPS_UPDATE_DELAY){
            currentFPS = frameAcc / elapsedTime ;
            frameAcc = 0;
            prevUpdt = timeCurr;
            if(needResize){
                glfwGetWindowSize(window, &SCREEN_X, &SCREEN_Y);
                resizePixelsBuffer();
                needResize = false;
                paused = false;
            }
        }

    }
}//end namespace gbl


void clean(); void init();
namespace wdw{ void wdw_additional();}
namespace utl{
    
void initImGui(GLFWwindow* window){
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // IF using Docking Branch 
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;       //enables multi window


    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);          // Second param install_callback=true will install GLFW callbacks and chain to existing ones.
    ImGui_ImplOpenGL3_Init();
}

void newframeImGui(){
    // (Your code calls glfwPollEvents())
    // ...
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}


void toggleMode(int m){
    gbl::paused = true;
	clean();
	init();
    gbl::paused = false;
}

void wdw_info(int mode, int sx, int sy, int fps){
    ImGui::Begin("Base info");

    ImGui::Text("Mode : "); 
    ImGui::SameLine(); if(ImGui::RadioButton("CPU", &gbl::mode, CPU_MODE)){toggleMode(CPU_MODE);}
    ImGui::SameLine(); if(ImGui::RadioButton("GPU", &gbl::mode, GPU_MODE)){toggleMode(GPU_MODE);}

    ImGui::Text("Current Window size : %d x %d", sx, sy);
    ImGui::Text("FPS : %d", fps);

    if(mode == GPU_MODE){
        wdw::wdw_additional();
    }
    ImGui::End();
}

void multiViewportImGui(GLFWwindow* window){
    ImGuiIO& io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable){
        // Update and Render additional Platform Windows
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();

        // TODO for OpenGL: restore current GL context.
        glfwMakeContextCurrent(window);
    }
}

void endframeImGui(){
    // Rendering
    // (Your code clears your framebuffer, renders your other stuff etc.)
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    // (Your code calls glfwSwapBuffers() etc.)
}

void shutdownImGui(){
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}


} //end namespace utl

