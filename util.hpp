//glfw includes
#include <GLFW/glfw3.h>
#include <iostream>

//imgui includes
#include "./imgui/imgui.h"
#include "./imgui/imgui_impl_glfw.h"
#include "./imgui/imgui_impl_opengl3.h"

#define CPU_MODE 0
#define GPU_MODE 1
#define NB_MODE 2 //assuming you want to add more mode. mode is update as such : mode = (mode+1)%NB_MODE

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

void wdw_info(){
    ImGui::Begin("Base info");
	if(ImGui::Button("update")){
		ImGui::Text("ahah");
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

