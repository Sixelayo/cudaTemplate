#include "util.hpp"
#include <random>

#define TITLE "Julia"


//mandatory forward declaration
namespace wdw{
    void automataParam();
}

namespace bugs{
    bool* h_env;
    bool* d_env;
}


//complexe numbe stored as a+ib
class Complex {
public:
    float a;
    float b;
    __device__ __host__ Complex(){}
    __device__ __host__ Complex(float a, float b) : a(a), b(b){}

    __device__ __host__ Complex operator+(const Complex& other) const {
        return Complex(a + other.a, b + other.b);
    }
    __device__ __host__ Complex operator*(const Complex& other) const {
        return Complex(a * other.a - b * other.b, a * other.b + b * other.a);
    }

};
struct MyCol{ //used for passing color to gpu
    float x, y, z, w;
    __device__ __host__ MyCol(){}
    __device__ __host__ MyCol(float x, float y, float z, float w) : x(x), y(y), z(z), w(w){}
    __device__ __host__ MyCol(float* c) : x(c[0]), y(c[1]), z(c[2]), w(c[3]){}
    __device__ __host__ MyCol(const MyCol& c) : x(c.x), y(c.y), z(c.z), w(c.w){}
};


__device__ __host__ float minkowski(Complex c, float order) {
    return pow((pow(c.a, order) + pow(c.b, order)), 1.0f / order);
}


struct Param {
    float scale;
    float mx, my; //mousepose
    Complex offset;

    MyCol col_alive;
    MyCol col_dead;

    int RANGE;
    int SURVIVE_LOW;
    int SURVIVE_HIGH;
    int BIRTH_LOW;
    int BIRTH_HIGH;


};
Param h_params;
__constant__ Param d_params;

namespace preset{
    void set_preset(int range, int surv_low, int surv_high, int birth_low, int birth_high){
        h_params.RANGE = range;
        h_params.SURVIVE_LOW = surv_low;
        h_params.SURVIVE_HIGH = surv_high;
        h_params.BIRTH_LOW = birth_low;
        h_params.BIRTH_HIGH = birth_high;
    }

    void random_config(){
        for(int i=0; i < gbl::SCREEN_X * gbl::SCREEN_Y; i++){
            bugs::h_env[i] = (0 ==rand()%2);
        }
        if(gbl::mode == GPU_MODE){
            checkCudaErrors( cudaMemcpy(bugs::d_env, bugs::h_env, gbl::SCREEN_X*gbl::SCREEN_Y, cudaMemcpyHostToDevice) ); //get pixels values from gpu
        }
    }

    void game_of_life(){
        set_preset(1,3,4,3,3);
    }
    void bugs(){
        set_preset(5,34,58,34,45);
    }
}//end namespace prs


namespace cpu{
    void imp_Bugs();

    void init(){
        gbl::pixels = (float4*)malloc(gbl::SCREEN_X*gbl::SCREEN_Y*sizeof(float4));
        bugs::h_env = (bool*)malloc(gbl::SCREEN_X*gbl::SCREEN_Y*sizeof(bool));
        gbl::display = imp_Bugs;
    }
    void clean(){
        free(gbl::pixels);
        free(bugs::h_env);
    }
    void reinit(){
        gbl::pixels = (float4*)realloc(gbl::pixels, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(float4));
        bugs::h_env = (bool*)realloc(gbl::pixels, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(bool));
    }

    int count_neighbors(int i, int j, int width, int height){
        int nb_neighbors = 0;
        for(int offset_i = - h_params.RANGE; offset_i <= h_params.RANGE; offset_i++){
            for(int offset_j = - h_params.RANGE; offset_j <= h_params.RANGE; offset_j++){
            
                if(offset_i == 0 && offset_j == 0) continue; //ignore self
                
                //warp in a donut shape
                int coor_i = (i + offset_i + height) % height;
                int coor_j = (j + offset_j + width) % width;

                //if(coor_i%50==0 && coor_j%50==0) std::cout <<"/i:"<<coor_i<<"/j:"<<coor_j; //torm

                if(*(bugs::h_env + (coor_i * width + coor_j))) nb_neighbors++;
            }
        }
        return nb_neighbors;
    }

    void imp_Bugs() {

        if(gbl::otherWindow) {
            wdw::automataParam();
        }

		int i, j;
		for (i = 0; i < gbl::SCREEN_Y; i++)
			for (j = 0; j < gbl::SCREEN_X; j++)
			{
				float4* p = gbl::pixels + (i * gbl::SCREEN_X + j);
                bool* alive = bugs::h_env + (i * gbl::SCREEN_X + j);

                //update environnement
				int nb_neighbors = count_neighbors(i, j, gbl::SCREEN_X, gbl::SCREEN_Y);
                if(*alive){
                    if(h_params.SURVIVE_LOW <= nb_neighbors && nb_neighbors <= h_params.SURVIVE_HIGH){}
                    else{*alive = false;}
                }
                else{ //if no cel, does it birth ?
                    if(h_params.BIRTH_LOW <= nb_neighbors && nb_neighbors <= h_params.BIRTH_HIGH){}
                    else{*alive = true;}
                }

                

                //update color
                if(*alive){
                    p->x = h_params.col_alive.x;p->y = h_params.col_alive.y;p->z = h_params.col_alive.z;
                }
                else{
                    p->x = h_params.col_dead.x;p->y = h_params.col_dead.y;p->z = h_params.col_dead.z;          
                }
				p->w = 1.0f;


			}
	}

}//end namespace cpu

namespace gpu{
    void imp_Bugs();

    void init(){
        checkCudaErrors( cudaMallocHost((void**) &gbl::pixels, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(float4)) );
	    checkCudaErrors( cudaMalloc((void**)&gbl::d_pixels, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(float4)) );
        checkCudaErrors( cudaMallocHost((void**) &bugs::h_env, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(bool)) );
	    checkCudaErrors( cudaMalloc((void**)&bugs::d_env, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(bool)));
        gbl::display = imp_Bugs;
    }
    void clean(){
        checkCudaErrors( cudaFreeHost(gbl::pixels));
	    checkCudaErrors( cudaFree(gbl::d_pixels) );
        checkCudaErrors( cudaFreeHost(bugs::h_env));
	    checkCudaErrors( cudaFree(bugs::d_env) );
    }
    void reinit(){
        clean();
        init();
    }

    void setDeviceParameters(const Param& params) {
        checkCudaErrors( cudaMemcpyToSymbol(d_params, &params, sizeof(Param)) );
    }

    void imp_Bugs(){

    }

}//end namespace gpu

namespace wdw{
    void automataParam(){
        ImGui::Begin("Celular automata");

        ImGui::InputInt("RANGE", &h_params.RANGE);
        ImGui::InputInt("SURVIVE_LOW", &h_params.SURVIVE_LOW);
        ImGui::InputInt("SURVIVE_HIGH", &h_params.SURVIVE_HIGH);
        ImGui::InputInt("BIRTH_LOW", &h_params.BIRTH_LOW);
        ImGui::InputInt("BIRTH_HIGH", &h_params.BIRTH_HIGH);

        if (ImGui::TreeNode("Presets"))
        {
            if(ImGui::Button("Conway's Game of life")) preset::game_of_life();
            if(ImGui::Button("bugs")) preset::bugs();

            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Options"))
        {
            if(ImGui::Button("random config")) preset::random_config();

            ImGui::TreePop();
        }



        /*
        if (ImGui::TreeNode("Color management"))
        {
            static float c_in[4] = { 0.8f, 0.3f, 0.4f, 1.0f };
            static float c_step[4] = { 1.0f, 0.5f, 0.2f, 1.0f };
            static float c_out[4] = { 0.4f, 0.7f, 0.9f, 1.0f };
            ImGui::ColorEdit4("ease in color", (float*)&c_in, ImGuiColorEditFlags_DisplayRGB | ImGuiColorEditFlags_InputRGB | ImGuiColorEditFlags_Float);
            ImGui::ColorEdit4("Step color", (float*)&c_step, ImGuiColorEditFlags_DisplayRGB | ImGuiColorEditFlags_InputRGB | ImGuiColorEditFlags_Float);
            ImGui::ColorEdit4("ease out color", (float*)&c_out, ImGuiColorEditFlags_DisplayRGB | ImGuiColorEditFlags_InputRGB | ImGuiColorEditFlags_Float);
            ImGui::DragFloat("in easing facor", &h_params.easing_fac_in, 0.005f,0.01f,10.0f);
            ImGui::DragFloat("out easing factor", &h_params.easing_fac_out, 0.005f,0.001f,2.0f);
            
            h_params.color_easeIn = MyCol{c_in};
            h_params.color_step = MyCol{c_step};
            h_params.color_easeOut = MyCol{c_out};

            if(ImGui::Button("foo")) { //torm
                std::cout << c_step[0] << " / " << c_step[1] << " / " << c_step[2] << " /" <<c_step[3] <<"\n";
                std::cout << c_out[0] << " / " << c_out[1] << " / " << c_out[2] << " /" << c_out[3] <<"\n";
                std::cout << c_in[0] << " / " << c_in[1] << " / " << c_in[2]<< " /"  << c_in[3] <<"\n";
            }

            ImGui::TreePop();
        }*/

        ImGui::End();
    }
}//end namespace wdw



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

    void updt_mpos(double xpos, double ypos){
        h_params.mx = h_params.offset.a + (float)(h_params.scale*(xpos - gbl::SCREEN_X / 2));
        h_params.my = h_params.offset.b - (float)(h_params.scale*(ypos - gbl::SCREEN_Y / 2));
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
                h_params.offset.a += (float)(h_params.scale * (xpos - gbl::SCREEN_X / 2));
		        h_params.offset.b += -(float)(h_params.scale * (ypos - gbl::SCREEN_Y / 2));
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
                updt_mpos(xpos, ypos);
            }
        }
    }

    void scroll(GLFWwindow* window, double xoffset, double yoffset){
        // Forward the event to ImGui
        ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
        
        //if ImGui doesn't want the event, process it
        ImGuiIO& io = ImGui::GetIO();
        if(!io.WantCaptureMouse){
            float fac = (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_LEFT_CONTROL)) ?  1.16f : 1.05f;
            if (yoffset >0) h_params.scale /= fac;
	        else h_params.scale *= fac;
        }
    }

    void window_size(GLFWwindow* window, int width, int height){
        //reszing logic handled in gbl::resizePixelsBuffer() called from gbl::calculate
        gbl::paused = true;
        gbl::needResize = true;
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

    /* malloc values ...*/
    init();
    {/* set up generic parameters*/
        h_params.scale = 0.003f;
        h_params.mx = 0.0f;
        h_params.my = 0.0f;
        h_params.offset = Complex(0.0f, 0.0f);
        
        //color control
        h_params.col_dead = MyCol(0.1f, 0.1f, 0.1f, 1.0f);
        h_params.col_alive = MyCol(0.3f, 0.4f, 0.3f, 1.0f);

        preset::bugs();
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
        gpu::setDeviceParameters(h_params);
        gbl::display();
        if(!gbl::paused) glDrawPixels(gbl::SCREEN_X, gbl::SCREEN_Y, GL_RGBA, GL_FLOAT, gbl::pixels);
        
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