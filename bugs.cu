#include "util.hpp"
#include <random>

#define TITLE "Bugs"


//mandatory forward declaration
namespace wdw{
    void automataParam();
}
namespace gbl{
    int max_fps;
}
namespace gpu{
    bool gridSwap;
}

namespace bugs{
    float4* h_grid;
    float4* d_grid1;
    float4* d_grid2;

    __device__ __host__ inline void aliveCell(float4* cell){
        cell->x = 1.0f;
        cell->y = 1.0f;
        cell->z = 0.0f;
    }
    __device__ __host__ inline void killCell(float4* cell){
        cell->x = 0.0f;
        cell->y = 0.0f;
        cell->z = 1.0f;
    }
    __device__ __host__ inline bool isAlive(float4* cell){
        return (cell->x > 0.9);
        
    }

    __device__ __host__ int count_neighbors(int i, int j, float4* oldgrid, int width, int height, int range){
        int nb_neighbors = 0;
        for(int offset_i = - range; offset_i <= range; offset_i++){
            for(int offset_j = - range; offset_j <= range; offset_j++){
            
                if(offset_i == 0 && offset_j == 0) continue; //ignore self
                
                //warp in a donut shape
                int coor_i = (i + offset_i + height) % height;
                int coor_j = (j + offset_j + width) % width;
                float4* cell = oldgrid + (coor_i * width + coor_j);

                if(bugs::isAlive(cell)) nb_neighbors++;
            }
        }
        return nb_neighbors;
    }
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

    //collor settings
    float live_decay_fac;
    float dead_decay_fac;


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
            if(0 ==rand()%2){
                bugs::h_grid[i].x = 1.0;
                bugs::h_grid[i].y = 1.0;
                bugs::h_grid[i].z = 1.0;
                bugs::h_grid[i].w = 1.0;
            } else{
                bugs::h_grid[i].x = 0.0;
                bugs::h_grid[i].y = 0.0;
                bugs::h_grid[i].z = 0.0;
                bugs::h_grid[i].w = 1.0;
            }
        }
        if(gbl::mode == GPU_MODE){
            //send grid to gpu
            checkCudaErrors( cudaMemcpy(bugs::d_grid1, bugs::h_grid, gbl::SCREEN_X*gbl::SCREEN_Y*sizeof(float4), cudaMemcpyHostToDevice) );
            gpu::gridSwap = false;
        }
    }
    void clear_all(){
        for(int i=0; i < gbl::SCREEN_X * gbl::SCREEN_Y; i++){
            bugs::h_grid[i].x = 0.0f;
            bugs::h_grid[i].x = 0.0f;
            bugs::h_grid[i].x = 0.0f;
            bugs::h_grid[i].x = 1.0f;
        }
        if(gbl::mode == GPU_MODE){
            checkCudaErrors( cudaMemcpy(bugs::d_grid1, bugs::h_grid, gbl::SCREEN_X*gbl::SCREEN_Y*sizeof(float4), cudaMemcpyHostToDevice) ); //get pixels values from gpu
            gpu::gridSwap = false;
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
        bugs::h_grid = (float4*)malloc(gbl::SCREEN_X*gbl::SCREEN_Y*sizeof(float4));
        gbl::display = imp_Bugs;
    }
    void clean(){
        free(bugs::h_grid);
    }
    void reinit(){
        bugs::h_grid = (float4*)realloc(bugs::h_grid, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(float4));
    }

//torm todo
    // int count_neighbors(int i, int j, int width, int height){
    //     int nb_neighbors = 0;
    //     for(int offset_i = - h_params.RANGE; offset_i <= h_params.RANGE; offset_i++){
    //         for(int offset_j = - h_params.RANGE; offset_j <= h_params.RANGE; offset_j++){
            
    //             if(offset_i == 0 && offset_j == 0) continue; //ignore self
                
    //             //warp in a donut shape
    //             int coor_i = (i + offset_i + height) % height;
    //             int coor_j = (j + offset_j + width) % width;

    //             if((bugs::h_grid + (coor_i * width + coor_j))->x > .9) nb_neighbors++;
    //         }
    //     }
    //     return nb_neighbors;
    // }


    void imp_Bugs() {
		int i, j;
		for (i = 0; i < gbl::SCREEN_Y; i++){
			for (j = 0; j < gbl::SCREEN_X; j++)
			{
                float4* cell = bugs::h_grid + (i * gbl::SCREEN_X + j);

                //update values
				int nb_neighbors = bugs::count_neighbors(i, j, bugs::h_grid, gbl::SCREEN_X, gbl::SCREEN_Y, h_params.RANGE);
                if(bugs::isAlive(cell)){
                    if(h_params.SURVIVE_LOW <= nb_neighbors && nb_neighbors <= h_params.SURVIVE_HIGH){}
                    else{bugs::killCell(cell);}
                }
                else{ //if no cel, does it birth ?
                    if(h_params.BIRTH_LOW <= nb_neighbors && nb_neighbors <= h_params.BIRTH_HIGH){bugs::aliveCell(cell);}
                    else{}
                }
			}
        }
	}

}//end namespace cpu

namespace gpu{
    void imp_Bugs();

    void init(){
        checkCudaErrors( cudaMallocHost((void**) &bugs::h_grid, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(float4)));
	    checkCudaErrors( cudaMalloc((void**)&bugs::d_grid1, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(float4)) );
	    checkCudaErrors( cudaMalloc((void**)&bugs::d_grid2, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(float4)) );

        //not possible to fetch previous grid because it was cleaned
        //checkCudaErrors( cudaMemcpy(bugs::d_grid1, bugs::h_grid, gbl::SCREEN_X*gbl::SCREEN_Y*sizeof(float4), cudaMemcpyHostToDevice) );
        gridSwap = false;
        gbl::display = imp_Bugs;
    }
    void clean(){
        checkCudaErrors( cudaFreeHost(bugs::h_grid));
	    checkCudaErrors( cudaFree(bugs::d_grid1) );
	    checkCudaErrors( cudaFree(bugs::d_grid2) );

    }
    void reinit(){
        clean();
        init();
    }

    void setDeviceParameters(const Param& params) {
        checkCudaErrors( cudaMemcpyToSymbol(d_params, &params, sizeof(Param)) );
    }



    __global__ void kernelBugs(float4* gridOld, float4* gridNew, int SCREENX, int SCREENY) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index < SCREENX * SCREENY) {
            //access constant memory once per thread !
            Param t_params = d_params;
            float scale = t_params.scale;
            Complex offset = t_params.offset;
            float sx = t_params.mx;
            float sy = t_params.my;


            //deduce i, j (pixel coordinate) from threadIdx, blockIdx ...
            int i = index / SCREENX;
		    int j = index - i * SCREENX;

			//new cell
            float4* cellOld = gridOld + (i * SCREENX + j);
            float4* cellNew = gridNew + (i * SCREENX + j);
            
            //compute alive in neightborhood
            int nb_neighbors = bugs::count_neighbors(i, j, gridOld, SCREENX, SCREENY, d_params.RANGE);
            
            if(bugs::isAlive(cellOld)){
                if(d_params.SURVIVE_LOW <= nb_neighbors && nb_neighbors <= d_params.SURVIVE_HIGH)
                    //{bugs::aliveCell(cellNew);}
                    {*cellNew=*cellOld;
                    cellNew->y*=d_params.live_decay_fac;}
                else{bugs::killCell(cellNew);}
            }
            else{ //if no cel, does it birth ?
                if(d_params.BIRTH_LOW <= nb_neighbors && nb_neighbors <= d_params.BIRTH_HIGH)
                    {bugs::aliveCell(cellNew);}
                else
                    //{bugs::killCell(cellNew);}
                    {*cellNew = *cellOld;
                    cellNew->z*=d_params.live_decay_fac;}
            }
		}
	}

    void imp_Bugs(){
        //initialisation
        int N = gbl::SCREEN_X * gbl::SCREEN_Y;
		int M = 256;

        //grid swapping 1=>2 / 2=>1
        float4* oldgrid, *newgrid;
        if(!gridSwap){
            oldgrid = bugs::d_grid1;
            newgrid = bugs::d_grid2;
        } else{
            oldgrid = bugs::d_grid2;
            newgrid = bugs::d_grid1;
        }
        gridSwap = !gridSwap;

        //computation
        kernelBugs << <(N + M - 1) / M, M >> > (oldgrid, newgrid, gbl::SCREEN_X, gbl::SCREEN_Y);

        //fecth grid from GPU to CPU
        checkCudaErrors( cudaMemcpy(bugs::h_grid, newgrid, N * sizeof(float4), cudaMemcpyDeviceToHost));
    }

}//end namespace gpu

namespace wdw{
    void automataParam(){
        ImGui::Begin("Celular automata");

        ImGui::InputInt("max iter/frame", &gbl::max_fps);
        ImGui::NewLine();

        ImGui::InputInt("RANGE", &h_params.RANGE);
        ImGui::InputInt("SURVIVE_LOW", &h_params.SURVIVE_LOW);
        ImGui::InputInt("SURVIVE_HIGH", &h_params.SURVIVE_HIGH);
        ImGui::InputInt("BIRTH_LOW", &h_params.BIRTH_LOW);
        ImGui::InputInt("BIRTH_HIGH", &h_params.BIRTH_HIGH);

        if (ImGui::TreeNode("Colors management"))
        {
            ImGui::SliderFloat("live decay fac", &h_params.live_decay_fac, 0.0f, 1.0f);
            ImGui::SliderFloat("dead decay fac", &h_params.dead_decay_fac, 0.0f, 1.0f);

            ImGui::TreePop();
        }
        

        if (ImGui::TreeNode("Presets"))
        {
            if(ImGui::Button("Conway's Game of life")) preset::game_of_life();
            if(ImGui::Button("bugs")) preset::bugs();

            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Options"))
        {
            if(ImGui::Button("clear all")) preset::clear_all();
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
        h_params.live_decay_fac = 0.9;
        h_params.dead_decay_fac = 0.9;

        //framerate
        gbl::max_fps = 20;
        preset::game_of_life();
    }

    /* Initialize callback*/
    glfwSetKeyCallback(window, cbk::key);
    glfwSetMouseButtonCallback(window, cbk::mouse_button);
    glfwSetCursorPosCallback(window, cbk::cursor_position);
    glfwSetScrollCallback(window, cbk::scroll);
    glfwSetWindowSizeCallback(window, cbk::window_size);

    /*start timer*/
    double last_frame_time = glfwGetTime();

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Poll for and process events */
        glfwPollEvents();

        /* Interface*/
        utl::newframeImGui();
        if(gbl::otherWindow) {
            utl::wdw_info(gbl::mode, gbl::SCREEN_X,gbl::SCREEN_Y,gbl::currentFPS);
            wdw::automataParam();
        }
        
        //timer management
        double curr_time = glfwGetTime();
        double framerate = (double)1 / (double)gbl::max_fps;

        /* Render */
        gbl::calculate(window);
        gpu::setDeviceParameters(h_params);
        if(curr_time -  last_frame_time > framerate){
            gbl::display();
            last_frame_time = curr_time;
        }            
        if(!gbl::paused) glDrawPixels(gbl::SCREEN_X, gbl::SCREEN_Y, GL_RGBA, GL_FLOAT, bugs::h_grid);
        
  

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