#include "util.hpp"
#include <random>
#include <math.h>

#include "camera.cpp"


#define DEBUGA(x) std::cout << __FILE__ << ":" << __LINE__ << " - " << #x << " = " << (x) << std::endl
#define DEBUGV(x) std::cout << (x)


#define TITLE "NBODY"
#define MAXBODYCOUNT 1024

//mandatory forward declaration
namespace wdw{
    void nbodyParam();
}
namespace gbl{
    int max_fps;
}
namespace gpu{
}

struct Body{
    float4 pos;
    float4 vel;
    float mass;
};

namespace nbd{
    Body* h_bodies1;
    Body* h_bodies2;
    Body* d_bodies1;
    Body* d_bodies2;

    float dpos = 1.0f;
    float dvel = 0.0001f;
    static float minmass = 1.0f;
    static float maxmass = 5.0f;

    void randomBodies(Body* bodies, int bodycount){
        float x, y, z, r;
        for(int i =0; i<bodycount; i++){
            Body& body = bodies[i];
            //pos
            x = (2*((rand()%1000)/1000.0f)-1);
            y = (2*((rand()%1000)/1000.0f)-1);
            z = (2*((rand()%1000)/1000.0f)-1);
            r = (rand()%1000)/1000.0f/sqrt(x*x+y*y+z*z);
            body.pos.x = r*dpos*x;
            body.pos.y = r*dpos*y;
            body.pos.z = r*dpos*z;
            body.pos.w = 1.0f;

            //vel
            x = (2*((rand()%1000)/1000.0f)-1);
            y = (2*((rand()%1000)/1000.0f)-1);
            z = (2*((rand()%1000)/1000.0f)-1);
            r = (rand()%1000)/1000.0f/sqrt(x*x+y*y+z*z);
            body.vel.x = r*dvel*x;
            body.vel.y = r*dvel*y;
            body.vel.z = r*dvel*z;
            body.vel.w = 1.0f;

            //mass
            body.mass = minmass+(maxmass-minmass)*((rand()%1000)/1000.0f);
        }
    }

    void updtBody(Body* body){

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

    //nbd
    int nbBodies;
};
Param h_params;
__constant__ Param d_params;



namespace cpu{
    void imp_NBody();

    void init(){
        //bugs::h_grid1 = (float4*)malloc(gbl::SCREEN_X*gbl::SCREEN_Y*sizeof(float4));
        //bugs::h_grid2 = (float4*)malloc(gbl::SCREEN_X*gbl::SCREEN_Y*sizeof(float4));
        nbd::h_bodies1 = (Body*) malloc(MAXBODYCOUNT * sizeof(Body));
        nbd::h_bodies2 = (Body*) malloc(MAXBODYCOUNT * sizeof(Body));
        gbl::display = imp_NBody;
    }
    void clean(){
        free(nbd::h_bodies1); nbd::h_bodies1 = nullptr;
        free(nbd::h_bodies2); nbd::h_bodies2 = nullptr;
    }
    void reinit(){
        //bugs::h_grid1 = (float4*)realloc(bugs::h_grid1, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(float4));
        //bugs::h_grid2 = (float4*)realloc(bugs::h_grid2, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(float4));
    }


    void imp_NBody() {
		// your N-body algorithm here!
        for (int i=0;i<h_params.nbBodies;i++)
        {
            nbd::updtBody(nbd::h_bodies1+i);
        }
	}

}//end namespace cpu

namespace gpu{
    void imp_Bugs();

    void init(){
        // checkCudaErrors( cudaMallocHost((void**) &bugs::h_grid1, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(float4)));
	    // checkCudaErrors( cudaMalloc((void**)&bugs::d_grid1, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(float4)) );
	    // checkCudaErrors( cudaMalloc((void**)&bugs::d_grid2, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(float4)) );

        //ideally transfer active grid in cpu to gpu but due to architecture not possible to fetch previous grid because it was cleaned
        //checkCudaErrors( cudaMemcpy(bugs::d_grid1, bugs::h_grid, gbl::SCREEN_X*gbl::SCREEN_Y*sizeof(float4), cudaMemcpyHostToDevice) );
        gbl::display = imp_Bugs;
    }
    void clean(){
        // checkCudaErrors( cudaFreeHost(bugs::h_grid1));
	    // checkCudaErrors( cudaFree(bugs::d_grid1) );
	    // checkCudaErrors( cudaFree(bugs::d_grid2) );

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
            //float scale = t_params.scale;
            //float sx = t_params.mx;
            //float sy = t_params.my;


            //deduce i, j (pixel coordinate) from threadIdx, blockIdx 
            int i = index / SCREENX;
		    int j = index - i * SCREENX;

			//new cell
            float4* cellOld = gridOld + (i * SCREENX + j);
            float4* cellNew = gridNew + (i * SCREENX + j);
            
		}
	}

    void imp_Bugs(){
        //initialisation
        int N = gbl::SCREEN_X * gbl::SCREEN_Y;
		int M = 256;

        //always swap from grid 1 to grid 2 and sawp pointers after

        //computation
        //kernelBugs << <(N + M - 1) / M, M >> > (bugs::d_grid1, bugs::d_grid2, gbl::SCREEN_X, gbl::SCREEN_Y);

        //fecth grid from GPU to CPU and swap grid
        //checkCudaErrors( cudaMemcpy(bugs::h_grid1, bugs::d_grid2, N * sizeof(float4), cudaMemcpyDeviceToHost));
        //std::swap(bugs::d_grid1, bugs::d_grid2);
    }

}//end namespace gpu




namespace wdw{
    static void HelpMarker(const char* desc){
        ImGui::TextDisabled("(?)");
        if (ImGui::BeginItemTooltip())
        {
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextUnformatted(desc);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }

    void automataParam(){
        ImGui::Begin("Nbodies");



        ImGui::NewLine();

        ImGui::SeparatorText("Advanced parameters");
        ImGui::InputInt("max iter/frame", &gbl::max_fps);
        if(ImGui::InputInt("display count", &h_params.nbBodies)) h_params.nbBodies = h_params.nbBodies > MAXBODYCOUNT ? MAXBODYCOUNT : h_params.nbBodies;

        
       
        ImGui::SeparatorText("Options");
        if(ImGui::Button("regenerate")){
            gbl::paused = true;
            nbd::randomBodies(nbd::h_bodies1, MAXBODYCOUNT);
            if(gbl::mode == GPU_MODE) checkCudaErrors( cudaMemcpy(nbd::d_bodies1, nbd::h_bodies1, MAXBODYCOUNT*sizeof(Body), cudaMemcpyHostToDevice) );
            gbl::paused = false;
        }
        ImGui::DragFloat("dpos", &nbd::dpos,0.01f,0.0f,2.0f);
        ImGui::DragFloat("dvel", &nbd::dvel, 0.01f, 0.0f, 2.0f, "%.5f");
        ImGui::DragFloat("min mass", &nbd::minmass,0.5f,0.1f,10.0f);
        ImGui::DragFloat("max mass", &nbd::maxmass,0.5f,1.0f,10.0f);


        ImGui::End();
    }

    void wdw_additional(){
        ImGui::SeparatorText("GPU mode");
        static int current_gpu_mode = 0;
        const char* items[] = { "default", "shared"};

        if (ImGui::Combo("Combo", &current_gpu_mode, items, IM_ARRAYSIZE(items))) {
            switch (current_gpu_mode)
            {
            //TODO CHANGE CALLBACK
            case 0: /* gbl::display = gpu::imp_Bugs_default; */ break;
            case 1: /* gbl::display = gpu::imp_Bugs_shared; */ break;
            default: break;
            }
        }

        ImGui::SameLine(); HelpMarker(
                    "refer to readme.md for additional information\n"
                    "on how shared mode works");
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
        


        //framerate
        gbl::max_fps = 20;
        

        //nbd
        h_params.nbBodies = 32;

        glClearColor(0.0,0.0,0.0,0.0);
        glColor4f(1.0,1.0,1.0,1.0);
        glDisable(GL_DEPTH_TEST);
        glPointSize(2.0f);

        nbd::randomBodies(nbd::h_bodies1, MAXBODYCOUNT);
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
        if(!gbl::paused){
            cameraApply();
            glClear(GL_COLOR_BUFFER_BIT);
            glEnableClientState(GL_VERTEX_ARRAY);
            glVertexPointer(4, GL_FLOAT, sizeof(Body), &(nbd::h_bodies1->pos));
	        glDrawArrays(GL_POINTS, 0, h_params.nbBodies);
	        glDisableClientState(GL_VERTEX_ARRAY);
        }
  

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