#include "util.hpp"
#include <random>
#include <math.h>

#include "camera.cpp"


#define DEBUGA(x) std::cout << __FILE__ << ":" << __LINE__ << " - " << #x << " = " << (x) << std::endl
#define DEBUGV(x) std::cout << (x)


#define TITLE "NBODY"


//mandatory forward declaration
namespace wdw{
    void nbodyParam();
}
namespace gbl{
    int max_fps;
}
namespace gpu{
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
struct MyCol{ //used for passing color to gpu for type compatibility reasons
    float x, y, z, w;
    __device__ __host__ MyCol(){}
    __device__ __host__ MyCol(float x, float y, float z, float w) : x(x), y(y), z(z), w(w){}
    __device__ __host__ MyCol(float* c) : x(c[0]), y(c[1]), z(c[2]), w(c[3]){}
    __device__ __host__ MyCol(const MyCol& c) : x(c.x), y(c.y), z(c.z), w(c.w){}
};

struct Body{
    float3 pos; 
    float3 vel;
    float3 col;
    float mass;
};

namespace nbd{
    Body* h_bodies1;
    Body* h_bodies2;
    Body* d_bodies1;
    Body* d_bodies2;

    static int MAXBODYCOUNT = 1024;
    float dpos = 2.5f;
    float dvel = 0.0001f;
    static float minmass = 1.0f;
    static float maxmass = 5.0f;

    static int current_col_mode = 1;
    static float PARAM_slow[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
    static float PARAM_fast[4] = { 0.0f, 1.0f, 0.0f, 1.0f };


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
            //body.pos.w = 1.0f;

            //vel
            x = (2*((rand()%1000)/1000.0f)-1);
            y = (2*((rand()%1000)/1000.0f)-1);
            z = (2*((rand()%1000)/1000.0f)-1);
            r = (rand()%1000)/1000.0f/sqrt(x*x+y*y+z*z);
            body.vel.x = r*dvel*x;
            body.vel.y = r*dvel*y;
            body.vel.z = r*dvel*z;
            //body.vel.w = 1.0f;

            //mass
            body.mass = minmass+(maxmass-minmass)*((rand()%1000)/1000.0f);
        }
    }

    __host__ __device__ float3 add(const float3& a, const float3& b) {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }
    __host__ __device__ float3 minus(const float3& a, const float3& b) {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }
    __host__ __device__ float3 multiply(float s, const float3& v) {
        return make_float3(s * v.x, s * v.y, s * v.z);
    }

    __host__ __device__ inline float dist(const Body& b1, const Body& b2){
        return sqrtf(   (b1.pos.x - b2.pos.x) * (b1.pos.x - b2.pos.x) +
                        (b1.pos.y - b2.pos.y) * (b1.pos.y - b2.pos.y) +
                        (b1.pos.z - b2.pos.z) * (b1.pos.z - b2.pos.z));
    }
    __host__ __device__ inline float length2(float3 v) {
        return v.x * v.x + v.y * v.y + v.z * v.z;
    }
    __host__ __device__ void updtColors(Body& body, float minclamp, float maxclamp, MyCol* c1, MyCol* c2){
        float speed = sqrtf(length2(body.vel));
        speed = speed < minclamp ? minclamp : speed;
        speed = speed > maxclamp ? maxclamp : speed;
        float t = (speed - minclamp) / (maxclamp - minclamp);
        body.col.x = t*c2->x + (1-t) * c1->x;
        body.col.y = t*c2->y + (1-t) * c1->y;
        body.col.z = t*c2->z + (1-t) * c1->z;
    }



}



struct Param {
    float scale;
    float mx, my; //mousepose
    Complex offset;

    //nbd
    int nbBodies; //nb à afficher
    float G;
    float EPS2;

    bool updtcol;
    float slowspeed;
    float fastspeed;
    MyCol col_slow;
    MyCol col_fast;
};
Param h_params;
__constant__ Param d_params;



namespace cpu{
    void imp_NBody();

    void init(){
        nbd::h_bodies1 = (Body*) malloc(nbd::MAXBODYCOUNT * sizeof(Body));
        nbd::h_bodies2 = (Body*) malloc(nbd::MAXBODYCOUNT * sizeof(Body));
        gbl::display = imp_NBody;
    }
    void clean(){
        free(nbd::h_bodies1); nbd::h_bodies1 = nullptr;
        free(nbd::h_bodies2); nbd::h_bodies2 = nullptr;
    }
    void reinit(){
        nbd::h_bodies1 = (Body*)realloc(nbd::h_bodies1, nbd::MAXBODYCOUNT * sizeof(Body));
        nbd::h_bodies2 = (Body*)realloc(nbd::h_bodies2, nbd::MAXBODYCOUNT * sizeof(Body));
    }


    void imp_NBody() {
		// your N-body algorithm here!
        for (int i=0;i<h_params.nbBodies;i++){
            float3 acc = {0.0f, 0.0f, 0.0f};

            for (int j = 0; j < h_params.nbBodies; j++) {
                if (i != j) { 
                    float3 r = nbd::minus(nbd::h_bodies1[j].pos, nbd::h_bodies1[i].pos);
                    float d = nbd::length2(r) + h_params.EPS2;  
                    float factor = h_params.G * nbd::h_bodies1[j].mass / sqrtf(d * d * d);
                    acc = nbd::add(acc, nbd::multiply(factor, r));
                }
            }
            nbd::h_bodies2[i].pos = nbd::add(nbd::h_bodies1[i].pos, nbd::h_bodies1[i].vel);
            nbd::h_bodies2[i].vel = nbd::add(nbd::h_bodies1[i].vel, acc);
            if(h_params.updtcol) nbd::updtColors(nbd::h_bodies2[i],h_params.slowspeed, h_params.fastspeed, &h_params.col_slow, &h_params.col_fast);
        }

        std::swap(nbd::h_bodies1, nbd::h_bodies2);
	}

}//end namespace cpu

namespace gpu{
    void (*gpu_cbk)();
    void imp_NBody();

    void init(){
        checkCudaErrors( cudaMallocHost((void**) &nbd::h_bodies1, nbd::MAXBODYCOUNT * sizeof(Body)) );
	    checkCudaErrors( cudaMalloc((void**)&nbd::d_bodies1, nbd::MAXBODYCOUNT * sizeof(Body)) );
	    checkCudaErrors( cudaMalloc((void**)&nbd::d_bodies2, nbd::MAXBODYCOUNT * sizeof(Body)) );
        gbl::display = gpu_cbk; //uses intermediate gpu cbk for saving mode when switching back to cpu
    }
    void clean(){
        checkCudaErrors( cudaFreeHost(nbd::h_bodies1));
	    checkCudaErrors( cudaFree(nbd::d_bodies1) );
	    checkCudaErrors( cudaFree(nbd::d_bodies2) );

    }
    void reinit(){
        clean();
        init();
    }

    void setDeviceParameters(const Param& params) {
        checkCudaErrors( cudaMemcpyToSymbol(d_params, &params, sizeof(Param)) );
    }



    __global__ void kernelNBody(Body* oldBodies, Body* newBodies) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index < d_params.nbBodies) {
            //access constant memory once per thread !
            //Param t_params = d_params;
            //float scale = t_params.scale;
            //float sx = t_params.mx;
            //float sy = t_params.my;

            int i = index;
            float3 acc = {0.0f, 0.0f, 0.0f};

            for (int j = 0; j < d_params.nbBodies; j++) {
                if (i != j) { 
                    float3 r = nbd::minus(oldBodies[j].pos, oldBodies[i].pos);
                    float d = nbd::length2(r) + d_params.EPS2;  
                    float factor = d_params.G * oldBodies[j].mass / sqrtf(d * d * d);
                    acc = nbd::add(acc, nbd::multiply(factor, r));
                }
            }
            newBodies[i].pos = nbd::add(oldBodies[i].pos, oldBodies[i].vel);
            newBodies[i].vel = nbd::add(oldBodies[i].vel, acc);
            if(d_params.updtcol) nbd::updtColors(newBodies[i],d_params.slowspeed, d_params.fastspeed, &d_params.col_slow, &d_params.col_fast);         
		}
	}

    __global__ void kernelNBodySHARED(Body* oldBodies, Body* newBodies){
        //warning : harcodec values for block of size 256
        const int BDIM = 256;
        int index = threadIdx.x + blockIdx.x * blockDim.x; //which body will be updated same
        __shared__ float3 shared_pos[BDIM]; //the 256 position of nbodies
        __shared__ float shared_mass[BDIM]; //the 265 mass of nbodies
		if (index < d_params.nbBodies) {
            //access constant memory once per thread !
            //Param t_params = d_params;
            //float scale = t_params.scale;
            //float sx = t_params.mx;
            //float sy = t_params.my;

            int i = index;
            float3 mypos = oldBodies[i].pos;
            float3 acc = {0.0f, 0.0f, 0.0f};

            int loadIndex =0;
            unsigned int tid = threadIdx.x;
            while(loadIndex < d_params.nbBodies){
                if(tid+loadIndex< d_params.nbBodies){
                    shared_pos[tid] = oldBodies[tid+loadIndex].pos; //prevent illegal memory access at the end
                    shared_mass[tid] = oldBodies[tid+loadIndex].mass;
                } 
                else {
                    shared_pos[tid] = {0.0f,0.0f,0.0f};
                    shared_mass[tid] = 0.0f; //give a mass to zero for tails at the end
                }
                __syncthreads();

                //sum partial acc
                for (int j = 0; j < BDIM; j++) {
                    if (i != j) { 
                        float3 r = nbd::minus(shared_pos[j], mypos);
                        float d = nbd::length2(r) + d_params.EPS2;  
                        float factor = d_params.G * shared_mass[j]/ sqrtf(d * d * d);
                        acc = nbd::add(acc, nbd::multiply(factor, r));
                    }
                }

                //move on to the next batch
                loadIndex += BDIM;
            }


            newBodies[i].pos = nbd::add(oldBodies[i].pos, oldBodies[i].vel);
            newBodies[i].vel = nbd::add(oldBodies[i].vel, acc);
            if(d_params.updtcol) nbd::updtColors(newBodies[i],d_params.slowspeed, d_params.fastspeed, &d_params.col_slow, &d_params.col_fast);         
		}
    }

    void imp_NBody(){
        //initialisation
        int N = h_params.nbBodies;
		int M = 256;

        //computation
        kernelNBody << <(N + M - 1) / M, M >> > (nbd::d_bodies1, nbd::d_bodies2);
        checkKernelErrors();

        //fecth newly computed bodies from GPU to CPU and swap grid
        checkCudaErrors( cudaMemcpy(nbd::h_bodies1, nbd::d_bodies2, N * sizeof(Body), cudaMemcpyDeviceToHost));
        std::swap(nbd::d_bodies1, nbd::d_bodies2);
    }

    void imp_NBodySHARED(){
        //initialisation
        int N = h_params.nbBodies;
		int M = 256;

        //computation
        kernelNBodySHARED << <(N + M - 1) / M, M >> > (nbd::d_bodies1, nbd::d_bodies2);
        checkKernelErrors();

        //fecth newly computed bodies from GPU to CPU and swap grid
        checkCudaErrors( cudaMemcpy(nbd::h_bodies1, nbd::d_bodies2, N * sizeof(Body), cudaMemcpyDeviceToHost));
        std::swap(nbd::d_bodies1, nbd::d_bodies2);
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

    void nbodyParam(){
        ImGui::Begin("Nbodies");

        ImGui::SeparatorText("Simulation Settings");
        ImGui::InputFloat("gravitaional const", &h_params.G,0,0,"%.7f");
        ImGui::InputFloat("EPSILON²", &h_params.EPS2);
        ImGui::DragFloat("min mass", &nbd::minmass,0.5f,0.1f,10.0f);
        ImGui::DragFloat("max mass", &nbd::maxmass,0.5f,1.0f,10.0f);

 

        ImGui::SeparatorText("Advanced parameters");
        ImGui::InputInt("max iter/frame", &gbl::max_fps);
        static int buffermaxcount = 1024;
        ImGui::InputInt("loaded", &buffermaxcount);
        ImGui::SameLine(); HelpMarker(
                "The number of Bodies loaded in memory\n"
                "press apply to reload");
        ImGui::SameLine();
        if(ImGui::Button("apply")){
            gbl::paused = true;
            nbd::MAXBODYCOUNT = buffermaxcount;
            h_params.nbBodies = h_params.nbBodies > nbd::MAXBODYCOUNT ? nbd::MAXBODYCOUNT : h_params.nbBodies;
            reinit();
            nbd::randomBodies(nbd::h_bodies1, nbd::MAXBODYCOUNT);
            if(gbl::mode == GPU_MODE) checkCudaErrors( cudaMemcpy(nbd::d_bodies1, nbd::h_bodies1, nbd::MAXBODYCOUNT*sizeof(Body), cudaMemcpyHostToDevice) );
            gbl::paused = false;
        }
        //if(ImGui::InputInt("display count", &h_params.nbBodies)) h_params.nbBodies = h_params.nbBodies > nbd::MAXBODYCOUNT ? nbd::MAXBODYCOUNT : h_params.nbBodies;
        ImGui::SliderInt("displayed", &h_params.nbBodies, 1, nbd::MAXBODYCOUNT);
        ImGui::SameLine(); HelpMarker(
                "The number of Bodies processed\n"
                "(bodies hidden aren't evalueated when\n"
                "updating positions)");

        ImGui::SeparatorText("Options");
        if(ImGui::Button("regenerate")){
            gbl::paused = true;
            nbd::randomBodies(nbd::h_bodies1, nbd::MAXBODYCOUNT);
            if(gbl::mode == GPU_MODE) checkCudaErrors( cudaMemcpy(nbd::d_bodies1, nbd::h_bodies1, nbd::MAXBODYCOUNT*sizeof(Body), cudaMemcpyHostToDevice) );
            gbl::paused = false;
        }
        ImGui::DragFloat("dpos", &nbd::dpos, 0.01f, 1.0f, 5.0f, "%.5f");
        ImGui::DragFloat("dvel", &nbd::dvel, 0.01f, 0.0f, 1.0f, "%.5f");


        ImGui::SeparatorText("Coloring");
        const char* items[] = { "all white", "rgb cube", "speed scale"};

        if (ImGui::Combo("Color scheme", &nbd::current_col_mode, items, IM_ARRAYSIZE(items))) {
            h_params.updtcol = nbd::current_col_mode == 2;
        }
        switch (nbd::current_col_mode)
        {
        case 0: /* gbl::display = gpu::imp_Bugs_default; */ 
            break;
        case 1: /* gbl::display = gpu::imp_Bugs_shared; */ break;
        case 2: /* gbl::display = gpu::imp_Bugs_shared; */
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.3f);
            ImGui::DragFloat("##slow", &h_params.slowspeed, 0.01f, 0.0f, 1.0f, "%.3f"); //todo here setnexitem widht
            ImGui::SameLine(); ImGui::Text("Slow - Fast");
            ImGui::SameLine(); HelpMarker("Linear interpolation between the 2 colors factored by speed");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.3f);
            ImGui::DragFloat("##fast", &h_params.fastspeed, 0.01f, 0.0f, 1.0f, "%.3f");
            if(ImGui::ColorEdit4("##colslow", (float*)&nbd::PARAM_slow, ImGuiColorEditFlags_DisplayRGB | ImGuiColorEditFlags_InputRGB | ImGuiColorEditFlags_Float)) 
                h_params.col_slow = MyCol{nbd::PARAM_slow};
            if(ImGui::ColorEdit4("##colfast", (float*)&nbd::PARAM_fast, ImGuiColorEditFlags_DisplayRGB | ImGuiColorEditFlags_InputRGB | ImGuiColorEditFlags_Float)) 
                h_params.col_fast = MyCol{nbd::PARAM_fast};
            
            break;
        default: break;
        }



        ImGui::End();
    }

    void wdw_additional(){
        ImGui::SeparatorText("GPU mode");
        static int current_gpu_mode = 0;
        const char* items[] = { "version 1", "version 2"};

        if (ImGui::Combo("GPU", &current_gpu_mode, items, IM_ARRAYSIZE(items))) {
            switch (current_gpu_mode)
            {
            //save cbk for switching between modes
            case 0: gpu::gpu_cbk = gpu::imp_NBody; break;
            case 1: gpu::gpu_cbk = gpu::imp_NBodySHARED; break;
            default: break;
            }
            gbl::display = gpu::gpu_cbk;
        }
        if(current_gpu_mode == 1){
            ImGui::SameLine(); HelpMarker(
                    "Version 1 : default implementation N*N access"
                    "Version 2 : preload batchs of pos and mass into shared memory\n"
                    "to lower global memory access count");
        }
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
    nbd::randomBodies(nbd::h_bodies1, nbd::MAXBODYCOUNT);
    if(gbl::mode == GPU_MODE) checkCudaErrors( cudaMemcpy(nbd::d_bodies1, nbd::h_bodies1, nbd::MAXBODYCOUNT*sizeof(Body), cudaMemcpyHostToDevice) );
}
void reinit(){
    cudaDeviceSynchronize();
	switch (gbl::mode){
        case CPU_MODE: cpu::reinit(); break;
        case GPU_MODE: gpu::reinit(); break;
	}
    cudaDeviceSynchronize();
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
        h_params.mx = h_params.offset.a + (float)(0.003f*(xpos - gbl::SCREEN_X / 2));
        h_params.my = h_params.offset.b - (float)(0.003f*(ypos - gbl::SCREEN_Y / 2));
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
                h_params.offset.a += (float)(0.003f * (xpos - gbl::SCREEN_X / 2));
		        h_params.offset.b += -(float)(0.003f * (ypos - gbl::SCREEN_Y / 2));
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

void runColroing1(){
    cameraApply(-h_params.mx,h_params.my,h_params.scale);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, sizeof(Body), &(nbd::h_bodies1->pos));
    glDrawArrays(GL_POINTS, 0, h_params.nbBodies);
    glDisableClientState(GL_VERTEX_ARRAY);
}
void runColroing2(){
    cameraApply(-h_params.mx,h_params.my,h_params.scale);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(3, GL_FLOAT, sizeof(Body), &(nbd::h_bodies1->pos));
    glColorPointer(3, GL_FLOAT, sizeof(Body), &(nbd::h_bodies1->pos)); //todo replace with color
    glDrawArrays(GL_POINTS, 0, h_params.nbBodies);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
}
void runColroing3(){
    cameraApply(-h_params.mx,h_params.my,h_params.scale);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(3, GL_FLOAT, sizeof(Body), &(nbd::h_bodies1->pos));
    glColorPointer(3, GL_FLOAT, sizeof(Body), &(nbd::h_bodies1->col)); //todo replace with color
    glDrawArrays(GL_POINTS, 0, h_params.nbBodies);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
}


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
        h_params.scale = 1.0f;
        h_params.mx = 0.0f;
        h_params.my = 0.0f;
        h_params.offset = Complex(0.0f, 0.0f);
        


        //framerate
        gbl::max_fps = 60;
        
        //gpu modes
        gpu::gpu_cbk = gpu::imp_NBody;

        //nbd
        h_params.nbBodies = 32;
        h_params.G = 0.0000001f;
        h_params.EPS2 = 0.1f;

        h_params.updtcol = false; //disable color computation
        h_params.slowspeed = 0;
        h_params.fastspeed = 0.03;
        h_params.col_slow = MyCol(nbd::PARAM_slow);
        h_params.col_fast = MyCol(nbd::PARAM_fast);

        glClearColor(0.3,0.3,0.3,1.0);
        glColor4f(1.0,1.0,1.0,1.0);
        glDisable(GL_DEPTH_TEST);
        glPointSize(2.0f);
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
            wdw::nbodyParam();
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
            switch (nbd::current_col_mode)
            {
            case 0: runColroing1(); break;;
            case 1: runColroing2(); break;;
            case 2: runColroing3(); break;;            
            }
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