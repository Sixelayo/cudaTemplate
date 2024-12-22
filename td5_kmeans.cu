#include "util.hpp"
#include <random>
#include <math.h>

#include "camera.cpp"


#define DEBUGA(x) std::cout << __FILE__ << ":" << __LINE__ << " - " << #x << " = " << (x) << std::endl
#define DEBUGV(x) std::cout << (x)


#define TITLE "NBODY"


//mandatory forward declaration
namespace wdw{
    void kmeansParam();
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

struct Point{
    float3 pos;  //point position
    float3 col; //the point color (same of the cluster)
    int label; //which clust the point belongs to
};

namespace kmn{
    Point* h_points;
    Point* h_centroids;

    //value linked with buffer in wdw with buffer for chagnes
    static int NBPOINTS = 16*1024;
    static int NBCENTROIDS = 128;

    float dpoints = 1.0f;

    float3 randomColor()    {
        float3 color;
        color.x = (rand() % 1000) / 1000.0f;
        color.y = (rand() % 1000) / 1000.0f;
        color.z = (rand() % 1000) / 1000.0f;
        return color;
    }

    void randomPoints(){
        // create random points. Default colors white belonging to cluster -1 (none)
        // create artificial clusters

        //plug parm to function tempalte
        int n = NBPOINTS;
        int nbClusters = NBCENTROIDS;
        float d = dpoints;

        int i = 0;
        float x, y, z, r;
        float4* c = (float4*)malloc(nbClusters*sizeof(float4));
        float* s = (float*)malloc(nbClusters*sizeof(float));
        for (i = 0; i<nbClusters; i++)
        {
            x = (2 * ((rand() % 1000) / 1000.0f) - 1);
            y = (2 * ((rand() % 1000) / 1000.0f) - 1);
            z = (2 * ((rand() % 1000) / 1000.0f) - 1);
            r = powf(3 * (rand() % 1000) / 1000.0f, 4);

            c[i].x = r*d*x;
            c[i].y = r*d*y;
            c[i].z = r*d*z;
            c[i].w = 1.0f; // must be 1.0

            s[i] = (rand() % 1000) / 1000.0f + 0.5f;
        }

        //float4* a = (float4*)malloc(n*sizeof(float4));
        for (i = 0; i<n; i++)
        {
            int cl = rand() % NBCENTROIDS;
            x = (2 * ((rand() % 1000) / 1000.0f) - 1);
            y = (2 * ((rand() % 1000) / 1000.0f) - 1);
            z = (2 * ((rand() % 1000) / 1000.0f) - 1);
            r = powf(2 * (rand() % 1000) / 1000.0f / sqrt(x*x + y*y + z*z), 2.5);

            h_points[i].pos.x = c[cl].x + s[cl] * s[cl] * r*d*x;
            h_points[i].pos.y = c[cl].y + s[cl] * s[cl] * r*d*y;
            h_points[i].pos.z = c[cl].z + s[cl] * s[cl] * r*d*z;

            h_points[i].col = {1.0f,1.0f,1.0f}; //color set to white at first
            h_points[i].label = -1; //label set to -1 at first doesn't belong to any cluster

        }
        free(c);
        free(s);
    }
    void randomClusters(){
        for(int i=0; i < kmn::NBCENTROIDS; i++){
            kmn::h_centroids[i].pos = kmn::h_points[i].pos;
            kmn::h_centroids[i].col = randomColor();

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

    __host__ __device__ inline float length2(float3 v) {
        return v.x * v.x + v.y * v.y + v.z * v.z;
    }



}



struct Param {
    float scale;
    float mx, my; //mousepose
    Complex offset;

    //nbd
};
Param h_params;
__constant__ Param d_params;



namespace cpu{
    void imp_KMeans();

    void init(){
        kmn::h_points = (Point*) malloc(kmn::NBPOINTS * sizeof(Point));
        kmn::h_centroids = (Point*) malloc(kmn::NBCENTROIDS * sizeof(Point));
        gbl::display = imp_KMeans;
    }
    void clean(){
        free(kmn::h_points); kmn::h_points = nullptr;
        free(kmn::h_centroids); kmn::h_centroids = nullptr;
    }
    void reinit(){
        kmn::h_points = (Point*)realloc(kmn::h_points, kmn::NBPOINTS * sizeof(Point));
        kmn::h_centroids = (Point*)realloc(kmn::h_centroids, kmn::NBCENTROIDS * sizeof(Point));
    }


    void imp_KMeans() {
		// your N-body algorithm here!
        // your kmeans algorithm here
        int i;
        for (i = 0; i<kmn::NBPOINTS; i++)
        {
            kmn::h_points[i].label = i%kmn::NBCENTROIDS;
            kmn::h_points[i].col = kmn::h_centroids[kmn::h_points[i].label].col;
        }
	}

}//end namespace cpu

namespace gpu{
    void (*gpu_cbk)();
    void imp_NBody();

    void init(){
        // checkCudaErrors( cudaMallocHost((void**) &kmn::h_bodies1, kmn::MAXBODYCOUNT * sizeof(Body)) );
	    // checkCudaErrors( cudaMalloc((void**)&kmn::d_bodies1, kmn::MAXBODYCOUNT * sizeof(Body)) );
	    // checkCudaErrors( cudaMalloc((void**)&kmn::d_bodies2, kmn::MAXBODYCOUNT * sizeof(Body)) );
        gbl::display = gpu_cbk; //uses intermediate gpu cbk for saving mode when switching back to cpu
    }
    void clean(){
        // checkCudaErrors( cudaFreeHost(kmn::h_bodies1));
	    // checkCudaErrors( cudaFree(kmn::d_bodies1) );
	    // checkCudaErrors( cudaFree(kmn::d_bodies2) );

    }
    void reinit(){
        clean();
        init();
    }

    void setDeviceParameters(const Param& params) {
        checkCudaErrors( cudaMemcpyToSymbol(d_params, &params, sizeof(Param)) );
    }



    // __global__ void kernelNBody(Body* oldBodies, Body* newBodies) {
	// 	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// 	if (index < d_params.nbBodies) {
    //         //access constant memory once per thread !
    //         //Param t_params = d_params;
    //         //float scale = t_params.scale;
    //         //float sx = t_params.mx;
    //         //float sy = t_params.my;

    //         int i = index;
    //         float3 acc = {0.0f, 0.0f, 0.0f};

    //         for (int j = 0; j < d_params.nbBodies; j++) {
    //             if (i != j) { 
    //                 float3 r = kmn::minus(oldBodies[j].pos, oldBodies[i].pos);
    //                 float d = kmn::length2(r) + d_params.EPS2;  
    //                 float factor = d_params.G * oldBodies[j].mass / sqrtf(d * d * d);
    //                 acc = kmn::add(acc, kmn::multiply(factor, r));
    //             }
    //         }
    //         newBodies[i].pos = kmn::add(oldBodies[i].pos, oldBodies[i].vel);
    //         newBodies[i].vel = kmn::add(oldBodies[i].vel, acc);
    //         if(d_params.updtcol) kmn::updtColors(newBodies[i],d_params.slowspeed, d_params.fastspeed, &d_params.col_slow, &d_params.col_fast);         
	// 	}
	// }


    void imp_NBody(){
        // //initialisation
        // int N = h_params.nbBodies;
		// int M = 256;

        // //computation
        // kernelNBody << <(N + M - 1) / M, M >> > (kmn::d_bodies1, kmn::d_bodies2);
        // checkKernelErrors();

        // //fecth newly computed bodies from GPU to CPU and swap grid
        // checkCudaErrors( cudaMemcpy(kmn::h_bodies1, kmn::d_bodies2, N * sizeof(Body), cudaMemcpyDeviceToHost));
        // std::swap(kmn::d_bodies1, kmn::d_bodies2);
    }


}//end namespace gpu




namespace wdw{

    //warning, must be coherent with km::NBPOINTS and kmn::NBCENTROIDS
    static int bufferPointCount = 16*1024;
    static int bufferCentroCount = 128; 

    void applyParam(){
        gbl::paused = true;
        kmn::NBPOINTS = bufferPointCount;
        kmn::NBCENTROIDS = bufferCentroCount;
        reinit();
        //TODO HERE copy points to gpu
        //if(gbl::mode == GPU_MODE) checkCudaErrors( cudaMemcpy(kmn::d_bodies1, kmn::h_bodies1, kmn::MAXBODYCOUNT*sizeof(Body), cudaMemcpyHostToDevice) ); 
        gbl::paused = false;
    }

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

    void kmeansParam(){
        ImGui::Begin("Kmeans");


        ImGui::SeparatorText("Advanced parameters");
        ImGui::InputInt("max iter/frame", &gbl::max_fps);

        ImGui::InputInt("points", &bufferPointCount);
        ImGui::SameLine(); HelpMarker("Number of points");

        ImGui::InputInt("clusters", &bufferCentroCount);
        ImGui::SameLine(); HelpMarker("Number of centroids (cluster)");
        if(ImGui::Button("apply")) applyParam();


        ImGui::SeparatorText("Options");
        if(ImGui::Button("randomize points")){
            gbl::paused = true;
            kmn::randomPoints();
            kmn::randomClusters();
            //todo here pass to gpu
            //if(gbl::mode == GPU_MODE) checkCudaErrors( cudaMemcpy(kmn::d_bodies1, kmn::h_bodies1, kmn::MAXBODYCOUNT*sizeof(Body), cudaMemcpyHostToDevice) );
            gbl::paused = false;
        }

        ImGui::DragFloat("dpos", &kmn::dpoints, 0.01f, 0.0f,3.0f, "%.3f");




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
            //todo here addcase 1
            //case 1: gpu::gpu_cbk = gpu::imp_NBodySHARED; break;
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
    kmn::randomPoints();
    kmn::randomClusters();
    //todo here copy to gpu
    //if(gbl::mode == GPU_MODE) checkCudaErrors( cudaMemcpy(kmn::d_bodies1, kmn::h_bodies1, kmn::MAXBODYCOUNT*sizeof(Body), cudaMemcpyHostToDevice) );
}
void reinit(){
    cudaDeviceSynchronize();
	switch (gbl::mode){
        case CPU_MODE: cpu::reinit(); break;
        case GPU_MODE: gpu::reinit(); break;
	}
    kmn::randomPoints();
    kmn::randomClusters();
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
        h_params.scale = 10.0f;
        h_params.mx = 0.0f;
        h_params.my = 0.0f;
        h_params.offset = Complex(0.0f, 0.0f);
        


        //framerate
        gbl::max_fps = 60;
        
        //gpu modes
        gpu::gpu_cbk = gpu::imp_NBody;

        //kmn
        


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
            wdw::kmeansParam();
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
            cameraApply(-h_params.mx,h_params.my,h_params.scale);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glPointSize(1.0f);
            glEnableClientState(GL_VERTEX_ARRAY);
            glEnableClientState(GL_COLOR_ARRAY);
            glVertexPointer(3, GL_FLOAT, sizeof(Point), &(kmn::h_points->pos));
            glColorPointer(3, GL_FLOAT, sizeof(Point), &(kmn::h_points->col)); 
            glDrawArrays(GL_POINTS, 0, kmn::NBPOINTS);
            glDisableClientState(GL_COLOR_ARRAY);
            glDisableClientState(GL_VERTEX_ARRAY);


            glPointSize(3.0f);
            glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
            glEnableClientState(GL_VERTEX_ARRAY);
            glVertexPointer(3, GL_FLOAT, sizeof(Point), &(kmn::h_centroids->pos));//todo points position
            glDrawArrays(GL_POINTS, 0, kmn::NBCENTROIDS);
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