#include "util.hpp"
#include <random>

#define TITLE "ray tracer"
#define INF 2e10f

//mandatory forward declaration
namespace wdw{
    void rayTracerParam();
}
namespace gbl{
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

struct Sphere {
	float r, g, b;
	float radius;
	float x, y, z;
    __host__ __device__ Sphere(){}
	__host__ __device__ Sphere(float r, float g, float b, float rad, float x, float y, float z) : r(r), g(g), b(b), radius(rad), x(x),y(y),z(z){}

	__host__ __device__ float hit(float cx, float cy, float* sh, float ox, float oy) const {
		float dx = ox + cx - x;
		float dy = oy + cy - y;
		float dz2 = radius * radius - dx * dx - dy * dy;
		if (dz2 > 0) {
			float dz = sqrtf(dz2);
			*sh = dz / radius;
			return dz + z;
		}
		return -INF;
	}
};

namespace rtc{
    //colors (messy architectures because erm ... types compatibility)
    static float PARAM_ambient[4] = { 0.2f, 0.4f, 0.2f, 1.0f };
    static int streamCount = 4;

    //global variables
    static int max_nb_sphere = 500;
    static bool use_cst_mem = false;
    static bool use_streams = false;
    Sphere* h_spheres;
    Sphere* d_spheres;
    const int SIZECONST = 400;
    __constant__ Sphere cm_spheres[SIZECONST]; //hardcoded

    //streams for stream implementation //improper implementation in term of architecture
    cudaStream_t stream[16]; // array of N streams

    inline float rndf(float min, float max) {
        return min + ((float)(rand() % 10000) / 10000) * (max-min);
    }

    void loadSpheres(){
        checkCudaErrors( cudaMallocHost((void**) &h_spheres, max_nb_sphere * sizeof(Sphere)) );
	    checkCudaErrors( cudaMalloc((void**)&d_spheres, max_nb_sphere * sizeof(Sphere)) );

        for(int i=0; i < max_nb_sphere; i++){
            // generate a random sphere
            float r = rndf(0.0f, 1.0f);
            float g = rndf(0.0f, 1.0f);
            float b = rndf(0.0f, 1.0f);
            float radius = rndf(0.1f, 0.4f);
            float x = rndf(-1.0f, 1.0f);
            float y = rndf(-1.0f, 1.0f);
            float z = rndf(-1.0f, 1.0f);

            h_spheres[i] = Sphere(r, g, b, radius, x, y, z);
        }
        checkCudaErrors( cudaMemcpy(d_spheres, h_spheres, max_nb_sphere*sizeof(Sphere), cudaMemcpyHostToDevice) );
        checkCudaErrors( cudaMemcpyToSymbol(cm_spheres, h_spheres, SIZECONST*sizeof(Sphere)) ); //hardcoded
    }
    void unloadSpheres(){
        checkCudaErrors( cudaFreeHost(h_spheres) );
	    checkCudaErrors( cudaFree(d_spheres) );
    }



}

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

    int displayCount;

    MyCol ambientLight;
    float ambient_intensity;
};
Param h_params;
__constant__ Param d_params;



namespace cpu{
    void imp_raytracer();

    void init(){
        gbl::pixels = (float4*)malloc(gbl::SCREEN_X*gbl::SCREEN_Y*sizeof(float4));
        gbl::display = imp_raytracer;
    }
    void clean(){
        free(gbl::pixels);
    }
    void reinit(){
        gbl::pixels = (float4*)realloc(gbl::pixels, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(float4));
    }


    void imp_raytracer() {

		int i, j;
		for (i = 0; i < gbl::SCREEN_Y; i++){
			for (j = 0; j < gbl::SCREEN_X; j++){
				float x = (float)(h_params.scale * (j - gbl::SCREEN_X / 2));
				float y = (float)(h_params.scale * (i - gbl::SCREEN_Y / 2));
				float4* p = gbl::pixels + (i * gbl::SCREEN_X + j);
				// default: black
                p->x = 0.0f;
                p->y = 0.0f;
                p->z = 0.0f;
                p->w = 1.0f;
                float dmin = -INF + 30;
                for (int k = 0; k < h_params.displayCount; k++) {
                    float sha = 0;
                    const Sphere& sphere = rtc::h_spheres[k];
                    float ds = sphere.hit(x, y, &sha, h_params.mx, h_params.my);
                    if (ds > dmin) {
                        dmin = ds;
                        float ambr = h_params.ambientLight.x;
                        float ambg = h_params.ambientLight.y;
                        float ambb = h_params.ambientLight.z;
                        float ai = h_params.ambient_intensity;
                        p->x = (ambr*ai) + (sha * sphere.r) * (1 - (ambr*ai));
                        p->y = (ambg*ai) + (sha * sphere.g) * (1 - (ambg*ai));
                        p->z = (ambb*ai) + (sha * sphere.b) * (1 - (ambb*ai));
                    }
                }

			}
	    }
	}

}//end namespace cpu

namespace gpu{
    void imp_RayTracer();

    void init(){
        checkCudaErrors( cudaMallocHost((void**) &gbl::pixels, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(float4)) );
	    checkCudaErrors( cudaMalloc((void**)&gbl::d_pixels, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(float4)) );
        gbl::display = imp_RayTracer;
    }
    void clean(){
        checkCudaErrors( cudaFreeHost(gbl::pixels));
	    checkCudaErrors( cudaFree(gbl::d_pixels) );
    }
    void reinit(){
        clean();
        init();
    }

    void setDeviceParameters(const Param& params) {
        checkCudaErrors( cudaMemcpyToSymbol(d_params, &params, sizeof(Param)) );
        //checkCudaErrors( cudaMemcpyToSymbolAsync(d_params, &params, sizeof(Param), 0, cudaMemcpyHostToDevice, rtc::stream[0]) );
        //cudaMemcpyToSymbolAsync(d_data, &h_data, sizeof(int), 0, cudaMemcpyHostToDevice, stream);

    }



    __global__ void kernelRayTracer(float4* d_pixels, const Sphere* spheres, int SCREENX, int SCREENY) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index < SCREENX * SCREENY) {
            //access constant memory once per thread !
            Param t_params = d_params;

            //deduce i, j (pixel coordinate) from threadIdx, blockIdx 
            int i = index / SCREENX;
		    int j = index - i * SCREENX;

            float x = (float)(t_params.scale * (j - SCREENX / 2));
            float y = (float)(t_params.scale * (i - SCREENY / 2));
            float4* p = d_pixels + (i * SCREENX + j);
            // default: black
            p->x = 0.0f; p->y = 0.0f; p->z = 0.0f; p->w = 1.0f; 
            float dmin = -INF + 30;
            for (int k = 0; k < t_params.displayCount; k++) {
                float sha = 0;
                const Sphere& sphere = spheres[k];
                float ds = sphere.hit(x, y, &sha, t_params.mx, t_params.my);
                if (ds > dmin) {
                    dmin = ds;
                    float ambr = t_params.ambientLight.x;
                    float ambg = t_params.ambientLight.y;
                    float ambb = t_params.ambientLight.z;
                    float ai = t_params.ambient_intensity;
                    p->x = (ambr*ai) + (sha * sphere.r) * (1 - (ambr*ai));
                    p->y = (ambg*ai) + (sha * sphere.g) * (1 - (ambg*ai));
                    p->z = (ambb*ai) + (sha * sphere.b) * (1 - (ambb*ai));
                }
            }

        }
	}

    //unfortunately forced to use another kernel because can't use constant memomy as const ptr
    __global__ void kernelRayTracerCONSTANT(float4* d_pixels, int SCREENX, int SCREENY, int start_pos = 0) {
		int index = threadIdx.x + blockIdx.x * blockDim.x+start_pos; 
        //use start_pos when using multiples streams
		if (index < SCREENX * SCREENY) {
            //access constant memory once per thread !
            Param t_params = d_params;

            //deduce i, j (pixel coordinate) from threadIdx, blockIdx 
            int i = index / SCREENX;
		    int j = index - i * SCREENX;

            float x = (float)(t_params.scale * (j - SCREENX / 2));
            float y = (float)(t_params.scale * (i - SCREENY / 2));
            float4* p = d_pixels + (i * SCREENX + j);
            // default: black
            p->x = 0.0f; p->y = 0.0f; p->z = 0.0f; p->w = 1.0f; 
            float dmin = -INF + 30;
            for (int k = 0; k < t_params.displayCount; k++) {
                float sha = 0;
                const Sphere& sphere = rtc::cm_spheres[k];
                float ds = sphere.hit(x, y, &sha, t_params.mx, t_params.my);
                if (ds > dmin) {
                    dmin = ds;
                    float ambr = t_params.ambientLight.x;
                    float ambg = t_params.ambientLight.y;
                    float ambb = t_params.ambientLight.z;
                    float ai = t_params.ambient_intensity;
                    p->x = (ambr*ai) + (sha * sphere.r) * (1 - (ambr*ai));
                    p->y = (ambg*ai) + (sha * sphere.g) * (1 - (ambg*ai));
                    p->z = (ambb*ai) + (sha * sphere.b) * (1 - (ambb*ai));
                }
            }

        }
	}


    void imp_RayTracer(){
        //initialisation
        int N = gbl::SCREEN_X * gbl::SCREEN_Y;
		int M = 256;

        //computation
        if(!rtc::use_streams){
            if(!rtc::use_cst_mem) kernelRayTracer << <(N + M - 1) / M, M >> > (gbl::d_pixels, rtc::d_spheres, gbl::SCREEN_X, gbl::SCREEN_Y);
            else kernelRayTracerCONSTANT << <(N + M - 1) / M, M >> > (gbl::d_pixels, gbl::SCREEN_X, gbl::SCREEN_Y);

            checkKernelErrors();
		    checkCudaErrors( cudaMemcpy(gbl::pixels, gbl::d_pixels, N * sizeof(float4), cudaMemcpyDeviceToHost) ); //get pixels values from gpu
        }
        //no implementation for stream + gbl mem bc useless
        else{
            int workPerStream = (N + rtc::streamCount - 1) / rtc::streamCount; // Divide work equally across streams
            for (int k=0;k<rtc::streamCount;k++) { // do 1/Nth of the work
                int startIdx = k * workPerStream;
                int endIdx = min(N, (k + 1) * workPerStream); // Ensure last stream doesn't exceed bounds
                int chunkSize = endIdx - startIdx;
                
                //kernelRayTracerCONSTANT << <(N + M - 1) / M, M, 0, rtc::stream[k] >> > (gbl::d_pixels, gbl::SCREEN_X, gbl::SCREEN_Y);

                // Launch kernel for this chunk
                kernelRayTracerCONSTANT<<<(chunkSize + M - 1) / M, M, 0, rtc::stream[k]>>>(
                    gbl::d_pixels, gbl::SCREEN_X, gbl::SCREEN_Y, startIdx);
                

                //checkKernelErrors();
                //checkCudaErrors( ...)

                // std::cout << "\nstream k =  :" << k << "\n";
                // std::cout << "startIdx :" << startIdx << "\t";
                // std::cout << "endIdx :" << endIdx << "\t";
                // std::cout << "chunksize :" << chunkSize << "\t";

                cudaMemcpyAsync(gbl::pixels + startIdx, gbl::d_pixels + startIdx,
                        chunkSize * sizeof(float4), cudaMemcpyDeviceToHost, rtc::stream[k]);
            }
        }
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

    void rayTracerParam(){
        ImGui::Begin("Ray tracer");

        
        //float inputWidth = ImGui::CalcTextSize("-0.000").x + ImGui::GetStyle().FramePadding.x * 2;
        ImGui::Text("Camera Coordinate : %.2f/%.2f",h_params.mx, h_params.my);

        ImGui::SeparatorText("params : ");
        ImGui::InputInt("loaded sphere", &rtc::max_nb_sphere);
        ImGui::SameLine(); HelpMarker("the number of sphere loaded into memory");
        ImGui::SliderInt("displayed spheres", &h_params.displayCount, 1, rtc::max_nb_sphere);
        ImGui::SameLine(); HelpMarker("the number that will be displayed");
        if(ImGui::Button("regenerate")){
            gbl::paused = true;
            rtc::unloadSpheres();
            rtc::loadSpheres();
            gbl::paused = false;
        }
        ImGui::SameLine(); HelpMarker("get a new random set of spheres");

        ImGui::SeparatorText("The ambiant light color");
        ImGui::DragFloat("Ambient Intensity", &h_params.ambient_intensity, 0.005f,0.001f,1.0f);
        if(ImGui::ColorEdit4("##ambient", (float*)&rtc::PARAM_ambient, ImGuiColorEditFlags_DisplayRGB | ImGuiColorEditFlags_InputRGB | ImGuiColorEditFlags_Float)) 
            h_params.ambientLight = MyCol{rtc::PARAM_ambient};
        ImGui::End();
    }

    void wdw_additional(){
        ImGui::SeparatorText("GPU mode");
        static int current_gpu_mode = 0;
        const char* items[] = { "no chagnes", "here", "for now"};

        if (ImGui::Combo("##gpumode", &current_gpu_mode, items, IM_ARRAYSIZE(items))) {
            switch (current_gpu_mode)
            {
            case 0: /* gbl::display = gpu::imp_Bugs_default; */ break;
            case 1: /* gbl::display = gpu::imp_Bugs_shared; */ break;
            default: break;
            }
        }
        ImGui::SameLine(); 
        switch (current_gpu_mode)
            {
            case 0: 
                HelpMarker("version 1 : each trade <=> 1 pixel");
                break;
            case 1:
                //ImGui::SetNextItemWidth(20); 
                //ImGui::InputInt("stream count :", &rtc::streamCount, 0, 0);
                break;
            default: break;
            }
        ImGui::Checkbox("use constant memory", &rtc::use_cst_mem);
        ImGui::SameLine(); HelpMarker("version 2 : sphere array in GPU memory");
        //we can't do this bc we can't treat cm as pointer
        // if ... rtc::gpuSpheres = rtc::use_cst_mem ? (const Sphere*)rtc::cm_spheres : rtc::d_spheres;



        ImGui::SetNextItemWidth(25); 
        if (rtc::use_streams) {
            ImGui::InputInt("nb stream", &rtc::streamCount, 0, 0, ImGuiInputTextFlags_ReadOnly);
        } else {
            ImGui::InputInt("nb stream", &rtc::streamCount, 0, 0);
        }
        ImGui::SameLine(); HelpMarker("16 max");
        if(rtc::streamCount > 16) rtc::streamCount = 16;
        ImGui::SameLine();

        if(ImGui::Checkbox("use streams memory", &rtc::use_streams)){
            gbl::paused = true;
            if(rtc::use_streams){
                for (int k=0;k<rtc::streamCount;k++) cudaStreamCreate(&rtc::stream[k]);
            }
            else{
                for (int k=0;k<rtc::streamCount;k++) cudaStreamDestroy(rtc::stream[k]);
            }
            gbl::paused = false;
        }
        ImGui::SameLine(); HelpMarker("version 3 : streams for task parallelization\n"
            "We can notice a small improvement when using streams. (35 -> 40 fps with 500 spheres)\n"
            "This is very likely because the kernel is computationally light\n"
            "and thus paralellization is almost irrelevant");
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


        
        rtc::loadSpheres();
        h_params.ambientLight = MyCol{rtc::PARAM_ambient};
        h_params.displayCount = 10;
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
        if(gbl::otherWindow) {
            utl::wdw_info(gbl::mode, gbl::SCREEN_X,gbl::SCREEN_Y,gbl::currentFPS);
            wdw::rayTracerParam();
        }
        

        /* Render */
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