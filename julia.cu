#include "util.hpp"
#define TITLE "template"

/*todo
<addresse repo git>
Copier / colé du readme

*/

//mandatory forward declaration
namespace wdw{
    void julMandParam();
    void julMandPreset();
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
    int nb_iter;
    float minkowski_order;
    float threshold;
    Complex offset;
    float easing_fac_in;
    float easing_fac_out;
    MyCol color_step;
    MyCol color_easeIn;
    MyCol color_easeOut;
};
Param h_params;
__constant__ Param d_params;

namespace preset{
    void center(){
        h_params.scale = 0.0035f;
        h_params.offset =  {0,0};
    }
    void gpu_default(){
        h_params.nb_iter = 40;
        h_params.minkowski_order = 2;
        h_params.threshold = 4;
    }
    void spiral1(){
        h_params.mx = -0.5251993f;
        h_params.my = -0.5251993f;
    }
    void spiral2(){
        h_params.mx = -0.77146f;
        h_params.my = -0.10119f;
    }
    void douady(){
        h_params.mx = -0.12f;
        h_params.my = 0.75f;
    }
    void branches(){
        h_params.mx = 0.35; 
        h_params.my = 0.35f;
    }
}//end namespace prs


namespace cpu{
    void imp_Julia();

    void init(){
        gbl::pixels = (float4*)malloc(gbl::SCREEN_X*gbl::SCREEN_Y*sizeof(float4));
        gbl::display = imp_Julia;
    }
    void clean(){
        free(gbl::pixels);
    }
    void reinit(){
        gbl::pixels = (float4*)realloc(gbl::pixels, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(float4));
    }

    float juliaColor(float x, float y, float sx, float sy, int p, float order, float thresh) {
		Complex a = Complex(x, y);
		Complex seed = Complex(sx, sy);
		for (int i = 0; i < p; i++) {
			a = a * a + seed;
			if (minkowski(a, order) > thresh) {
				return 1 - (float)i / p;
			}
		}
		return 0;
	}

    void imp_Julia() {
        if(gbl::otherWindow) {
            wdw::julMandParam();
            wdw::julMandPreset();
        }

		int i, j;
		for (i = 0; i < gbl::SCREEN_Y; i++)
			for (j = 0; j < gbl::SCREEN_X; j++)
			{
				float x = (float)(h_params.scale * (j - gbl::SCREEN_X / 2));
				float y = (float)(h_params.scale * (i - gbl::SCREEN_Y / 2));
				float4* p = gbl::pixels + (i * gbl::SCREEN_X + j);
				// default: black
				p->x = 0.0f;
				p->y = 0.0f;
				p->z = cpu::juliaColor(x - h_params.offset.a, y - h_params.offset.b, h_params.mx  , h_params.my, h_params.nb_iter, h_params.minkowski_order, h_params.minkowski_order);
				p->w = 1.0f;

			}
	}

}//end namespace cpu

namespace gpu{
    void imp_Julia();

    void init(){
        checkCudaErrors( cudaMallocHost((void**) &gbl::pixels, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(float4)) );
	    checkCudaErrors( cudaMalloc((void**)&gbl::d_pixels, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(float4)) );
        gbl::display = imp_Julia;
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
    }

    __global__ void juliaColor(float4* pixels, int SCREENX, int SCREENY) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index < SCREENX * SCREENY) {
            //access constant memory once per thread !
            Param t_params = d_params;
            float scale = t_params.scale;
            Complex offset = t_params.offset;
            float sx = t_params.mx;
            float sy = t_params.my;
            int nb_iter = t_params.nb_iter;
            float order = t_params.minkowski_order;
            float thresh = t_params.threshold;
            float fac_in = t_params.easing_fac_in;
            float fac_out = t_params.easing_fac_out;
            MyCol col1 = MyCol(t_params.color_easeIn);
            MyCol col2 = MyCol(t_params.color_easeOut);
            MyCol col3 = MyCol(t_params.color_step);

            //deduce i, j (pixel coordinate) from threadIdx, blockIdx ...
            int i = index / SCREENX;
		    int j = index - i * SCREENX;

		    //deduces x,y (position in complex plane) from i,j...
		    float x = (float)(scale * (j - SCREENX / 2)) + offset.a;
		    float y = (float)(scale * (i - SCREENY / 2)) + offset.b;

			float4* pixel = pixels + (i * SCREENX + j);
			Complex a = Complex(x, y);
			Complex seed = Complex(sx, sy);
			float escFac, outNormFac; //color fac if point escape
            float inNormFac=0.0f;   //color fac if point doesn't escape
            float norm; int k=0;
			for (; k < nb_iter; k++) {
				a = a * a + seed;
                norm = minkowski(a, order);
                
                if (norm > thresh) {                    
                    escFac = 1.0f - (float)k / nb_iter;                      // red based on escape time
                    outNormFac = min((norm - thresh) * fac_out, 1.0f);   // green based on norm when escaping
                    break;  // stop iterating after escape
                }
			}
            if(k==nb_iter) inNormFac = min(norm* fac_in, 1.0f);
            //if you want to hanlde alpha blend, multiply each fac*col by col.w
            pixel->x = min(inNormFac * col1.x + outNormFac * col2.x + escFac * col3.x,1.0f);
            pixel->y = min(inNormFac * col1.y + outNormFac * col2.y + escFac * col3.y,1.0f); 
            pixel->z = min(inNormFac * col1.z + outNormFac * col2.z + escFac * col3.z,1.0f); 
			pixel->w = 1.0;
		}
	}

    __global__ void mandelbrotColor(float4* pixels, int SCREENX, int SCREENY) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index < SCREENX * SCREENY) {
            //access constant memory once per thread !
            Param t_params = d_params;
            float scale = t_params.scale;
            Complex offset = t_params.offset;
            int nb_iter = t_params.nb_iter;
            float order = t_params.minkowski_order;
            float thresh = t_params.threshold;
            float fac_in = t_params.easing_fac_in;
            float fac_out = t_params.easing_fac_out;
            MyCol col1 = MyCol(t_params.color_easeIn);
            MyCol col2 = MyCol(t_params.color_easeOut);
            MyCol col3 = MyCol(t_params.color_step);

            //deduce i, j (pixel coordinate) from threadIdx, blockIdx ...
            int i = index / SCREENX;
		    int j = index - i * SCREENX;

		    //deduces x,y (position in complex plane) from i,j...
		    float x = (float)(scale * (j - SCREENX / 2)) + offset.a;
		    float y = (float)(scale * (i - SCREENY / 2)) + offset.b;

			float4* pixel = pixels + (i * SCREENX + j);
			Complex a = Complex(0, 0);
			Complex seed = Complex(x, y);
			float escFac, outNormFac; //color fac if point escape
            float inNormFac=0.0f;   //color fac if point doesn't escape
            float norm; int k=0;
			for (; k < nb_iter; k++) {
				a = a * a + seed;
                norm = minkowski(a, order);
                
                if (norm > thresh) {                    
                    escFac = 1.0f - (float)k / nb_iter;                      // red based on escape time
                    outNormFac = min((norm - thresh) * fac_out, 1.0f);   // green based on norm when escaping
                    break;  // stop iterating after escape
                }
			}
            if(k==nb_iter) inNormFac = min(norm* fac_in, 1.0f);
            //if you want to hanlde alpha blend, multiply each fac*col by col.w
            pixel->x = min(inNormFac * col1.x + outNormFac * col2.x + escFac * col3.x,1.0f);
            pixel->y = min(inNormFac * col1.y + outNormFac * col2.y + escFac * col3.y,1.0f); 
            pixel->z = min(inNormFac * col1.z + outNormFac * col2.z + escFac * col3.z,1.0f); 
			pixel->w = 1.0;
		}
	}

    __global__ void juliaBshipColor(float4* pixels, int SCREENX, int SCREENY) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index < SCREENX * SCREENY) {
            //access constant memory once per thread !
            Param t_params = d_params;
            float scale = t_params.scale;
            Complex offset = t_params.offset;
            float sx = t_params.mx;
            float sy = t_params.my;
            int nb_iter = t_params.nb_iter;
            float order = t_params.minkowski_order;
            float thresh = t_params.threshold;
            float fac_in = t_params.easing_fac_in;
            float fac_out = t_params.easing_fac_out;
            MyCol col1 = MyCol(t_params.color_easeIn);
            MyCol col2 = MyCol(t_params.color_easeOut);
            MyCol col3 = MyCol(t_params.color_step);

            //deduce i, j (pixel coordinate) from threadIdx, blockIdx ...
            int i = index / SCREENX;
		    int j = index - i * SCREENX;

		    //deduces x,y (position in complex plane) from i,j...
		    float x = (float)(scale * (j - SCREENX / 2)) + offset.a;
		    float y = (float)(scale * (i - SCREENY / 2)) + offset.b;

			float4* pixel = pixels + (i * SCREENX + j);
			Complex a = Complex(x, y);
			Complex seed = Complex(sx, sy);
			float escFac, outNormFac; //color fac if point escape
            float inNormFac=0.0f;   //color fac if point doesn't escape
            float norm; int k=0;
			for (; k < nb_iter; k++) {
				a = a * a + seed;
                a.a = abs(a.a); a.b= abs(a.b);
                norm = minkowski(a, order);
                
                if (norm > thresh) {                    
                    escFac = 1.0f - (float)k / nb_iter;                      // red based on escape time
                    outNormFac = min((norm - thresh) * fac_out, 1.0f);   // green based on norm when escaping
                    break;  // stop iterating after escape
                }
			}
            if(k==nb_iter) inNormFac = min(norm* fac_in, 1.0f);
            //if you want to hanlde alpha blend, multiply each fac*col by col.w
            pixel->x = min(inNormFac * col1.x + outNormFac * col2.x + escFac * col3.x,1.0f);
            pixel->y = min(inNormFac * col1.y + outNormFac * col2.y + escFac * col3.y,1.0f); 
            pixel->z = min(inNormFac * col1.z + outNormFac * col2.z + escFac * col3.z,1.0f); 
			pixel->w = 1.0;
		}
	}

    __global__ void burningshipColor(float4* pixels, int SCREENX, int SCREENY) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index < SCREENX * SCREENY) {
            //access constant memory once per thread !
            Param t_params = d_params;
            float scale = t_params.scale;
            Complex offset = t_params.offset;
            int nb_iter = t_params.nb_iter;
            float order = t_params.minkowski_order;
            float thresh = t_params.threshold;
            float fac_in = t_params.easing_fac_in;
            float fac_out = t_params.easing_fac_out;
            MyCol col1 = MyCol(t_params.color_easeIn);
            MyCol col2 = MyCol(t_params.color_easeOut);
            MyCol col3 = MyCol(t_params.color_step);

            //deduce i, j (pixel coordinate) from threadIdx, blockIdx ...
            int i = index / SCREENX;
		    int j = index - i * SCREENX;

		    //deduces x,y (position in complex plane) from i,j...
		    float x = (float)(scale * (j - SCREENX / 2)) + offset.a;
		    float y = (float)(scale * (i - SCREENY / 2)) + offset.b;

			float4* pixel = pixels + (i * SCREENX + j);
			Complex a = Complex(0, 0);
			Complex seed = Complex(x, y);
			float escFac, outNormFac; //color fac if point escape
            float inNormFac=0.0f;   //color fac if point doesn't escape
            float norm; int k=0;
			for (; k < nb_iter; k++) {
				a = a * a + seed;
                a.a = abs(a.a); a.b= abs(a.b);
                norm = minkowski(a, order);
                
                if (norm > thresh) {                    
                    escFac = 1.0f - (float)k / nb_iter;                      // red based on escape time
                    outNormFac = min((norm - thresh) * fac_out, 1.0f);   // green based on norm when escaping
                    break;  // stop iterating after escape
                }
			}
            if(k==nb_iter) inNormFac = min(norm* fac_in, 1.0f);
            //if you want to hanlde alpha blend, multiply each fac*col by col.w
            pixel->x = min(inNormFac * col1.x + outNormFac * col2.x + escFac * col3.x,1.0f);
            pixel->y = min(inNormFac * col1.y + outNormFac * col2.y + escFac * col3.y,1.0f); 
            pixel->z = min(inNormFac * col1.z + outNormFac * col2.z + escFac * col3.z,1.0f); 
			pixel->w = 1.0;
		}
	}

	void imp_Julia(){
        if(gbl::otherWindow){
            wdw::julMandParam();
            wdw::julMandPreset();
        }

		int N = gbl::SCREEN_X * gbl::SCREEN_Y;
		int M = 256;
		
		//... nothings to send to gpu
		// juliaColor << <(N + M - 1) / M, M >> > (gbl::d_pixels, h_params.mx, h_params.my, 
        //         h_params.nb_iter, h_params.minkowski_order, h_params.threshold,
        //         gbl::SCREEN_X, gbl::SCREEN_Y, h_params.scale, h_params.offset); 
		juliaColor << <(N + M - 1) / M, M >> > (gbl::d_pixels, gbl::SCREEN_X, gbl::SCREEN_Y); 
		checkKernelErrors();
		checkCudaErrors( cudaMemcpy(gbl::pixels, gbl::d_pixels, N * sizeof(float4), cudaMemcpyDeviceToHost) ); //get pixels values from gpu
	}

    void imp_Mandelbrot(){
        if(gbl::otherWindow){//todo replace with mandelbrot specific
            //wdw::mandelbrotParam();
            //wdw::mandelbrotPreset();
            wdw::julMandParam();
            wdw::julMandPreset();
        }

		int N = gbl::SCREEN_X * gbl::SCREEN_Y;
		int M = 256;
		
		//... nothings to send to gpu
		mandelbrotColor << <(N + M - 1) / M, M >> > (gbl::d_pixels, gbl::SCREEN_X, gbl::SCREEN_Y); 
		checkKernelErrors();
		checkCudaErrors( cudaMemcpy(gbl::pixels, gbl::d_pixels, N * sizeof(float4), cudaMemcpyDeviceToHost) ); //get pixels values from gpu
	}

    void imp_JuliaBship(){
        if(gbl::otherWindow){
            wdw::julMandParam();
            wdw::julMandPreset();
        }

		int N = gbl::SCREEN_X * gbl::SCREEN_Y;
		int M = 256;
		
		//... nothings to send to gpu
		juliaBshipColor << <(N + M - 1) / M, M >> > (gbl::d_pixels, gbl::SCREEN_X, gbl::SCREEN_Y); 
		checkKernelErrors();
		checkCudaErrors( cudaMemcpy(gbl::pixels, gbl::d_pixels, N * sizeof(float4), cudaMemcpyDeviceToHost) ); //get pixels values from gpu
	}

    void imp_Burningship(){
        if(gbl::otherWindow){//todo replace with mandelbrot specific
            //wdw::mandelbrotParam();
            //wdw::mandelbrotPreset();
            wdw::julMandParam();
            wdw::julMandPreset();
        }

		int N = gbl::SCREEN_X * gbl::SCREEN_Y;
		int M = 256;
		
		//... nothings to send to gpu
		burningshipColor << <(N + M - 1) / M, M >> > (gbl::d_pixels, gbl::SCREEN_X, gbl::SCREEN_Y); 
		checkKernelErrors();
		checkCudaErrors( cudaMemcpy(gbl::pixels, gbl::d_pixels, N * sizeof(float4), cudaMemcpyDeviceToHost) ); //get pixels values from gpu
	}


    
}//end namespace gpu

namespace wdw{
    void julMandParam(){
        ImGui::Begin("Mandelbrot & Julia Param");
        if(ImGui::Button("center")) preset::center();
        ImGui::SameLine(); if(ImGui::Button("gpu default")) preset::gpu_default();

        float inputWidth = ImGui::CalcTextSize("-0.000").x + ImGui::GetStyle().FramePadding.x * 2;

        ImGui::Text("Julia set for z²+c where c =");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(inputWidth); // Set the width for real part
        ImGui::InputFloat("##real", &h_params.mx);
        ImGui::SameLine(); ImGui::Text("+");
        ImGui::SameLine(); ImGui::SetNextItemWidth(inputWidth); // Set the width for imaginary part
        ImGui::InputFloat("##imaginary", &h_params.my);
        ImGui::SameLine(); ImGui::Text("i");

        ImGui::Text("Center in complex plane : %.2f+%.2fi",h_params.offset.a, h_params.offset.b);
        ImGui::Text("Width : %.2f, height : %.2f", h_params.scale * gbl::SCREEN_X, h_params.scale * gbl::SCREEN_Y);


        ImGui::InputInt("nb step", &h_params.nb_iter);
        ImGui::InputFloat("threshold", &h_params.threshold, 0.01f, 1.0f, "%.1f");
        ImGui::InputFloat("minkowski order", &h_params.minkowski_order, 0.01f, 1.0f, "%.4f");


        if (ImGui::TreeNode("Fractal-type"))
        {
            static int selected = 0;
            if (ImGui::Selectable("Julia", selected == 0))      {selected = 0; gbl::display = gpu::imp_Julia;}
            if (ImGui::Selectable("Mandelbrot", selected == 1)) {selected = 1; gbl::display = gpu::imp_Mandelbrot;}
            if (ImGui::Selectable("Bship julia", selected == 2)){selected = 2; gbl::display = gpu::imp_JuliaBship;}
            if (ImGui::Selectable("Burning ship", selected == 3)){selected = 3; gbl::display = gpu::imp_Burningship;}
            
            ImGui::TreePop();
        }

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
        }

        ImGui::End();
    }
    void julMandPreset(){
        ImGui::Begin("Julia Presets");

        if (ImGui::TreeNode("Julia"))
        {
            if(ImGui::Button("spiral1")) preset::spiral1();
            if(ImGui::Button("spiral2")) preset::spiral2();
            if(ImGui::Button("douady")) preset::douady();
            if(ImGui::Button("branches")) preset::branches();

            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Bship Julia")){
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Colors scheme")){
            ImGui::TreePop();
        }
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

namespace gbl{
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
}


namespace cbk{ 
    /*various callback
    You must ALWAYS forward the event to ImGui before processing it (except window resizing)
    You can find relevant ImGui callback in ./imgui/imgui_impl_glfw.cpp line 536 in function ImGui_ImplGlfw_InstallCallbacks
    */

    inline void updt_mpos(double xpos, double ypos){
        h_params.mx = h_params.offset.a + (float)(h_params.scale*(xpos - gbl::SCREEN_X / 2));
        h_params.my = h_params.offset.b - (float)(h_params.scale*(ypos - gbl::SCREEN_Y / 2));
    }

    void mouse_button(GLFWwindow* window, int button, int action, int mods){
        // Forward the event to ImGui
        ImGuiIO& io = ImGui::GetIO();
        ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
        
        //if ImGui doesn't want the event, process it
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

    static void cursor_position(GLFWwindow* window, double xpos, double ypos){
        //forward the event to ImGui
        ImGuiIO& io = ImGui::GetIO();
        ImGui_ImplGlfw_CursorPosCallback(window, xpos, ypos);

        //if ImGui doesn't want the event, process i
        if(!io.WantCaptureMouse){
            int leftState = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
            if(leftState == GLFW_PRESS){
                updt_mpos(xpos, ypos);
            }
        }
    }

    void scroll(GLFWwindow* window, double xoffset, double yoffset){
        // Forward the event to ImGui
        ImGuiIO& io = ImGui::GetIO();
        ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
        
        //if ImGui doesn't want the event, process it
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
    {/* set up parameter*/
        h_params.scale = 0.003f;
        h_params.mx = 0.0f;
        h_params.my = 0.0f;
        
        h_params.nb_iter = 7;
        h_params.minkowski_order = 2.0f;
        h_params.threshold = 4.0f;
        h_params.offset = Complex(0.0f, 0.0f);
        
        //color control
        h_params.easing_fac_in = 1.0f;
        h_params.easing_fac_out = 0.2f;
         h_params.color_step = MyCol(1.0f, 0.5f, 0.2f, 1.0f);
        h_params.color_easeIn = MyCol(0.8f, 0.3f, 0.4f, 1.0f);
        h_params.color_easeOut = MyCol(0.4f, 0.7f, 0.9f, 1.0f);
    }

    /* Initialize callback*/
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