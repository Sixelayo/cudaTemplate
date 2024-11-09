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
		__device__ __host__ Complex(float a, float b) : a(a), b(b){}

		__device__ __host__ Complex operator+(const Complex& other) const {
			return Complex(a + other.a, b + other.b);
		}
		__device__ __host__ Complex operator*(const Complex& other) const {
			return Complex(a * other.a - b * other.b, a * other.b + b * other.a);
		}

	};

	__device__ __host__ float minkowski(Complex c, float order) {
		return pow((pow(c.a, order) + pow(c.b, order)), 1.0f / order);
	}


//various parameters
namespace prm{
    float scale = 0.003f;
    float mx, my; //mouse coordinate

    int nb_iter = 7;
    float minkowski_order = 2;
    float threshhold = 4;

    Complex offset{0,0};

}//end namespace prm

namespace preset{
    void center(){
        prm::scale = 0.0035f;
        prm::offset =  {0,0};
    }
    void gpu_default(){
        prm::nb_iter = 40;
        prm::minkowski_order = 2;
        prm::threshhold = 4;
    }
    void spiral1(){
        prm::mx = -0.5251993f;
        prm::my = -0.5251993f;
    }
    void spiral2(){
        prm::mx = -0.77146f;
        prm::my = -0.10119f;
    }
    void douady(){
        prm::mx = -0.12f;
        prm::my = 0.75f;
    }
    void branches(){
        prm::mx = 0.35; 
        prm::my = 0.35f;
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
				float x = (float)(prm::scale * (j - gbl::SCREEN_X / 2));
				float y = (float)(prm::scale * (i - gbl::SCREEN_Y / 2));
				float4* p = gbl::pixels + (i * gbl::SCREEN_X + j);
				// default: black
				p->x = 0.0f;
				p->y = 0.0f;
				p->z = cpu::juliaColor(x - prm::offset.a, y - prm::offset.b, prm::mx  , prm::my, prm::nb_iter, prm::minkowski_order, prm::minkowski_order);
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

    __global__ void juliaColor(float4* pixels, float sx, float sy, int p, float order, float thresh, int SCREENX, int SCREENY, float scale, Complex offset) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index < SCREENX * SCREENY) {
            //deduce i, j (pixel coordinate) from threadIdx, blockIdx ...
            int i = index / SCREENX;
		    int j = index - i * SCREENX;

		    //deduces x,y (position in complex plane) from i,j...
		    float x = (float)(scale * (j - SCREENX / 2)) + offset.a;
		    float y = (float)(scale * (i - SCREENY / 2)) + offset.b;

			float4* pixel = pixels + (i * SCREENX + j);
			Complex a = Complex(x, y);
			Complex seed = Complex(sx, sy);
			pixel->x = 0.0;
			pixel->y = 0.0;
			pixel->z = 0.0;
            bool br = false; bool bg = false;
            float norm;
			for (int i = 0; i < p; i++) {
				a = a * a + seed;
                norm = minkowski(a, order);
                if(!br){
                    if (norm > thresh) {
                        pixel->x=min(norm*0.6f,1.0f);
                        br = true;
                    }
                }
                if(!bg){
                    if (norm > thresh) {
                        pixel->y = 1 - (float)i / p;
                        bg = true;
                    }
                }
			}
            pixel->z=min(norm,1.0f); //coloring interior
			pixel->w = 1.0;
		}
	}

    __global__ void mandelbrotColor(float4* pixels, int p, float order, float thresh, int SCREENX, int SCREENY, float scale, Complex offset) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index < SCREENX * SCREENY) {
            //deduce i, j (pixel coordinate) from threadIdx, blockIdx ...
            int i = index / SCREENX;
		    int j = index - i * SCREENX;

		    //deduces x,y (position in complex plane) from i,j...
		    float x = (float)(scale * (j - SCREENX / 2)) + offset.a;
		    float y = (float)(scale * (i - SCREENY / 2)) + offset.b;

			float4* pixel = pixels + (i * SCREENX + j);
			Complex a = Complex(0, 0);
			Complex seed = Complex(x, y);
			pixel->x = 0.0;
			pixel->y = 0.0;
			pixel->z = 0.0;
            bool br = false; bool bg = false;
            float norm;
			for (int i = 0; i < p; i++) {
				a = a * a + seed;
                norm = minkowski(a, order);
                if(!br){
                    if (norm > thresh) {
                        pixel->x=min(norm*0.6f,1.0f);
                        br = true;
                    }
                }
                if(!bg){
                    if (norm > thresh) {
                        pixel->y = 1 - (float)i / p;
                        bg = true;
                    }
                }
			}
            pixel->z=min(norm,1.0f); //coloring interior
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
		juliaColor << <(N + M - 1) / M, M >> > (gbl::d_pixels, prm::mx, prm::my, 
                prm::nb_iter, prm::minkowski_order, prm::threshhold,
                gbl::SCREEN_X, gbl::SCREEN_Y, prm::scale, prm::offset); 
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
		mandelbrotColor << <(N + M - 1) / M, M >> > (gbl::d_pixels,
                prm::nb_iter, prm::minkowski_order, prm::threshhold,
                gbl::SCREEN_X, gbl::SCREEN_Y, prm::scale, prm::offset); 
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
        ImGui::InputFloat("##real", &prm::mx);
        ImGui::SameLine(); ImGui::Text("+");
        ImGui::SameLine(); ImGui::SetNextItemWidth(inputWidth); // Set the width for imaginary part
        ImGui::InputFloat("##imaginary", &prm::my);
        ImGui::SameLine(); ImGui::Text("i");

        ImGui::Text("Center in complex plane : %.2f+%.2fi",prm::offset.a, prm::offset.b);
        ImGui::Text("Width : %.2f, height : %.2f", prm::scale * gbl::SCREEN_X, prm::scale * gbl::SCREEN_Y);


        ImGui::InputInt("nb step", &prm::nb_iter);
        ImGui::InputFloat("threshold", &prm::threshhold, 0.01f, 1.0f, "%.1f");
        ImGui::InputFloat("minkowski order", &prm::minkowski_order, 0.01f, 1.0f, "%.4f");


        if (ImGui::TreeNode("Single-Select"))
        {
            static int selected = 0;
            if (ImGui::Selectable("Julia", selected == 0))      {selected = 0; gbl::display = gpu::imp_Julia;}
            if (ImGui::Selectable("Mandelbrot", selected == 1)) {selected = 1; gbl::display = gpu::imp_Mandelbrot;}
            if (ImGui::Selectable("Burningship", selected == 2)){selected = 2; gbl::display = gpu::imp_Julia;}
            if (ImGui::Selectable("Bship julia", selected == 3)){selected = 3; gbl::display = gpu::imp_Julia;}
            
            ImGui::TreePop();
        }

        ImGui::End();
    }
    void julMandPreset(){
        ImGui::Begin("Julia Presets");
        if(ImGui::Button("spiral1")) preset::spiral1();
        if(ImGui::Button("spiral2")) preset::spiral2();
        if(ImGui::Button("douady")) preset::douady();
        if(ImGui::Button("branches")) preset::branches();
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
        prm::mx = prm::offset.a + (float)(prm::scale*(xpos - gbl::SCREEN_X / 2));
        prm::my = prm::offset.b - (float)(prm::scale*(ypos - gbl::SCREEN_Y / 2));
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
                prm::offset.a += (float)(prm::scale * (xpos - gbl::SCREEN_X / 2));
		        prm::offset.b += -(float)(prm::scale * (ypos - gbl::SCREEN_Y / 2));
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
            if (yoffset >0) prm::scale /= 1.05f;
	        else prm::scale *= 1.05f;
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