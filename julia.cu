#include "util.hpp"
#define TITLE "template"

/* use task to compile or run command TODO*/

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
    //mouse coordinate
    float mx, my;

    int nb_iter = 7;
    float minkowski_order = 2;
    float threshhold = 4;

    Complex offset{0,0};

}//end namespace prm


namespace cpu{
    void init(){
        gbl::pixels = (float4*)malloc(gbl::SCREEN_X*gbl::SCREEN_Y*sizeof(float4));
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

    void imp_julia() {
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
    void init(){
        checkCudaErrors( cudaMallocHost((void**) &gbl::pixels, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(float4)) );
	    checkCudaErrors( cudaMalloc((void**)&gbl::d_pixels, gbl::SCREEN_X * gbl::SCREEN_Y * sizeof(float4)) );
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
		//deduce i, j from threadIdx, blockIdx ...
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		int i = index / SCREENX;
		int j = index - i * SCREENX;
		//deduces x,y from i,j...
		float x = (float)(scale * (j - SCREENX / 2)) + offset.a;
		float y = (float)(scale * (i - SCREENY / 2)) + offset.b;
		if (index < SCREENX * SCREENY/*x, y TODO correspondent à un block cohérent*/) {
			float4* pixel = pixels + (i * SCREENX + j);
			Complex a = Complex(x, y);
			Complex seed = Complex(sx, sy);
			pixel->x = 0.0;
			pixel->y = 0.0;
			pixel->z = 0.0;
			for (int i = 0; i < p+3; i++) {
				a = a * a + seed;
				if (minkowski(a, order) > thresh) {
					pixel->x = 1 - (float)i / p;
				}
			}
			for (int i = 0; i < p-3; i++) {
				a = a * a + seed;
				if (minkowski(a, order) > thresh) {
					pixel->y = 1 - (float)i / (p-3);
				}
			}
			for (int i = 0; i < p - 6; i++) {
				a = a * a + seed;
				if (minkowski(a, order) > thresh) {
					pixel->z = 1 - (float)i / (p-6);
				}
			}
			pixel->w = 1.0;
		}
	}

    //todo chane name !
	void imp_Julia(){
		int N = gbl::SCREEN_X * gbl::SCREEN_Y;
		int M = 256;
		
		//... nothings to send to gpu
		juliaColor << <(N + M - 1) / M, M >> > (gbl::d_pixels, prm::mx, prm::my, 
                prm::nb_iter, prm::minkowski_order, prm::threshhold,
                gbl::SCREEN_X, gbl::SCREEN_Y, prm::scale, prm::offset); 
		checkKernelErrors();
		checkCudaErrors( cudaMemcpy(gbl::pixels, gbl::d_pixels, N * sizeof(float4), cudaMemcpyDeviceToHost) ); //get pixels values from gpu
	}
    
}//end namespace gpu

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
        prm::mx = (float)(prm::scale*(xpos - gbl::SCREEN_X / 2));
        prm::my = -(float)(prm::scale*(ypos - gbl::SCREEN_Y / 2));
    }

    void mouse_button(GLFWwindow* window, int button, int action, int mods){
        // Forward the event to ImGui
        ImGuiIO& io = ImGui::GetIO();
        ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
        
        //if ImGui doesn't want the event, process it
        if(!io.WantCaptureMouse){
            if(button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS){
                double xpos, ypos;
                glfwGetCursorPos(window, &xpos, &ypos);
                updt_mpos(xpos, ypos);
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

    /* init render specific values*/
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
        utl::wdw_info(gbl::mode, gbl::SCREEN_X,gbl::SCREEN_Y,gbl::currentFPS);
        
        /* Render here */
        gbl::calculate(window);
        //use a callback wo param instead ! torm
        gpu::imp_Julia(); 
        if(!gbl::needResize) glDrawPixels(gbl::SCREEN_X, gbl::SCREEN_Y, GL_RGBA, GL_FLOAT, gbl::pixels);
        
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