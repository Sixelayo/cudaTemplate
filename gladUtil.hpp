//glad includes
#include <glad/gl.h>
#include <glad/gl.c>

//glfw includes
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <iostream>

bool use_interop = false;

namespace interopPBO{
    //global var
    GLuint glBuffer;
    GLuint glTex;
    struct cudaGraphicsResource* cuBuffer;

    void step1(GLFWwindow* window){
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        
        // create a buffer
        glGenBuffers(1, &glBuffer);
        // make it the active buffer
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, glBuffer);
        // allocate memory, but dont copy data (NULL)
        glBufferData(GL_PIXEL_UNPACK_BUFFER,
        width*height*sizeof(float4), NULL, GL_STREAM_DRAW);

        
        glEnable(GL_TEXTURE_2D); // Enable texturing
        glGenTextures(1,&glTex); // Generate a texture ID
        glBindTexture(GL_TEXTURE_2D, glTex); // Set as the current texture
        // Allocate the texture memory.
        // The last parameter is NULL:
        // we only want to allocate memory, not initialize it
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height,
        0, GL_RGBA, GL_FLOAT, NULL);
        // Must set the filter mode:
        // GL_LINEAR enables interpolation when scaling
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    }

    void step2(){
        cudaGLSetGLDevice(0); // explicitly set device 0
        cudaGraphicsGLRegisterBuffer(&cuBuffer,glBuffer,cudaGraphicsMapFlagsWriteDiscard);
        // cudaGraphicsMapFlagsWriteDiscard:
        // CUDA will only write and will not read from this resource

    }

    void step34(float4* cuPixels){
        size_t num_bytes;
        cudaGraphicsMapResources(1, &cuBuffer, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&cuPixels, &num_bytes, cuBuffer);

    }

    void step6(){
        cudaGraphicsUnmapResources(1,&cuBuffer);
    }

    void step7(GLFWwindow* window){
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        // Select the appropriate buffer
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, glBuffer);
        // Select the appropriate texture
        glBindTexture(GL_TEXTURE_2D, glTex);
        // Make a texture from the buffer
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
        width, height, GL_RGBA, GL_FLOAT, NULL); 

        glBegin(GL_QUADS);
            glTexCoord2f(0, 1.0f);
            glVertex3f(0,0,0);
            glTexCoord2f(0,0);
            glVertex3f(0,height,0);
            glTexCoord2f(1.0f,0);
            glVertex3f(width,height,0);
            glTexCoord2f(1.0f,1.0f);
            glVertex3f(width,0,0);
        glEnd();
    }
    void step8(){
        cudaGraphicsUnregisterResource(cuBuffer);
        glDeleteTextures(1, &glTex);
        glDeleteBuffers(1, &glBuffer);
    }



    
};
