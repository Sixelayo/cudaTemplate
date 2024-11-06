# cudaTemplate
 
starter kit for GPU programming, if you want to use Visual Studio Code

# Requierement

## CUDA toolkit 12.5

Download the version matching your OS :
https://developer.nvidia.com/cuda-12-5-0-download-archive

## Recommended VS Code extension

- C/C++ : for intellisense
- Nsight Visual Studio Code extension : for advanced debugging. for more details see https://www.youtube.com/watch?v=gN3XeFwZ4ng

## Compiler

While we're using Visual Studio Code, VSCommunity22 is still probably needed because nvcc likely relies on some of it toolchains (**TO BE DETERMINED**)
https://visualstudio.microsoft.com/fr/vs/community/


## GLFW 

This repo already contains `/lib/glfw3_mt.lib` which is a pre-compilled glfw runtime binaries for windows 10  / Visual C++ 2022.

if you're using a different OS, you need to replace it with the expected binaries :
- glfw binaries : https://www.glfw.org/download.html


# Compiling


If you're using VSCode You can simply run the provided task (`ctrl+shift+p > Run Task > build nvcc`)

It will simply run the following command :
nvcc -Xcompiler="/MT" foo.cpp ./imgui/*.cpp -o foo.exe  -I./include -L./lib -lglfw3_mt -lopengl32 -luser32 -lgdi32 -lshell32

## Static vs dynamic compilation for GLFW

The VScode task will compile statically. (as it's obviously recommanded).
If for some reason you want to compile dynamically, check the readme of glfw pre-compiled binairies
At the very least you'll need to add glfw3.dll to your .exe directory (or path) and linking with glfw3dll.lib. Note that you can get rid of `-luser32`, `-lgdi32`, `-lshell32` if you're compiling dynamically






