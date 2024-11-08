# cuda template CY Visual
 
starter kit for GPU programming, if you want to use Visual Studio Code.
This template can replace the old one using freeglut.
- uses opengl / GLFW
- has the same functionnalities than the previous one
- support window resizing
- added ImGui - docking branch 

# Requierement

## Compiler

While this repo is made for VSCode, it is **highly recommended** (*and probably mandatory*) to install [VSCommunity](https://visualstudio.microsoft.com/fr/vs/community/)  with c/c++ desktop dev before installing cuda, because the nvcc compiler relies on MSVC toolchains

## CUDA toolkit 12.5

**FIXME** replace with exact string

Download the version matching your OS :
https://developer.nvidia.com/cuda-12-5-0-download-archive

At this point you can easily check that your cuda install is working by cloning the [cuda samples repo](https://github.com/NVIDIA/cuda-samples). 

then in VSCommunity > open solution > select cuda samples 22 
on any project : right clic > set as start up project
Project to check :
- param
- bandwidth
- fluid


## Recommended VS Code extension

- C/C++ : for intellisense
- Nsight Visual Studio Code extension : for advanced debugging. for more details see https://www.youtube.com/watch?v=gN3XeFwZ4ng


## GLFW 

This repo already contains `/lib/glfw3_mt.lib` which is a pre-compilled glfw runtime binaries for windows 10  / Visual C++ 2022.

if you're using a different OS, you need to replace it with the expected binaries :
- glfw binaries : https://www.glfw.org/download.html


# Compiling

## Instruction

If you're using VSCode You can simply run the provided task (`ctrl+shift+p > Run Task > build nvcc`)

It will simply run the following command :
```shell
nvcc -Xcompiler="/MT" foo.cpp ./imgui/*.cpp -o foo.exe  -I./include -L./lib -lglfw3_mt -lopengl32 -luser32 -lgdi32 -lshell32
```

## Static vs dynamic compilation for GLFW

The VScode task will compile statically. (as it's obviously recommanded).
If for some reason you want to compile dynamically, check the readme of glfw pre-compiled binairies
At the very least you'll need to add glfw3.dll to your .exe directory (or path) and linking with glfw3dll.lib. Note that you can get rid of `-luser32`, `-lgdi32`, `-lshell32` if you're compiling dynamically


# Using ImGui

You can learn more about ImGui here : [Original repo](https://github.com/ocornut/imgui)

The easiest way to learn about ImGui is by running the example file and fooling arround. Then just ctrl+f in `imgui_demo.cpp` to find out how things are done

If you MSYS2 - mingw64 then you can install GLFW in mingw64 shell with 
```shell
pacman -S mingw-w64-x86_64-glfw
```

And compile `imgui_example.cpp` with :
```shell
g++ imgui_example.cpp ./imgui/*.cpp -o ./exe/example.exe -lopengl32 -lglfw3
```

Otherwise, you need to to precise include path to g++.
If you're using windows, you'll also need to make glfw3.dll visible from you're OS. either add `glfw3.dll` in the same directory than your executable or add `glfw-3.4.bin.WIN64\lib-vc2022` to PATH environnement variable
```shell
g++ imgui_example.cpp ./imgui/*.cpp -o ./exe/example.exe -I./include/ -L./lib/ -lopengl32 -lglfw3
```
