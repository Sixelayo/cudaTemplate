# How to compile

- see : [this repo](https://github.com/Sixelayo/CY_Visual_GPU)
- additionally, for  tp4 and tp5, you'll need to link with glm headers. I've tried one installed with the MSYS2 commande lines but there seemed to be compatibility issues with MSVC (even tho it's supposed to be only header files ?), so I got glm with vcpkg `vcpkg install glm:x64-windows`

# generic remarks

- The whole architecture of passing a Param Struct to gpu could be reworked as it was made early wihtout optimisation knowledge. (I have a small `struct Param` in cpu that I pass into constant memory each frame which is good but I've done some weird preloading parameters as locals variables in some kernels)
- unlike previous work, I felt much more confident about overall architecture (that it to say, it wouldn't be impossible to come back to this code whereas I would be lost if I tried to get back to PhysicsEngine code), even if it's far from perfect it's way cleaner and easier to work with.


# List of .cu files

> Press w at any point to show / hide additional windows
> hold `ctrl`while scrolling to zoom faster

## julia

### functionalities & remarks

- everything asked

### preset

I did not have time to properly bind presets to all parameteres so it can be a little unintuitive. Here's a video where 


## ray marching

- everything asked
- a fixed number of spheres are always loaded into memory
- you can limit the number of sphere displayed or regenerate random spheres
- you add ambient light color and intensity
- you can move camera holding left click

- sadly we can't used constant memory as const ptr (well it's normal because its nonsense in 99.99% of use cases but the only situation where it usefull is to re use code that access data either in constant or global memory). This forces ugly code duplication.
- I couldn't notice any performance difference between constant and global memory  on my gtx980. This was to be expected.
- constant memory limited to 64kb. I've limited sphere count to 500 when using constant memory
- technically for performance I should've always duplicate code and use callbacks but I haven't done that for readability and cleaner code (esp for streams / constant memory), I preferred boolean switch
- We can notice a small improvement when using streams. (35 -> 40 fps with 500 spheres) This is very likely because the kernel is computationally light and thus paralellization is almost irrelevant"


## bugs 

### functionalities & remarks

- everything asked except shared memory TODO
- Big confusion : between standard notation : is a cell its own neighbor ? (online ressources uses different convention)
- you can freely resize window in CPU or GPU mode (but grid state is'nt saved)
- cells are stored as float4 (and not boolean) to allow easier implementation of color variation.
- for more control and visibility (especialy for conway's version in GPU) it's possible to limit framerate in order to limit the number of iteration per second. (note that enforcing slower framerate will naturaly render irrelevant the computed FPS as bottleneck won't be computation speed anymore)
- due to limitation to the architecture this program isn't compliant to specific rules (ex : ANNEAL, a cell birth/survive if neighbor count is in {4,7,8,9} which isn't connexe, https://www.moreno.marzolla.name/teaching/HPC/handouts/cuda-anneal.html). Another example would be https://conwaylife.com/wiki/OCA:2%C3%972

### Exploring presets

- an overview of all present can be seen on my YT channel : TODO
- Simply click on the presets button. 
- Remember to keep "Pre choosen config" checked to immediatly have visually interresting behaviors
- things to do when exploring preset :
    - tweak framerate to have faster / slower simulation
    - use the "random configuration" and "random rectangle" buttons alongside with the %spawn parameter to reset the grid initial configuration
- 12 presets are currently here, all with differents behaviors of various interest. Most use a neighboring distance of 1 (aside from bugs and blob). Currently includes presets are :
    - Conway's game of life (1,3,4,3,3)  // B34/S3
    - Bugs (5,34,58,34,45)
    - Life without death (1,1,9,3,3) // B3/S012345678
    - Maze (1,3,3,1,5) // B3/S12345
    - Mazectric (1,3,3,1,4) // B3/S1234
    - ... other that aren't all named

### Sources 

- https://catagolue.hatsya.com/home 
- https://conwaylife.com/wiki/Main_Page


### TODO

Mandatory
- utiliser la shared memory pour lire tuile par tuile

Optional things I would've had given infinite time :
- add support for any Rulestring, not only those where Survive / Birth are connex intervals 2x2 /Move / Day and Night / Anneal / Diamoeba ...
- keeping grid configuration when switching from CPU to GPU and GPU to CPU
- better color management
- dynamically grid editing (draw directly onto grid with mouse, add saved figured like ships)
- introduce a zoom
- pause button


## td4 nbody

- everything asked is present (TODO CHECK)
- I also adapted camera.c for glfw and plugged callback into my template file but things could be more way more clean
- Only primitive camera control to move eye arround the center were added
- I rewrite a few things and now use a struct Body{} instead of raw global array.
- you can add colors to particles (based on position, or a mapranged of normalized speed)

Faster version with shared memory (I was confused at first but here's the strategy I adopted, I'm unsure if that's what was asked)
Assuming N body, declare N thread with a block size 256
- kernel v1 : loop over all other positions (ie N\*N access to constant memory)
- kernel v2 : for each block, load the 256 first positions. Add partial acceleration for each body. Then load the next 256 bodies and repeat

We notice a MASSIVE performance increase when using shared memory and loading batch of pos / mass data. when pushing simulation to its limit (20k bodies)

## td5 kmeans

- everything asked is present
- I used a `struct Point` to store points and cluster. an int serve as both label (for points) and count (for cluster during phase 2)
- there is technically useless data copying from gpu as I also fetch points position which don't change
- we can observe configuration where kmeans wrongly converges when we really crank up the numbers of point (IE : one cluster (during the algorithm) will match 2 cluster (from random initialisaiton))
- swithcing from gpu version 1 to version 2 is little buggy because of unproper data transfer, when switching gpu implementation regenerate random data.
- While I do indeed have reduceKernel that makes things faster (we can see it for sure with 500k points / 500 cluster that GPU version 2 is definitly faster that version 1 that does phase 2 on cpu), I'm quite confident this wasn't the proper way to do it. I've used one kernel per cluster, and used the same trick as in the previous TD kmeans (preloading a batch of points). I'm pretty sure the correct way to do things was to use one kernel per points, and doing a reduction for summing closter partial sum / count (because most of the time there will be way more points than cluster so it's smater to have kernel / per points and do the well known standart reduction)


# Other remarkss

- left to do : VSCode task for automatic copile without debug option
- the whole "param" architecture is a little messy as I've done it before the optimization class. I think the compiler probably gets rid of my messy buffering and loading the param struct but still I should've only used d_param and not t_param (per-thread copie of d_param which is constant memory ... when there are many access to d_param I should've use individual local variable and not the whole param structure)
- understand the warning comming from glm which I've shut done with `"-diag-suppress=20012"`
- `compute-sanitizer` command usefull for debugging cuda. It probably saved me hours of debugging by properly hightlighting the exact line in my code where there was a memory access whereas default crash only give you the line of the macro( useless)
- I may be missing some `cudaDeviceSynchronice` that cause crash because I didn't check every click combination with UI. Also I did'nt enforce heavy user input validation aside from slider min / max which you can override with ctrl+click so that may cause crash