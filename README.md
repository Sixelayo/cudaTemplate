# generic remarks

- The whole architecture of passing a Param Struct to gpu could be reworked as it was made early wihtout optimisation knowledge


# List of .cu files

> Press w at any point to show / hide additional windows

## julia

- utiliser la shared memory pour les params de julia !
-

### preset

- handle more smart preset switching from one mode to another so there's no side effect when playing arround, clicking abutton always reset ALL param


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
- I noticed a 30% performance drop when using stream. This is very likely because the kernel is computationally light and thus paralellization is irrelevant"

# todo

- uses stream

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
- proprement implémentaiton gpu ne plus utiliser gridSwap
- utiliser la shared memory pour lire tuile par tuile

Optional things I would've had given infinite time :
- add support for any Rulestring, not only those where Survive / Birth are connex intervals 2x2 /Move / Day and Night / Anneal / Diamoeba ...
- keeping grid configuration when switching from CPU to GPU and GPU to CPU
- better color management
- dynamically grid editing (draw directly onto grid with mouse, add saved figured like ships)
- introduce a zoom
- pause button




- seed growth