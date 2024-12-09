# current

## julia

- utiliser la shared memory pour les params de julia !
-

### preset

- handle more smart preset switching from one mode to another so there's no side effect when playing arround, clicking abutton always reset ALL param


## ray marching


## bugs 

REPORT
- cells are stored as float4 (and not boolean) to allow variation in colors. Also this file doesn't use gbl::pixels but bugs::h_grid
- for more control and visibility (especialy for conway's version in GPU) it's possible to clamp framerate in order to limit the number of iteration per second. (note that doing so will naturaly render the computed FPS will be irrelevant if the bottleneck is the forced framerate enforce slower fps).
- not possible to dynamically switch between implementation

TODO:
- sur les côté le clamp en donuts il déconne
- le game of life déconne en gpu ???

- utiliser la shared memory pour lire tuile par tuile
- it's possible to dynamically switch between GPU/CPU implementation ?? (flemne)

- un problème actuellement sur la bordure ??
