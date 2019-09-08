**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Eric Micallef
  * https://www.linkedin.com/in/eric-micallef-99291714b/
  
* Tested on: Windows 10, i7-6700 @ 3.4GHz 16GB, Nvidia Quadro P1000 (Moore 100B Lab)


#For each implementation, how does changing the number of boids affect performance? Why do you think this is?

**Boids Graph**
![alt text](https://raw.github.com/micallef25/Project1-CUDA-Flocking/master/images/boids.png)

**Boids Raw**
![alt text](https://raw.github.com/micallef25/Project1-CUDA-Flocking/master/images/boidsraw.PNG)

#For each implementation, how does changing the block count and block size affect performance? Why do you think this is?

**Blocksize Raw**
![alt text](https://raw.github.com/micallef25/Project1-CUDA-Flocking/master/images/blocks_raw.PNG)

**Blocksize Graph**
![alt text](https://raw.github.com/micallef25/Project1-CUDA-Flocking/master/images/blocksize.png)

For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?


Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!
