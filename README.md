**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Eric Micallef
  * https://www.linkedin.com/in/eric-micallef-99291714b/
  
* Tested on: Windows 10, i7-6700 @ 3.4GHz 16GB, Nvidia Quadro P1000 (Moore 100B Lab)

[![Watch the video](https://img.youtube.com/vi/PD1hAzGXRkg/0.jpg)](https://youtu.be/PD1hAzGXRkg)

### Performance Analysis

**For each implementation, how does changing the number of boids affect performance? Why do you think this is?**

The amount of boids effects appears to effect all three system in an exponential way. as the the amount of boids
are introduced the system decays. The naive approach has a much quicker decay. This is because each boid must check every other boid in the system so as the amount of boids grows the more memory reads and writes grows very quickly. 

The decay for the Coherent and the grid based approach is much less and is also shown. This is because we are only checking for neighbors around us so instead of checking 5000 other boids we may only have to check 5 or 6 boids. So as the system encounters more boids we the memory hit is not as severe.

![alt text](https://raw.github.com/micallef25/Project1-CUDA-Flocking/master/images/boids.png)

![alt text](https://raw.github.com/micallef25/Project1-CUDA-Flocking/master/images/boidsraw.PNG)

**For each implementation, how does changing the block count and block size affect performance? Why do you think this is?**

This data was collected with 5000 boids. 

As we can see there is not too much of a difference until we get down to a blocksize of about 16 or 8. we see the naive implementation suffers around 75 and 50 percent respectively. The grid and coherent based approach suffer as well but not as much. We see this effect because there are less threads to hide the memory latency. In the naive approach we have alot of memory reads and writes and with less threads means we can not hide these latencies as well. We begin to see some of this in the other approaches but the system has less memory reads and writes so the impact is not as severe. 

Intererstingly with a higher block size the frame rate increased but not noticeably. So we can see that the use of more resources does not always lead to higher performance.

![alt text](https://raw.github.com/micallef25/Project1-CUDA-Flocking/master/images/blocksize.png)

![alt text](https://raw.github.com/micallef25/Project1-CUDA-Flocking/master/images/blocks_raw.PNG)

**For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?**

There was an improvement as the graphs above show which I expected. I expected this because the algorithms are very similar in nature. One difference is in our main update function when we loop over X,Y,Z we have one less memory read. This can help speed up our implementation. 

Another hypothesis is that since our reads are a bit more contiguous in nature now maybe the GPU reads a few more bytes at a time and store them in registers. For example, maybe it can read neighbors 1,2,3,4 at once from main memory and store them in registers for quick access next iteration. 


**Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!**

Yes, the system actually slowed down about 50% when increasing from 8 to 27. I was not expecting this 

