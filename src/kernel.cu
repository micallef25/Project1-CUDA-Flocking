#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

//#define NEIGHBORS 81
/*****************
* Configuration *
*****************/

// change to something more mathematical...
//#ifdef NEIGHBORS == 8
	#define MIN_NEIGHBORS -2
	#define MAX_NEIGHBORS 2
//#elif NEIGHBORS == 27
//	#define MIN_NEIGHBORS -2
//	#define MAX_NEIGHBORS 2
//#elif NEIGHBORS == 81
//	#define MIN_NEIGHBORS -3
//	#define MAX_NEIGHBORS 3
//#endif

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// Additional buffers for 2.3
glm::vec3 *dev_CoherentPos;
glm::vec3 *dev_CoherentVel;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // all the grid stuff is calulated so we can do some grid mallocing the mallocs for 2.1
  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("malloc failed");
  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("malloc failed");

  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("malloc failed");
  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("malloc failed");

  cudaMalloc((void**)&dev_CoherentPos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("malloc failed");
  cudaMalloc((void**)&dev_CoherentVel, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("malloc failed");


  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}

/*
* this for the grid based search we want to clamp so we are not checking sometihng potentially far away 
* or something that would be on the other side of the grid.
*/
__device__ void clamp_x_y_z(int* x, int* y, int* z, int resolution)
{
	// now make sure nothing is going to wrap around or go out of bounds
	// ie if x was 0 so and we added -1 we want to check 0 for now
	(*x) = imax( (*x), 0);
	(*y) = imax( (*y), 0);
	(*z) = imax( (*z), 0);

	// now to make sure we don't wrap around the other way...
	// ie if our max was 10 and we are in position 10 and added 1 we would be at 11 so set to grid max for now
	(*x) = imin( (*x), resolution -1);
	(*y) = imin( (*y), resolution -1);
	(*z) = imin( (*z), resolution-1);
}

/*
* these rules are based off the parkinsons notes found here http://www.vergenet.net/~conrad/boids/pseudocode.html
*/
__device__ void compute_rules(int my_tid, int neighbor1, int neighbor3, glm::vec3* perceived_center, glm::vec3* perceived_velocity, glm::vec3* small_distance_away, glm::vec3* v1, glm::vec3* v2, glm::vec3* v3, const glm::vec3* pos )
{
	// our weights are calculated. now we can scale appropriately	
	// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
	if (neighbor1)
	{
		(*perceived_center) /= neighbor1;
		(*v1) = ( (*perceived_center) - pos[my_tid]) * rule1Scale;
	}

	// Rule 2: boids try to stay a distance d away from each other
	(*v2) = (*small_distance_away) * rule2Scale;


	// Rule 3: boids try to match the speed of surrounding boids. Avoid div by zero
	if (neighbor3)
	{
		(*perceived_velocity) /= neighbor3;
		(*v3) = (*perceived_velocity) * rule3Scale;
	}
	return;
}

/*
* this is also based off the pseduo code from the notes http://www.vergenet.net/~conrad/boids/pseudocode.html
*/
__device__ void determine_distances(int my_tid, int other_tid, float distance, int* neighbor1, int* neighbor3, glm::vec3* perceived_center, glm::vec3* perceived_velocity, glm::vec3* small_distance_away, const glm::vec3* pos, const glm::vec3* vel)
{
	// give weight if we are close enough
	// rule 1
	if (distance < rule1Distance)
	{
		(*perceived_center) += pos[other_tid];
		(*neighbor1)++;
	}

	// give weight if we are close enough
	// rule 2
	if (distance < rule2Distance)
	{
		(*small_distance_away) -= (pos[other_tid] - pos[my_tid]);
	}

	// rule 3
	if (distance < rule3Distance)
	{
		(*neighbor3)++;
		(*perceived_velocity) += vel[other_tid];
	}

	return;
}

/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  
    float distance = 0.0f;	
    int neighbor1 = 0;
    int neighbor3 = 0;

    glm::vec3 v1 = glm::vec3(0.0f,0.0f,0.0f);
    glm::vec3 v2 = glm::vec3(0.0f,0.0f,0.0f);
    glm::vec3 v3 = glm::vec3(0.0f,0.0f,0.0f);
    glm::vec3 own_velocity = glm::vec3(0.0f, 0.0f, 0.0f);
    
    glm::vec3 perceived_center = glm::vec3(0.0f,0.0f,0.0f); 
    glm::vec3 perceived_velocity = glm::vec3(0.0f,0.0f,0.0f); 
    glm::vec3 small_distance_away = glm::vec3(0.0f,0.0f,0.0f); 
    
    // search through all positions and give appropriate weights based on our position.
	// if we have alot of neighbors we will give weight to stay close
	// if we have an enemy near we will give weight to avoid.
	for (int i = 0; i < N; i++)
	{
		if( i != iSelf)
		{
			distance = glm::distance(pos[i], pos[iSelf]);
			determine_distances(iSelf, i, distance, &neighbor1, &neighbor3, &perceived_center, &perceived_velocity, &small_distance_away, pos, vel);
		}
        else
        {
            own_velocity = vel[i];
        }
	}
	
	compute_rules(iSelf, neighbor1, neighbor3, &perceived_center, &perceived_velocity, &small_distance_away, &v1, &v2, &v3, pos);
    return v1+v2+v3+own_velocity;
}
    

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x); 

  // no work to be done for this thread
  if( tid >= N )
      return;

  // compute the speed algo based off psuedo code in parkinsons notes
  // http://www.vergenet.net/~conrad/boids/pseudocode.htmlZZ
  glm::vec3 new_velocity = computeVelocityChange(N,tid,pos,vel1);

  
  // Clamp the speed
  if( glm::length(new_velocity) > maxSpeed )
	  new_velocity = glm::normalize(new_velocity) * maxSpeed; // returns a vector in the same direction but with length of 1
  
 
  // Record the new velocity into vel2. Question: why NOT vel1?
  // we are reading from vel1 in computechange thus we do not want to overwrite the data for another thread.
  vel2[tid] = new_velocity;


}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
  
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  // no work   
  if(tid >= N)
      return;
  
  
  // TODO-2.1
    // - Label each boid with the index of its grid cell. 
  glm::ivec3 BoidPos = (pos[tid] - gridMin) * inverseCellWidth;
  gridIndices[tid] = gridIndex3Dto1D(BoidPos.x, BoidPos.y, BoidPos.z,gridResolution);
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
  indices[tid] = tid;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"

  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if( tid >= N)
      return;

  int cell = particleGridIndices[tid];
  int next_cell = particleGridIndices[tid+1];

  if( tid == 0 )
    gridCellStartIndices[cell] = 0; // start 

  else if( tid == N-1 )
      gridCellEndIndices[cell] = N-1; // end 
  
  else if( cell != next_cell ) // cell != cell + 1  
  {
    gridCellStartIndices[next_cell] = tid+1;
    gridCellEndIndices[cell] = tid;
  }


}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
	
	float distance = 0.0f;
	int neighbor1 = 0;
	int neighbor3 = 0;

	glm::vec3 v1 = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 v2 = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 v3 = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 own_velocity = glm::vec3(0.0f, 0.0f, 0.0f);

	glm::vec3 perceived_center = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 perceived_velocity = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 small_distance_away = glm::vec3(0.0f, 0.0f, 0.0f);

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid >= N)
	{
		return;
	}

	// get the boid index to see the grid we placed ourself into
	glm::ivec3 boid = (pos[tid] - gridMin) * inverseCellWidth;
	

	// search through all positions and give appropriate weights based on our position.
	// if we have alot of neighbors we will give weight to stay close
	// if we have an enemy near we will give weight to avoid.
	for (int i = MIN_NEIGHBORS; i <=MAX_NEIGHBORS; i++)
		for (int j = MIN_NEIGHBORS; j <= MAX_NEIGHBORS; j++)
			for (int k = MIN_NEIGHBORS; k <= MAX_NEIGHBORS; k++)
			{
				// compute the neighbors x,y,z
				int neighbor_x = boid.x + i;
				int neighbor_y = boid.y + j;
				int neighbor_z = boid.z + k;
				
				// clamp our neighbors
				clamp_x_y_z(&neighbor_x,&neighbor_y,&neighbor_z, gridResolution);
				
				// now convert to a grid position
				int Cell = gridIndex3Dto1D(neighbor_x, neighbor_y, neighbor_z, gridResolution);

				// is there even anything in the grid? remember we reset these every dt step
				if (gridCellStartIndices[Cell] != -1)
				{
					// traverse through start until end
					for (int indice = gridCellStartIndices[Cell]; indice <= gridCellEndIndices[Cell]; indice++)
					{
						// this boid is one to check with its data finally
						int boid_idx = particleArrayIndices[indice];
						//dont compute yourself similar to naive now compute the rules
						if (boid_idx != tid)
						{
							distance = glm::distance(pos[boid_idx], pos[tid]);
							determine_distances(tid, boid_idx, distance, &neighbor1, &neighbor3, &perceived_center, &perceived_velocity, &small_distance_away, pos, vel1);
						}
						else
						{
							own_velocity = vel1[tid];
						}
					}
				}
			}

	// copute our velocities
	compute_rules(tid, neighbor1, neighbor3, &perceived_center, &perceived_velocity, &small_distance_away, &v1, &v2, &v3, pos);

	glm::vec3 new_velocity = v1 + v2 + v3 + own_velocity;

	// Clamp the speed
	if (glm::length(new_velocity) > maxSpeed)
		new_velocity = glm::normalize(new_velocity) * maxSpeed;

	vel2[tid] = new_velocity;
}

// copy to the coherent buffers. so we have one less level of indirection and a few less memory reads per thread in our main loop
// make a kern so many threads can compute quickly
__global__ void kernCreateCoherentBuffs(int N, int* particlearray, glm::vec3* pos, glm::vec3* vel, glm::vec3* coherentpos, glm::vec3* coherentvel)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid >= N)
		return;

	// 
	int c_index = particlearray[tid];

	coherentpos[tid] = pos[c_index];
	coherentvel[tid] = vel[c_index];

}


__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
  
	// this is the same routine as before except we reordered our grid so we have one less table read
	float distance = 0.0f;
	int neighbor1 = 0;
	int neighbor3 = 0;

	glm::vec3 v1 = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 v2 = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 v3 = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 own_velocity = glm::vec3(0.0f, 0.0f, 0.0f);

	glm::vec3 perceived_center = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 perceived_velocity = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 small_distance_away = glm::vec3(0.0f, 0.0f, 0.0f);

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid >= N)
	{
		return;
	}

	// fingure our grid or what bin we would be dumped into
	glm::ivec3 boid = (pos[tid] - gridMin) * inverseCellWidth;


	// search through all positions and give appropriate weights based on our position.
	// if we have alot of neighbors we will give weight to stay close
	// if we have an enemy near we will give weight to avoid.
	// thi s loop is changed from x.y.z to y,z,x so that we can read memory sequentially
	// the with z.y.x the call to grindIndex3dto1d is more sequential
	for (int k = MIN_NEIGHBORS; k <= MAX_NEIGHBORS; k++)
		for (int j = MIN_NEIGHBORS; j <= MAX_NEIGHBORS; j++)
			for (int i = MIN_NEIGHBORS; i <= MAX_NEIGHBORS; i++)
			{
				// compute the neighbors x,y,z check z,y,x
				int neighbor_x = boid.x + i;
				int neighbor_y = boid.y + j;
				int neighbor_z = boid.z + k;

				clamp_x_y_z(&neighbor_x,&neighbor_y,&neighbor_z, gridResolution);

				// now convert to a grid position
				int Cell = gridIndex3Dto1D(neighbor_x, neighbor_y, neighbor_z, gridResolution);

				// is there even anything in the grid?
				if (gridCellStartIndices[Cell] != -1)
				{
					// traverse through start until end
					for (int indice = gridCellStartIndices[Cell]; indice <= gridCellEndIndices[Cell]; indice++)
					{
						//dont compute yourself similar to naive now compute the rules
						if (indice != tid)
						{
							// these are coherent positions. The only difference is from the grid is we do not now have to
							// get hte boid idx. So a little less memory read per thread.
							distance = glm::distance(pos[indice], pos[tid]);
							determine_distances(tid, indice, distance, &neighbor1, &neighbor3, &perceived_center, &perceived_velocity, &small_distance_away, pos, vel1);
						}
						else
						{
							own_velocity = vel1[tid];
						}
					}
				}
			}


	compute_rules(tid, neighbor1, neighbor3, &perceived_center, &perceived_velocity, &small_distance_away, &v1, &v2, &v3, pos);

	glm::vec3 new_velocity = v1 + v2 + v3 + own_velocity;

	// Clamp the speed
	if (glm::length(new_velocity) > maxSpeed)
		new_velocity = glm::normalize(new_velocity) * maxSpeed;

	vel2[tid] = new_velocity;

}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  // TODO-1.2 ping-pong the velocity buffers
  dim3 blockspergrid((numObjects + blockSize -1) / blockSize );

  kernUpdateVelocityBruteForce <<< blockspergrid, blockSize >>>(numObjects, dev_pos, dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("brute force failed");

  kernUpdatePos <<< blockspergrid,blockSize >>>(numObjects,dt, dev_pos,dev_vel1);
  checkCUDAErrorWithLine("update pos failed");

  // ping pong the buffer
  std::swap( dev_vel1, dev_vel2 );
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  dim3 blockspergrid((gridCellCount + blockSize - 1) / blockSize);
  dim3 boidsblockspergrid((numObjects + blockSize - 1) / blockSize);
  // Reset the buffers to -1 indicating nothing init'd
  kernResetIntBuffer <<< blockspergrid,blockSize >>>(gridCellCount, dev_gridCellStartIndices, -1);
  checkCUDAErrorWithLine("int buffer");
  kernResetIntBuffer <<< blockspergrid,blockSize >>>(gridCellCount,dev_gridCellEndIndices, -1);
  checkCUDAErrorWithLine("int buffer ");

  kernComputeIndices << < boidsblockspergrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  checkCUDAErrorWithLine("compute indices");

// now that we have placed boids in grids we can sort
  thrust::device_ptr<int> keys(dev_particleGridIndices);
  thrust::device_ptr<int> values(dev_particleArrayIndices);

  thrust::sort_by_key(keys,keys+numObjects,values); 
  checkCUDAErrorWithLine("thrust");

// 
  kernIdentifyCellStartEnd << < boidsblockspergrid, blockSize >> > (numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
  checkCUDAErrorWithLine("cell end");

// the "update" 
  kernUpdateVelNeighborSearchScattered << < boidsblockspergrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("neighbor search failed");
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  //	https://thrust.github.io/doc/group__sorting_gabe038d6107f7c824cf74120500ef45ea.html#gabe038d6107f7c824cf74120500ef45ea
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - Perform velocity updates using neighbor search
  // - Update positions
  kernUpdatePos << < blockspergrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel1);
  checkCUDAErrorWithLine("update pos failed");

  // - Ping-pong buffers as needed
  std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
	
	// same methdology as the scattered grid approach
	dim3 blockspergrid((gridCellCount + blockSize - 1) / blockSize);
	dim3 boidsblockspergrid((numObjects + blockSize - 1) / blockSize);
	// Reset the buffers to -1 indicating nothing init'd
	kernResetIntBuffer << < blockspergrid, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
	checkCUDAErrorWithLine("int buffer");
	kernResetIntBuffer << < blockspergrid, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
	checkCUDAErrorWithLine("int buffer ");

	kernComputeIndices << < boidsblockspergrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
	checkCUDAErrorWithLine("compute indices");

	thrust::device_ptr<int> keys(dev_particleGridIndices);
	thrust::device_ptr<int> values(dev_particleArrayIndices);

	thrust::sort_by_key(keys, keys + numObjects, values);
	checkCUDAErrorWithLine("thrust");

	kernIdentifyCellStartEnd << < boidsblockspergrid, blockSize >> > (numObjects ,dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
	checkCUDAErrorWithLine("cell end");
	
	// the difference between grid and coherent. create our coherent buffer and use that in update
	kernCreateCoherentBuffs << < boidsblockspergrid, blockSize >> > (numObjects, dev_particleArrayIndices,dev_pos,dev_vel1, dev_CoherentPos, dev_CoherentVel);

	kernUpdateVelNeighborSearchCoherent << < boidsblockspergrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_CoherentPos, dev_CoherentVel, dev_vel2);
	checkCUDAErrorWithLine("neighbor search failed");
	// Uniform Grid Neighbor search using Thrust sort.
	// In Parallel:
	// - label each particle with its array index as well as its grid index.
	//   Use 2x width grids.
	// - Unstable key sort using Thrust. A stable sort isn't necessary, but you
	//   are welcome to do a performance comparison.
	//	https://thrust.github.io/doc/group__sorting_gabe038d6107f7c824cf74120500ef45ea.html#gabe038d6107f7c824cf74120500ef45ea
	// - Naively unroll the loop for finding the start and end indices of each
	//   cell's data pointers in the array of boid indices
	// - Perform velocity updates using neighbor search
	// - Update positions
	kernUpdatePos << < blockspergrid, blockSize >> > (numObjects, dt, dev_CoherentPos, dev_vel2);
	checkCUDAErrorWithLine("update pos failed");

	// ping pong th position this is important because when we create the buffers we read from the last position.
	std::swap(dev_CoherentPos, dev_pos);

	// - Ping-pong buffers as needed
	std::swap(dev_vel1, dev_vel2);
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  cudaFree(dev_gridCellEndIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);

  cudaFree(dev_CoherentPos);
  cudaFree(dev_CoherentVel);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);

  
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
