#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

#define NUM_THREADS 256

int number, totalN;
double da;

extern double size;
//
//  benchmarking program
//



__global__ int place(double x, double y, double da, int number) {
    int xID = x / da;
    int yID = y / da;
    return xID * number + yID;
}

__global__ int place(particle_t &particle, double da, int number) {
    int xID = particle.x / da;
    int yID = particle.y / da;
    return xID * number + yID;
}

__global__ void assign_particles(int n, particle_t * particles, int* d_next, int* d_grids, double da, int number) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= n) return;

    int k = locationToID(particles[tid], da, number);
    d_next[tid] = atomicExch(&d_grids[k], tid);
}

__global__ void grids(int totalN, int* d_grids) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= totalN) return;

    d_grids[tid] = -1;
}



__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if( r2 > cutoff*cutoff )
        return;
    //r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
    double r = sqrt( r2 );

    //
    //  very simple short-range repulsive force
    //
    double coef = ( 1 - cutoff / r ) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;

}

__global__ void compute_forces_gpu(particle_t * particles, int n)
{
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= n) return;

    particles[tid].ax = particles[tid].ay = 0;
    for(int j = 0 ; j < n ; j++)
        apply_force_gpu(particles[tid], particles[j]);

}


__global__ void compute_grid_forces_gpu(particle_t * particles, int * d_next,int tot_num, int * d_grids, double dim, int num)
{
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= tot_num) return;

    int xID = tid / num;
    int yID = tid % num;
    int k = tid;

    for(int i = d_grids[tid]; i != -1; i = d_next[i]) {
        particle_t * p = &particles[i];

        p->ax = p->ay = 0;

        // check self
        compute_self_grid_forces(i, particles, d_next, d_grids[k]);

        // check other
        if(xID > 0) {
            compute_grid_forces(i, particles, d_next, d_grids[k - num]);
            if(yID > 0)
                compute_grid_forces(i, particles, d_next, d_grids[k - num - 1]);
            if(yID < num - 1)
                compute_grid_forces(i, particles, d_next, d_grids[k - num + 1]);
        }
        if(xID < num - 1) {
            compute_grid_forces(i, particles, d_next, d_grids[k + num]);
            if(yID > 0)
                compute_grid_forces(i, particles, d_next, d_grids[k + num - 1]);
            if(yID < num - 1)
                compute_grid_forces(i, particles, d_next, d_grids[k + num + 1]);
        }
        if(yID > 0) compute_grid_forces(i, particles, d_next, d_grids[k - 1]);
        if(yID < num - 1) compute_grid_forces(i, particles, d_next, d_grids[k + 1]);
    }
}


__global__ void move_gpu (particle_t * particles, int n, double size)
{

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= n) return;

    particle_t * p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    //
    //  bounce from walls
    //
    while( p->x < 0 || p->x > size )
    {
        p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
        p->vx = -(p->vx);
    }
    while( p->y < 0 || p->y > size )
    {
        p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
        p->vy = -(p->vy);
    }

}



int main( int argc, char **argv )
{
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize();

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));

    set_size( n );

    init_particles( n, particles );

    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

  //  cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;

    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize();      // can I commented out from original
    double simulation_time = read_timer( );

    num = (int)ceil(size*1.0 / DIM); // we get the num of the grid for one directions
    tot_num = num * num; // total number of grids
    da = size/num; // the acutal size of a subgrid

    int blks = (n + NUM_THREADS - 1) / NUM_THREADS;
    int g_blks = (tot_num + NUM_THREADS - 1) / NUM_THREADS;


    int * d_grids;
    cudaMalloc((void **) &d_grids, tot_num * sizeof(int));
    int * d_next;
    cudaMalloc((void **) &d_next, n * sizeof(int));

    cudaThreadSynchronize();

 //   clear_grids <<< g_blks, NUM_THREADS >>> (tot_num, d_grids);

    assign_particles <<< blks, NUM_THREADS >>> (n, d_particles, d_next, d_grids, da, num);

    cudaThreadSynchronize();

    set_grid_time = read_timer() - set_grid_time;


  //  cudaThreadSynchronize();
    double simulation_time = read_timer( );



    for( int step = 0; step < NSTEPS; step++ )
    {
        //
        //  compute forces
        //

        int blks = (n + NUM_THREADS - 1) / NUM_THREADS;
        compute_forces_gpu <<< blks, NUM_THREADS >>> (d_particles, n);

        //
        //  move particles
        //
        move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);

        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 ) {
            // Copy the particles back to the CPU
            cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save( fsave, n, particles);
        }
    }
   // cudaThreadSynchronize();        //may need to commet out to upspeed
    simulation_time = read_timer( ) - simulation_time;

    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );

    free( particles );
    cudaFree(d_particles);
    if( fsave )
        fclose( fsave );

    return 0;
}