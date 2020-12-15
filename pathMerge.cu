/**
    Merges to array A and B in M using a path

    @file pathMerge.cu
    @author Dang Vu Laurent Durand Homer
    @version 1.0 14/12/20
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#include <string>
#include <iostream>
#include <stdlib.h>


/**
    Verify cuda calls and return cuda error if any
*/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


/**
    Initialise ascendant array with random values
    @param array : the array to fill with random ascendant values
          size : Size of the arrays
          adder : Each x_i is higher than x_i-1 of a random value between 0 and adder
*/
void init_array(int* array, int size, int const adder=10)
{
    array[0] = rand()%adder;
    for(int i = 0; i < size;i++)
    {
        array[i] = array[i-1] + rand()%adder;
    }
}

/**
    Print an array of size size
    @param a : array to print
           size : size of arrays
*/
void print_array(int* a, int size)
{
    printf("[");
    for(int i = 0; i < size;i++)
    {
        printf("%d " , a[i]);
    }
    printf("]\n");
}

/**
    Sequential version of merge
    @param a_k, b_k : array to merge
           m_k : merge of a and b
           n_a, n_b, n_b : respective sizes of a_k, b_k, m_k
*/
void mergeSeq(int *a_k, int *b_k, int *m_k, int n_a, int n_b, int n_m)
{
  int i, j;
  i=0;
  j=0;
  while(i+j < n_m)
  {
    if (i>= n_a)
    {
      m_k[i+j]=b_k[j];
      j++;
    }
    else if (j>= n_b || a_k[i] < b_k[j])
    {
      m_k[i+j]=a_k[i];
      i++;
    }
    else
    {
      m_k[i+j]=b_k[j];
      j++;
    }
  }
}


/**
    Parallel version of merge of A and B with |A| + |B| <= 1024
    @param d_a, d_b : device versions of arrays to merge
           d_m : device version of merge of a and b
           n_a, n_b, n_b : respective sizes of d_a, d_b, d_m
*/
__device__ void mergeSmall_k(int* d_a, int* d_b, int* d_m, int n_a, int n_b, int n_m){
    int i = threadIdx.x;
    if(i < n_m)
    {
        int2 K;
        int2 P;
        int2 Q;
        if(i > n_a)
        {
            K.x = i - n_a;
            K.y = n_a;
            P.x = n_a;
            P.y = i - n_a;
        }
        else
        {
            K.x = 0;
            K.y = i;
            P.x = i;
            P.y = 0;
        }

        int offset = 0;
        while(1)
        {
            offset = abs(K.y - P.y)/2;
            Q.x = K.x + offset;
            Q.y = K.y - offset;
            if(Q.y >= 0 && Q.x <= n_b && (Q.y == n_a || Q.x == 0 || d_a[Q.y] > d_b[Q.x - 1]))
            {
                if(Q.x == n_b || Q.y == 0 || d_a[Q.y - 1] <= d_b[Q.x])
                {
                    if(Q.y < n_a && (Q.x == n_b || d_a[Q.y] <= d_b[Q.x]))
                    {
                        d_m[i] = d_a[Q.y];
                    }
                    else
                    {
                        d_m[i] = d_b[Q.x];
                    }
                    break;
                }
                else
                {
                    K.x = Q.x + 1;
                    K.y = Q.y - 1;
                }
            }
            else
            {
              P.x = Q.x - 1;
              P.y = Q.y + 1;

            }

        }
    }
}


/**
    Parallel version of merge of A and B of any sizes
    @param a, b : device versions of arrays to merge
           m : device version of merge of a and b
           n_a, n_b, n_b : respective sizes of d_a, d_b, d_m
           path : points of the path to cut A and B to pieces to merge
           n_path : number of points in the path
           nb_partition : number of pieces of A and B (a_k and b_k) to merge with mergeSmall_k
*/
__global__ void mergeBig_k(int *m, int n_m, int *a, int n_a, int *b, int n_b, int2 *path, int n_path, int nbPartitions)
{
  int blockId = blockIdx.x;
  int threadId = threadIdx.x;
  int i = blockId * blockDim.x + threadId;
  if (blockId <= nbPartitions)//On utilise un block pour chaque partition
  {
    int x0, y0, x1, y1;
    x0 = path[blockId].x;
    y0 = path[blockId].y;
    x1 = path[blockId+1].x;
    y1 = path[blockId+1].y;

    const int dimx=x1-x0;
    const int dimy = y1-y0;

    //A modifier par dimx dimy dimx+dimy
     __shared__ int a_k[1024];
     __shared__ int b_k[1024];
     __shared__ int m_k[1024];

    if (threadId < dimx) //On rempli a_k[i] :  0 <= i < dimx
    {
      a_k[threadId] = a[x0+threadId];
    }
    else if (threadId < dimy+dimx)//On rempli b_k[i] : indice dimx <= i < dimx+dimy+1
    {
      b_k[threadId-dimx] = b[y0+threadId-dimx];
    }
    __syncthreads();
    mergeSmall_k(a_k, b_k, m_k, dimx, dimy, dimx+dimy);
    m[i] = m_k[threadId];

  }
}

/**
    Genearte the path to devide A and B to pieces that we'll give to mergeSmall_k
    @param pas: size of pieces
           path : store the points of the path
           n_path : number of points in the path
           nb_partition : number of pieces of A and B (a_k and b_k) to merge with mergeSmall_k
           d_a, d_b : device versions of arrays to merge
           n_a, n_b : respective sizes of d_a, d_b
*/
__global__ void pathBig_k(int pas, int2* path, int n_path , int* d_a, int n_a ,int* d_b, int n_b)
{
    int thread_i = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_i <= (n_a + n_b)/pas)        //<------------//On vérifie que l'indice du thread est inférieur à la taille du tableau de retour et qu'il est un multiple du pas
    {
        int i = thread_i*pas;
        int2 K;
        int2 P;
        int2 Q;
        if(i > n_a)
        {
            K.x = i - n_a;
            K.y = n_a;
            P.x = n_a;
            P.y = i - n_a;
        }
        else
        {
            K.x = 0;
            K.y = i;
            P.x = i;
            P.y = 0;
        }

        int offset = 0;
        while(1)
        {
            //Calcul des coordonnées du milieu de P et K
            offset = abs(K.y - P.y)/2;
            Q.x = K.x + offset;
            Q.y = K.y - offset;

            //
            if(Q.y >= 0 && Q.x <= n_b && (Q.y == n_a || Q.x == 0 || d_a[Q.y] > d_b[Q.x - 1]))
            {
                //
                if(Q.x == n_b || Q.y == 0 || d_a[Q.y - 1] <= d_b[Q.x])
                {
                  break;
                }
                else
                {
                    K.x = Q.x + 1;
                    K.y = Q.y - 1;
                }
            }
            else
            {
              P.x = Q.x - 1;
              P.y = Q.y + 1;

            }

        }
        //printf("thread : %d => (%d, %d)\n", thread_i, Q.y, Q.x);
        //!\\ Problème ordre x et y
        path[thread_i].x=Q.y;
        path[thread_i].y=Q.x;
    }
    //Si |m| n'est pas un mutliple de pas, le thread 0 ajoute (n_a, n_b) à la fin du tableau
    if (thread_i==0 && (n_a+n_b)%pas!=0)
    {
      //printf("thread : %d => (%d, %d)\n", thread_i, n_a, n_b);
      path[n_path-1].x=n_a;
      path[n_path-1].y=n_b;
    }

}

/**
    verify that A and B are correctly merge in M
*/
int assertMerge(int *tab, int *tab2, int size)
{
  for (int i=0; i<size-1; i++)
  {
    if (tab[i] > tab[i+1] || tab[i] != tab2[i] || (i>10000 && tab[i] == 0))
    {
      printf("WARNING : Unsuccessful merge on indice %d ...\n", i);
      return 0;
    }
  }
  printf("Successful merge !\n");
  return 1;

}


/**
    Merge 2 lists of arrays {A_i} and {B_i} in {M_i}1<=i<=N
    @param argv[1] : size of A
           argv[2] : size of B
*/
int main(int argc, char *argv[])
{
    std::clock_t startS, endS;
    float seqMergeTime, parMergeTime, DoH, HoD;

    srand(time(NULL));
    int n_a, n_b;
    int pas;
    if(argc>= 3)
    {
      n_a = atoi(argv[1]);
      n_b = atoi(argv[2]);
      pas = atoi(argv[3]);
    }
    else
    {
      n_a = 100;
      n_b = 100;
      pas = 1024;
    }
    int n_m = n_a+n_b;
     // <1024
    int nbPartitions = n_m/pas+(n_m%pas!=0); // On ajoute 1 si n_m n'est pas un mutliple de p
    int n_path = (1 + nbPartitions); //1(pour (0,0)) + |m|/pas(nbr de morceau de taille pas) + 1(si dernier morceau de taille < pas))
    printf("========== Merge of A and B ==========\n");
    printf("* Size of A : %d\n", n_a);
    printf("* Size of B : %d\n", n_b);
    printf("* Step : %d\n* Nbr of partitions : %d\n\n", pas, nbPartitions);

    //Initialisation des tableaux a et b
    int *a, *aGPU;
    a = (int*)malloc(n_a*sizeof(int));
    init_array(a, n_a, 10);
    gpuErrchk(cudaMalloc(&aGPU, n_a*sizeof(int)));

    int *b, *bGPU;
    b = (int*)malloc(n_b*sizeof(int));
    init_array(b, n_b, 10);
    gpuErrchk(cudaMalloc(&bGPU, n_b*sizeof(int)));

    // print_array(b, n_b);
    // print_array(a, n_a);

    int *m, *mGPU, *mseq;
    m = (int*)malloc(n_m*sizeof(int));
    mseq = (int*)malloc(n_m*sizeof(int));
    gpuErrchk(cudaMalloc(&mGPU, n_m*sizeof(int)));

    //Declaration et allocation de path
    int2 *pathGPU;
    gpuErrchk(cudaMalloc(&pathGPU, n_path*sizeof(int2)));

    startS = std::clock();
    gpuErrchk(cudaMemcpy(aGPU, a, n_a*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(bGPU, b, n_b*sizeof(int), cudaMemcpyHostToDevice));
    endS = std::clock();
    HoD = (endS - startS) / (float) CLOCKS_PER_SEC;

    printf("Merge of A and B of size %d and %d runing...\n", n_a, n_b);
    startS = std::clock();
    //================ Parallel : =======================\\

    pathBig_k<<<nbPartitions/1024+1, 1024>>>(pas, pathGPU, n_path, aGPU, n_a, bGPU, n_b);

    mergeBig_k<<<nbPartitions, pas>>>(mGPU, n_m, aGPU, n_a, bGPU, n_b, pathGPU, n_path, nbPartitions);

    cudaDeviceSynchronize();
    endS = std::clock();
    parMergeTime = (endS - startS) / (float) CLOCKS_PER_SEC;

    //Copy device to host
    startS = std::clock();
    cudaMemcpy(m, mGPU, n_m*sizeof(int), cudaMemcpyDeviceToHost);
    endS = std::clock();
    DoH = (endS - startS) / (float) CLOCKS_PER_SEC;

    printf("Merge done !\n\n");

    //================ Sequential : =======================\\
    startS = std::clock();
    mergeSeq(a, b, mseq, n_a, n_b, n_m);
    endS = std::clock();
    seqMergeTime = (endS - startS) / (float) CLOCKS_PER_SEC;

    //print_array(m, n_m);
    // print_array(mseq, n_m);

    printf("\n========= Sequential merge : =============\n");
    printf("Total time elapsed : %f s\n", seqMergeTime);
    printf("\n");

    printf("========= Parallel merge : =============\n");
    printf("Total time elapsed : %f s\n", parMergeTime+DoH+HoD);
    printf("Time running algorithm : %f s\n", parMergeTime);
    printf("Time to copy Host to Device : %f s\n", HoD);
    printf("Time to copy Device to Host : %f s\n", DoH);
    assertMerge(m, mseq, n_m);
    printf("Parrallel algorithm is %f times faster than sequential merge !\n", seqMergeTime/parMergeTime);
    printf("Parrallel merge is %f times faster than sequential merge !\n", seqMergeTime/(parMergeTime+HoD+DoH));






    //desallocation
    cudaFree(aGPU);
    cudaFree(bGPU);
    cudaFree(pathGPU);

    return 0;
}
