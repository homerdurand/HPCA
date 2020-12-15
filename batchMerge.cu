/**
    Giving a large number of arrays {A_i}1<=i<=N and
    {B_i}1<=i<=N with |A_i| + |B_i| = d < 1024, merges
    arrays twos by two and return a list of arrays {M_i}1<=i<=N

    @file batchMerge.cu
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
#define gpuCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
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
  if (size >=1)
  {
    array[0] = rand()%adder;
    if (size >=2)
    {
      for(int i = 1; i < size;i++)
      {
          array[i] = array[i-1] + rand()%adder;
      }
    }
  }
}

/**
    Initialise two lists of ascendant arrays A and B with random values
    @param A : the first list of arrays to fill with random ascendant values
           B : the Second list of arrays to fill with random ascendant values
          sizes : Sizes of each subarrays
          d : for all 1<=i<=N, |A_i| + |B_i| = d
          N : number of subarrays in A and B
*/
void init_list_array(int **A, int **B, int N, int d, int2 *sizes)
{
    int sizeAi, sizeBi;
    for (int i = 0; i < N; i++)
    {
      //Initialiser les tailles de Ai et Bi
      sizeAi = rand()%d;
      sizeBi = d - sizeAi;

      //Stocker les tailles dans sizes
      sizes[i].x = sizeAi;
      sizes[i].y = sizeBi;

      //Allocation A et B
      A[i] = (int*)malloc(sizes[i].x*sizeof(int));
      B[i] = (int*)malloc(sizes[i].y*sizeof(int));

      //Initialisation A et B
      init_array(A[i], sizes[i].x);
      init_array(B[i], sizes[i].y);
    }
}

/**
    Return first indices of each arrays wich have their sizes in sizes for A and B as flat arrays
    @param indices : INdices of first element of each subarrays in A and B
           sizes : Sizes of each subarrays in A and B
           N : number of subarrays in A and B
*/
void indices_array(int2 *indices, int2 *sizes, int N)
{
  indices[0].x=0;
  indices[0].y=0;
  for (int i = 1; i < N; i++)
  {
    indices[i].x = indices[i-1].x+sizes[i-1].x;
    indices[i].y = indices[i-1].y+sizes[i-1].y;

  }
}

/**
    Return total sizes of A and B with A and B as flat arrays
    @param sizes : Sizes of each subarrays in A and B
           N : number of subarrays in A and B
*/
int2 get_totalSizes(int2 *sizes, int N)
{
  int2 totalSizes;
  totalSizes.x = 0;
  totalSizes.y = 0;
  for (int i = 0; i < N; i++)
  {
    totalSizes.x = totalSizes.x + sizes[i].x;
    totalSizes.y = totalSizes.y + sizes[i].y;
  }
  return totalSizes;
}


/**
    Return flat arrays of  A and B
    @param flatA : flat version of A
           flat B : flat version of B
           A : 2D array of int
           B : 2D array of int2
           N : Number of subarrays in A and B
           d : |A_i| + |B_i| = d
           sizes : sizes of subarrays in A and B
*/
void flat_arrays(int *flatA, int *flatB, int **A, int **B, int N, int d, int2 *sizes)
{
  int ia = 0;
  int ib = 0;
  for (int i = 0; i < N; i++)
  {

    for (int j = 0; j < sizes[i].x; j++)
    {
      flatA[ia] = A[i][j];
      ia = ia +1;
    }
    for (int j = 0; j < sizes[i].y; j++)
    {
      flatB[ib] = B[i][j];
      ib = ib +1;
    }
  }
}

/**
    Return 2D version of a flat array with N subarrays of sizes d
    @param unFlatM : unFLatversion of M
           flatM : flat version of M
           N : Number of subarrays in M
           d : Sizes of subarrays
*/
void unflat_arrays(int **unFlatM, int *flatM, int N, int d)
{
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < d; j++)
    {
      unFlatM[i][j] = flatM[i*d + j];
    }
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
        //printf("i = %d | v = %d " ,i, a[i]);
        printf("%d " ,a[i]);
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

void mergeSeqBatch(int **a, int **b, int **m, int N, int d, int2 *sizes)
{
  for (int i = 0; i < N; i++)
  {
    mergeSeq(a[i], b[i], m[i], sizes[i].x, sizes[i].y, d);
  }
}

/**
    Globall version of parallel merge of {A_i} and {B_i} in {M_i} with |M|<1024
    @param aGPU : flat version of arrays {A_i} in device
           bGPU : flat version of array {B_i } in device
           mGPU : flat version of array {M_i} in device
           d : |A| + |B| = |M| = d
           N : number of subarrays in A and B
           indices : indices of the start of each subarrays in A and B
           sizes : sizes of each subarrays in A and B
*/
__global__ void mergeSmallBatch_k(int* aGPU, int* bGPU, int* mGPU, int d, int N, int2 *indices, int2 *sizes)
{
    int tidx = threadIdx.x%d; // The indice of the thread in the "new" blocks of size d
    int Qt = (threadIdx.x-tidx)/d;
    int gbx = Qt + blockIdx.x*(blockDim.x/d); // The indice of "new" blocks of size d
    int i = gbx*d + tidx; // The global indice of the thread
    if(i < N*d)
    {
        int2 K;
        int2 P;
        int2 Q;
        if(tidx > sizes[gbx].x)
        {
            K.x = tidx - sizes[gbx].x;
            K.y = sizes[gbx].x;
            P.x = sizes[gbx].x;
            P.y = tidx - sizes[gbx].x;
        }
        else
        {
            K.x = 0;
            K.y = tidx;
            P.x = tidx;
            P.y = 0;
        }

        int offset = 0;
        while(1)
        {
            offset = abs(K.y - P.y)/2;
            Q.x = K.x + offset;
            Q.y = K.y - offset;
            if(Q.y >= 0 && Q.x <= sizes[gbx].y && (Q.y == sizes[gbx].x || Q.x == 0 || aGPU[indices[gbx].x + Q.y] > bGPU[indices[gbx].y + Q.x - 1]))
            {
                if(Q.x == sizes[gbx].y || Q.y == 0 || aGPU[indices[gbx].x + Q.y - 1] <= bGPU[indices[gbx].y + Q.x])
                {
                    if(Q.y < sizes[gbx].x && (Q.x == sizes[gbx].y || aGPU[indices[gbx].x + Q.y] <= bGPU[indices[gbx].y + Q.x]))
                    {
                        mGPU[i] = aGPU[indices[gbx].x + Q.y];
                    }
                    else
                    {
                        mGPU[i] = bGPU[indices[gbx].y + Q.x];
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
    verify that each tab_i < tab_i+1 in tab
    @param tab : array
    @return : 0 any tab_i > tab_i+1, either 1
*/
int assertOrder(int *tab, int size){
  for (int i=0; i<size-1; i++){
    if (tab[i] > tab[i+1]){
      printf("WARNING : Unsuccessful merge or sort ... : unordered array on indice %d ...\n", i);
      printf("tab[i]= %d > tab[i+1] = %d\n", tab[i], tab[i+1]);
      return 0;
    }
  }
  return 1;
}

/**
    verify that each element of tab is in tab_2
    @param tab : first array
            n1 : size of tab
            tab2 : second array
            n2 : size of tab2

    @return : 0 any tab_i > tab_i+1, either 1
*/
int assertMergeAllValuesPresent(int *tab, int n1, int *tab2, int n2, int* m, int size)
{
  int verif[size]; //tableau avec des 1 là où l'on a déjà vérifié qu'il correspond déjà à un élément de a ou de b
  for(int i = 0;i<size;i++){
      verif[i] = 0;
  }

  for (int i=0; i<size; i++){
    for(int j = 0;j < n1;j++){
      if(tab[j] == m[i] && verif[i] == 0){ //si il y a une valeur identique et que celle-ci n'a pas été vérifiée
          verif[i] = 1;
      }
    }
  }
  for (int i=0; i<size; i++){
    for(int j = 0;j < n2;j++){
      if(tab2[j] == m[i] && verif[i] == 0){
          verif[i] = 1;
      }
    }
  }

  for(int i = 0;i<size;i++){
    if(verif[i] != 1){
        printf("\nWARNING : Unsuccessful merge : incorrect elements...\n");
        return 0;
    }
  }

  return 1;
}


/**
    verify that m is sortes and that it merges tab and tab2
    @param tab et tab2 : les deux tableaux qu'on veut fusionner
            m : le tableau qui est la fusion triée de tab et tab2

    @return : 0 any error, either 1
*/
int assertMerge(int *tab, int n1, int *tab2, int n2, int* m, int size){
    int successfulOrder = assertOrder(m, size);
    int successfulElements = assertMergeAllValuesPresent(tab, n1, tab2, n2, m, size);
    //assertMergeAllValuesPresent(int *tab, int n1, int *tab2, int n2, int* m, int size)
    if(successfulOrder && successfulElements){
        // printf("\nSuccessful merge !\n");
        return 1;
    }
    else{
        // printf("\nUnsuccessful merge !\n");
        return 0;
    }
}


/**
    verify that a list of arrays {M_i}1<=i<=N have correctly merge {A_i} and {B_i}
    @param A : First list of arrays to merge
           B : Second list of arrays to merge
           M : Merge of A and B
           sizes : sizes of subarrays A_i and B_i
           N : number of subarrays in A and B
    @return : 0 any error, either 1
*/
int assertMerge2D(int **A, int **B, int **M, int2 *sizes, int N)
{
  for (int i = 0; i < N; i++)
  {
    if (assertMerge(A[i], sizes[i].x, B[i], sizes[i].y, M[i], sizes[i].x + sizes[i].y) == 0)
    {
      printf("Unsuccessful merge on array %d !\n", i);
      return 0;
    }
  }
  printf("Successful merge !\n");
  return 1;
}


/**
    Merge 2 lists of arrays {A_i} and {B_i} in {M_i}1<=i<=N
    @param argv[1] : number of subarrays in A and B
           argv[2] : size of |A_i| + |B_i|
*/
int main(int argc, char *argv[])
{
    std::clock_t startS, endS;
    float seqMergeTime, parMergeTime, DoH, HoD;

    srand(time(NULL));
    int **A, **B, **unFlatM, **M;
    int *flatM, *flatA, *flatB, *aGPU, *bGPU, *mGPU;
    int N, d;
    int2 *sizes, *indices, *sizesGPU, *indicesGPU;
    int2 totalSizes;

    if(argc== 3)
    {
      N = atoi(argv[1]);
      d = atoi(argv[2]);
    }
    else
    {
      N= 1000;
      d = 4;
    }



    A = (int**)malloc(N*sizeof(int*));
    B = (int**)malloc(N*sizeof(int*));
    M = (int**)malloc(N*sizeof(int*));
    for (int i = 0; i < N; i++)
    {
      M[i] = (int*)malloc(d*sizeof(int));
    }
    sizes = (int2*)malloc(N*sizeof(int2)); // Sizes of each subarrays in A and B
    indices = (int2*)malloc(N*sizeof(int2)); // First indices of each subarrays in flat version of A and B
    init_list_array(A, B, N, d, sizes);
    indices_array(indices, sizes, N);
    totalSizes = get_totalSizes(sizes, N);

    //Flat versions of A, B and M to give to devices
    flatA = (int*)malloc(totalSizes.x*sizeof(int));
    flatB = (int*)malloc(totalSizes.y*sizeof(int));
    flatM = (int*)malloc(N*d*sizeof(int));
    flat_arrays(flatA, flatB, A, B, N, d, sizes); //Falt A and B to give them to devices

    //Allocation of devies objects
    gpuCheck(cudaMalloc(&mGPU, N*d*sizeof(int)));
    gpuCheck(cudaMalloc(&aGPU, totalSizes.x*sizeof(int)));
    gpuCheck(cudaMalloc(&bGPU, totalSizes.y*sizeof(int)));
    gpuCheck(cudaMalloc(&sizesGPU, N*sizeof(int2)));
    gpuCheck(cudaMalloc(&indicesGPU, N*sizeof(int2)));

    //Copy host to device of arrays to merge, with their respective sizes and indices
    startS = std::clock();
    gpuCheck( cudaMemcpy(aGPU, flatA, totalSizes.x*sizeof(int), cudaMemcpyHostToDevice) );
    gpuCheck( cudaMemcpy(bGPU, flatB, totalSizes.y*sizeof(int), cudaMemcpyHostToDevice) );
    gpuCheck( cudaMemcpy(sizesGPU, sizes, N*sizeof(int2), cudaMemcpyHostToDevice) );
    gpuCheck( cudaMemcpy(indicesGPU, indices, N*sizeof(int2), cudaMemcpyHostToDevice) );
    endS = std::clock();
    HoD = (endS - startS) / (float) CLOCKS_PER_SEC;

    //Merge of batches parallel
    printf("======== batch merge =======\n");
    printf("* Number of batches : %d\n* Sizes of batches : %d\n", N, d);
    startS = std::clock();
    mergeSmallBatch_k<<<N*d/1024+1, d*1024/d>>>(aGPU, bGPU, mGPU, d, N, indicesGPU, sizesGPU);
    gpuCheck(cudaPeekAtLastError());
    gpuCheck( cudaDeviceSynchronize() );
    endS = std::clock();
    parMergeTime = (endS - startS) / (float) CLOCKS_PER_SEC;
    startS = std::clock();
    gpuCheck( cudaMemcpy(flatM, mGPU, N*d*sizeof(int), cudaMemcpyDeviceToHost) );
    endS = std::clock();
    DoH = (endS - startS) / (float) CLOCKS_PER_SEC;

    //Merge of batches parallel
    startS = std::clock();
    mergeSeqBatch(A, B, M, N, d, sizes);
    endS = std::clock();
    seqMergeTime = (endS - startS) / (float) CLOCKS_PER_SEC;

    //Allocation unFlat version of M
    unFlatM = (int**)malloc(N*sizeof(int*));
    for (int i = 0; i < N; i++)
    {
      unFlatM[i] = (int*)malloc(d*sizeof(int));
    }

    //unflat M
    unflat_arrays(unFlatM, flatM, N, d);

    //Verify that M is correctly merge


    printf("\n========= Sequential merge : =============\n");
    printf("Total time elapsed : %f s\n", seqMergeTime);
    assertMerge2D(A, B, M, sizes, N);
    printf("\n");

    printf("========= Parallel merge : =============\n");
    printf("Total time elapsed : %f s\n", parMergeTime+DoH+HoD);
    printf("Time running algorithm : %f s\n", parMergeTime);
    printf("Time to copy Host to Device : %f s\n", HoD);
    printf("Time to copy Device to Host : %f s\n", DoH);
    assertMerge2D(A, B, unFlatM, sizes, N);
    printf("Parrallel algorithm is %f times faster than sequential merge !\n", seqMergeTime/parMergeTime);
    printf("Parrallel merge is %f times faster than sequential merge !\n", seqMergeTime/(parMergeTime+HoD+DoH));



    //Free device mermory
    gpuCheck(cudaFree(aGPU));
    gpuCheck(cudaFree(bGPU));
    gpuCheck(cudaFree(mGPU));
    gpuCheck(cudaFree(indicesGPU));
    gpuCheck(cudaFree(sizesGPU));

    //Free host memory
    free(A);
    free(B);
    free(unFlatM);
    free(flatM);
    free(flatA);
    free(flatB);
    free(sizes);
    free(indices);


    return 0;
}
