#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#include <bits/stdc++.h>
using namespace std;
#include <iostream>



//Function that verify cuda calls and return cuda error if any

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
 * Intialize ascendant array
 */
void init_array(int* array, int size, int const adder=10)
{
    array[0] = rand()%adder;
    for(int i = 0; i < size;i++)
    {
        array[i] = array[i-1] + rand()%adder;
    }
    printf("\n");
}

//Function that initialise array with random values
void init_array_no_order(int* array, int size, int const adder=10)
{
    array[0] = rand()%adder;
    for(int i = 0; i < size;i++)
    {
        array[i] = rand()%adder;
    }
    printf("\n");
}

//Function that copy array in another
void copy_array(int* a, int* a_copy, int n){
    for(int i = 0;i < n;i++){
        a_copy[i] = a[i];
    }
}

//Function that print an array of size size
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

//Globall version of parallel merge of a and b in m with |m|<1024
__global__ void mergeSmallBatch_k(int* aGPU, int* bGPU, int* mGPU, int d, int nb_batch, int2 *indices, int2 *sizes)
{
    int tidx = threadIdx.x%d;
    int Qt = (threadIdx.x-tidx)/d;
    int gbx = Qt + blockIdx.x*(blockDim.x/d);
    int i = gbx*d + tidx;
    if(i < nb_batch*d)
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
                      // printf("mGPU[%d] = %d | %d \n", i, mGPU[i], indices[gbx].x + Q.y);
                    }
                    else
                    {
                        mGPU[i] = bGPU[indices[gbx].y + Q.x];
                      // printf("mGPU[%d] = %d | %d\n", i, mGPU[i], indices[gbx].y + Q.x);
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


//Fonction de prétraitement qui trie chaque paire contigüe d'éléments d'un tableau m
__global__ void pretraitementFusionSort(int* mGPU, int n){
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int i = blockId * blockDim.x + threadId;
    int tmp;
    if(i < n/2)
    {
        int indice = 2*i;
        if(mGPU[indice] > mGPU[indice+1])
        {
            tmp = mGPU[indice];
            mGPU[indice] = mGPU[indice+1];
            mGPU[indice+1] = tmp;
        }
    }
}

__global__ void arrangeBatch(int *A, int *B, int *m, int2 *sizes, int2 *indices, int n_m, int nb_batch, int d)
{
  int tidx = threadIdx.x%d;
  int Qt = (threadIdx.x-tidx)/d;
  int gbx = Qt + blockIdx.x*(blockDim.x/d);
  int i = gbx*d + tidx;
  if (i < n_m)
  {
    if (tidx < d/2)
    {
      A[gbx*d/2 + tidx] = m[i];
    }
    else
    {
      B[gbx*d/2 + tidx - d/2] = m[i];
    }
    if (tidx == 0)
    {
      indices[i/d].x = i/2;
      indices[i/d].y = i/2;
      sizes[i/d].x = d/2;
      sizes[i/d].y = d/2;
    }
  }
}

__global__ void truncate(int *mTrunc, int *m, int n_m, int k, int nb_batch, int d)
{
  int tidx = threadIdx.x%d;
  int Qt = (threadIdx.x-tidx)/d;
  int gbx = Qt + blockIdx.x*(blockDim.x/d);
  int i = gbx*d + tidx;
  if(i < nb_batch*d)
  {
    // printf("i : %d\n", i);
    if (tidx < k)
    {
      mTrunc[gbx*k + tidx] = m[i];
    }
  }
}

//Function that sort any array
void batchMerge_k(int *mGPU, int n_m, int k)
{
  int *M, *aGPU, *bGPU, *mTrunc;
  int2 *sizesGPU, *indicesGPU;
  int nb_batch, d;
  d = 4;
  nb_batch = n_m/d;

  M = (int*)malloc(n_m*sizeof(int));
  gpuCheck(cudaMalloc(&aGPU, n_m/2*sizeof(int)));
  gpuCheck(cudaMalloc(&bGPU, n_m/2*sizeof(int)));
  gpuCheck(cudaMalloc(&sizesGPU, nb_batch*sizeof(int2)));
  gpuCheck(cudaMalloc(&indicesGPU, nb_batch*sizeof(int2)));


  while(nb_batch >= 1)
  {
    arrangeBatch<<<nb_batch, d>>>(aGPU, bGPU, mGPU, sizesGPU, indicesGPU, n_m, nb_batch, d);
    gpuCheck( cudaPeekAtLastError() );
    gpuCheck( cudaDeviceSynchronize() );

    mergeSmallBatch_k<<<nb_batch, d>>>(aGPU, bGPU, mGPU, d, nb_batch, indicesGPU, sizesGPU);
    gpuCheck( cudaPeekAtLastError() );
    gpuCheck( cudaDeviceSynchronize() );

    if (d > k)
    {
      gpuCheck(cudaMalloc(&mTrunc, k*nb_batch*sizeof(int)));
      truncate<<<nb_batch, d>>>(mTrunc, mGPU, n_m, k, nb_batch, d);
      gpuCheck( cudaPeekAtLastError() );
      gpuCheck( cudaDeviceSynchronize() );
      gpuCheck(cudaMalloc(&mGPU, k*nb_batch*sizeof(int)));
      mGPU = mTrunc;
      n_m = k*nb_batch;
      d = k;
      gpuCheck(cudaMalloc(&mTrunc, n_m/2*sizeof(int)));
      gpuCheck(cudaMalloc(&aGPU, n_m/2*sizeof(int)));
      gpuCheck(cudaMalloc(&bGPU, n_m/2*sizeof(int)));
      gpuCheck(cudaMalloc(&sizesGPU, nb_batch*sizeof(int2)));
      gpuCheck(cudaMalloc(&indicesGPU, nb_batch*sizeof(int2)));

    }
    // print_array(M, n_m);
    nb_batch = nb_batch/2;
    d *= 2;
  }
  gpuCheck( cudaMemcpy(M, mGPU, n_m*sizeof(int), cudaMemcpyDeviceToHost) );
  print_array(M, n_m);
}

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


// Function to swap position of elements
void swap(int *a, int *b) {
  int t = *a;
  *a = *b;
  *b = t;
}

// Function to print eklements of an array
void printArray(int array[], int size) {
  int i;
  for (i = 0; i < size; i++)
    cout << array[i] << " ";
  cout << endl;
}

// Function to partition the array on the basis of pivot element
int partition(int array[], int low, int high) {
  // Select the pivot element
  int pivot = array[high];
  int i = (low - 1);

  // Put the elements smaller than pivot on the left
  // and greater than pivot on the right of pivot
  for (int j = low; j < high; j++) {
    if (array[j] <= pivot) {
      i++;
      swap(&array[i], &array[j]);
    }
  }
  swap(&array[i + 1], &array[high]);
  return (i + 1);
}

void quickSort(int array[], int low, int high) {
  if (low < high) {
    // Select pivot position and put all the elements smaller
    // than pivot on left and greater than pivot on right
    int pi = partition(array, low, high);

    // Sort the elements on the left of pivot
    quickSort(array, low, pi - 1);

    // Sort the elements on the right of pivot
    quickSort(array, pi + 1, high);
  }
}

void knnSeq(int knn[], int *m, int n_m, int k)
{
  for (int i = 0; i < k; i++)
  {
    knn[i] = m[i];
  }


  quickSort(knn, 0, k-1);
  for (int i = k; i < n_m; i++)
  {
    if (knn[k-1] > m[i])
    {
      knn[k-1] = m[i];
      quickSort(knn, 0, k-1);
    }
  }
}

int assertPretraitement(int *tab, int size)
{
  if(size % 2 == 1)
  {
      size -= 1;
  }
  for (int i=0; i<size/2; i++)
  {
    if (tab[2*i] > tab[2*i+1])
    {
      printf("WARNING : Unsuccessful pretreatment ... : unordered paired array on indice %d ...\n", i);
      printf("tab[i]= %d > tab[i+1] = %d\n", tab[i], tab[i+1]);
      return 0;
    }
  }
  printf("\nSuccessful pretreatment !\n");
  return 1;
}

//Fonction qui trie un tableau M en parallèle par tri fusion itératif (question 3)

//Fonctions de vérification

//Fonction qui vérifie qu'un tableau est bien trié (tous ses éléments rangés dans l'ordre croissant)
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

//Fonction qui vérifie qu'on retrouve bien dans le nouveau tableau tous les éléments des deux tableaux qu'on veut fusionner
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

//Fonction qui vérifie qu'on retrouve bien dans le nouveau tableau tous les éléments du tableau qu'on veut trier
int assertSortAllValuesPresent(int* m, int* m_sorted, int size){
  int verif[size]; //tableau avec des 1 là où l'on a déjà vérifié qu'il correspond déjà à un élément de a ou de b
  for(int i = 0;i<size;i++){
      verif[i] = 0;
  }

  for (int i=0; i<size; i++){
    for(int j = 0;j < size;j++){
      if(m_sorted[j] == m[i]){ //si il y a une valeur identique
          verif[i] = 1;
      }
    }
  }

  for(int i = 0;i<size;i++){
    if(verif[i] != 1){
        printf("i : %d\n", i);
        printf("\nWARNING : Unsuccessful sort : incorrect elements...\n");
        return 0;
    }
  }

  return 1;
}

//Fonction qui vérifie qu'un tableau est bien trié et la fusion de deux tableaux
//tab et tab2 : les deux tableaux qu'on veut fusionner
//m : le tableau qui est la fusion triée de tab et tab2
int assertMerge(int *tab, int n1, int *tab2, int n2, int* m, int size){
    int successfulOrder = assertOrder(m, size);
    int successfulElements = assertMergeAllValuesPresent(tab, n1, tab2, n2, m, size);
    //assertMergeAllValuesPresent(int *tab, int n1, int *tab2, int n2, int* m, int size)
    if(successfulOrder && successfulElements){
        printf("\nSuccessful merge !\n");
        return 1;
    }
    else{
        printf("\nUnsuccessful merge !\n");
        return 0;
    }
}

//Fonction qui vérifie qu'un tableau est bien trié
//m : le tableau non trié qu'on veut trier
//m_sorted : le tableau m soi-disant trié (on veut vérifier si c'est bien le cas)
//size : la taille du tableau
int assertSorted(int* m, int* m_sorted, int size)
{
    int successfulOrder = assertOrder(m_sorted, size); // les éléments du tableau sont ils bien dans le bon ordre ?
    int successfulElements = assertSortAllValuesPresent(m, m_sorted, size); //retrouve t-on bien toutes les valeurs ?
    if(successfulOrder && successfulElements){
        printf("\nSuccessful sort !\n");
        return 1;
    }
    else{
        printf("\nUnsuccessful sort !\n");
        return 0;
    }
}

int main(int argc, char *argv[])
{
    std::clock_t startS, endS;
    float seqMergeTime, parMergeTime, DoH, HoD;

    srand(time(NULL));
    int n_m = pow(2, 20);
    int pas = 8; // 1024<1024
    int k = 8;

    if(argc== 3)
    {
      k = atoi(argv[1]);
      n_m = atoi(argv[2]);
    }
    int nbPartitions = n_m/pas+(n_m%pas!=0);
    int *m, *mGPU;
    int *knn = (int*)malloc(k*sizeof(int));
    m = (int*)malloc(n_m*sizeof(int));
    init_array_no_order(m, n_m, n_m);
    gpuCheck(cudaMalloc(&mGPU, n_m*sizeof(int)));

    startS = std::clock();
    gpuCheck(cudaMemcpy(mGPU, m, n_m*sizeof(int), cudaMemcpyHostToDevice));
    endS = std::clock();
    HoD = (endS - startS) / (float) CLOCKS_PER_SEC;

    printf("======== Parallel search of KNN =======\n");
    printf("* K : %d\n* Number of features : %d\n", k, n_m);

    //================ Parallel : =======================\\

    //Etape de prétraitement :
    startS = std::clock();
    pretraitementFusionSort<<<nbPartitions, pas>>>(mGPU, n_m);
    gpuCheck( cudaPeekAtLastError() );
    gpuCheck( cudaDeviceSynchronize() );

    //Sort array
    printf("========= Parallel merge : =============\n");
    printf("* K-Nearest Neighbors :");
    batchMerge_k(mGPU, n_m, k);
    gpuCheck( cudaDeviceSynchronize() );
    endS = std::clock();
    parMergeTime = (endS - startS) / (float) CLOCKS_PER_SEC;

    startS = std::clock();
    knnSeq(knn, m, n_m, k);
    endS = std::clock();
    seqMergeTime = (endS - startS) / (float) CLOCKS_PER_SEC;

    // startS = std::clock();
    // // gpuCheck( cudaMemcpy(knn, mGPU, k*sizeof(int), cudaMemcpyDeviceToHost) );
    // endS = std::clock();
    // DoH = (endS - startS) / (float) CLOCKS_PER_SEC;


    printf("Total time elapsed : %f s\n", parMergeTime+DoH+HoD);
    printf("Time running algorithm : %f s\n", parMergeTime);
    printf("Time to copy Host to Device : %f s\n", HoD);
    // printf("Time to copy Device to Host : %f s\n", DoH);
    printf("Parrallel algorithm is %f times faster than sequential merge !\n", seqMergeTime/parMergeTime);
    printf("Parrallel knn finding is %f times faster than sequential merge !\n\n", seqMergeTime/(parMergeTime+HoD));

    printf("========= Sequential merge : =============\n");
    printf("* K-Nearest Neighbors :");
    // print_array(knn, k);
    printf("Total time elapsed : %f s\n", seqMergeTime);



    // print_array(knn, k);


    return 0;
}
