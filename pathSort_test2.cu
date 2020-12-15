#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>


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

//Initialise ascendant array with random values in
void init_array(int* array, int size, int const adder=10)
{
    array[0] = rand()%adder;
    for(int i = 0; i < size;i++)
    {
        array[i] = array[i-1] + rand()%adder;
    }
}

//Function that initialise array with random values
void init_array_no_order(int* array, int size, int const adder=10)
{
    array[0] = rand()%adder;
    for(int i = 0; i < size;i++)
    {
        array[i] = rand()%adder;
    }
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

//Device version of parallel merge of a and b in m with |m|<1024
__global__ void mergeSmall_k(int* m, int n, int size)
{
    int gbx = blockIdx.x;
    int tidx = threadIdx.x;
    int i = gbx * blockDim.x + tidx;
    if(i < n)
    {
        int L1, R1, L2, R2;
        L1 = gbx*blockDim.x;
        R1 = gbx*blockDim.x + size-1;
        L2 = gbx*blockDim.x + size;
        R2 = gbx*blockDim.x + 2*size-1;
        if(L2 < n)
        {
          // printf("L1 : %d, R1 : %d, L2 : %d, R2 : %d\n", L1, R1, L2, R2);
          if(R2 >= n){
              R2 = n-1;
          }
          __shared__ int *d_a, *d_b;
          int n_a = R1-L1+1;
          int n_b = R2-L2+1;
          int n_m = n_a+n_b;
          d_a = (int*)malloc(n_a*sizeof(int));
          d_b = (int*)malloc(n_b*sizeof(int));
          __syncthreads();
          // printf("tidx : %d, n_a : %d\n", tidx, n_a);
          if (tidx < n_a)
          {
            // printf("m[%d] : %d\n", i, m[i]);
            d_a[tidx] = m[i];
            // printf("d_a_%d[%d] = %d\n", gbx, tidx, d_a[tidx]);
          }
          else if (tidx < n_m)
          {
            d_b[tidx - n_a] = m[i];
            // printf("d_b_%d[%d] = %d\n", gbx, tidx - n_a, d_b[tidx - n_a]);
          }
          __syncthreads();

          int2 K;
          int2 P;
          int2 Q;
          // printf("n_a : %d, n_b : %d\n", n_a, n_b);
          if(tidx > n_a)
          {
              K.x = tidx - n_a;
              K.y = n_a;
              P.x = n_a;
              P.y = tidx - n_a;
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
              if(Q.y >= 0 && Q.x <= n_b && (Q.y == n_a || Q.x == 0 || d_a[Q.y] > d_b[Q.x - 1]))
              {
                  if(Q.x == n_b || Q.y == 0 || d_a[Q.y - 1] <= d_b[Q.x])
                  {
                      if(Q.y < n_a && (Q.x == n_b || d_a[Q.y] <= d_b[Q.x]))
                      {
                          m[i] = d_a[Q.y];
                          // printf("## m[%d] : %d, d_a_%d[%d] : %d\n",i, m[i], gbx, Q.y, d_a[Q.y]);
                      }
                      else
                      {
                          m[i] = d_b[Q.x];
                          // printf("## m[%d] : %d, d_b_%d[%d] : %d\n",i, m[i], gbx, Q.x, d_b[Q.x]);
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

          __syncthreads();


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


//Giving a path ( from pathBig_k ) each block merge (with mergeParallel) each piece a_k and b_k in m_k of a and b. Then it replace elements in m
__global__ void mergeBig_k(int *m, int n_m, int *a, int n_a, int *b, int n_b, int2 *path, int n_path, int nbPartitions, int size)
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
    // mergeParallel(m_k, dimx+dimy, size);
    m[i] = m_k[threadId];

  }
}

//Function that generate a path to break down m into pieces that could be merge without conflict
//On appelle |m|/TPB blocks avec chacun un seul thread. Chaque thread s'occupe de la diagonale thread
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

//Function that sort any array
void fusionSort(int *mGPU, int n_m)
{
  //L1 : indice du premier élément de m_part1
  //R1 : indice du dernier élément de m_part1
  //L2 : indice du premier élément de m_part2
  //R2 : indice du dernier élément de m_part2
  int size = 1;
  int i;
  int *m = (int*)malloc(n_m*sizeof(int));

  while (size < n_m)
  {
    i = 0;
    if (size < 1024)
    {
      printf("Size : %d\n", size);
      mergeSmall_k<<<n_m/(2*size) + 1, 2*size>>>(mGPU, n_m, size);
      gpuCheck(cudaMemcpy(m, mGPU, n_m*sizeof(int), cudaMemcpyDeviceToHost));
      cudaDeviceSynchronize();
      print_array(m, n_m);
    }
    size *= 2;
  }
}


void fusionMergeSeq(int* A, int* tmp, int L1, int R1, int L2, int R2){
    int i = 0;
    while(L1 <= R1 && L2 <= R2){
        if(A[L1] <= A[L2]){
            tmp[i] = A[L1];
            i++;
            L1++;
        }
        else{
            tmp[i] = A[L2];
            i++;
            L2++;
        }
    }
    while(L1 <= R1){
        tmp[i] = A[L1];
        i++;
        L1++;
    }
    while(L2 <= R2){
        tmp[i] = A[L2];
        i++;
        L2++;
    }

}

void fusionSortSeq(int* A, int n){
    int len = 1;
    int i;
    int L1, R1, L2, R2;
    int* tmp = (int*)malloc(n*sizeof(int));
    while(len < n){
        i = 0;
        while(i < n){
            L1 = i;
            R1 = i + len - 1;
            L2 = i + len;
            R2 = i + 2*len - 1;
            tmp = (int*)realloc(tmp, (R2-L1+1)*sizeof(int));
            if(L2 >= n){
                break;
            }
            if(R2 >= n){
                R2 = n - 1;
            }
            fusionMergeSeq(A, tmp, L1, R1, L2, R2);
            for(int j = 0;j < R2-L1+1;j++){
                A[i+j] = tmp[j];
            }
            i = i + 2*len;
        }
        len *= 2;
    }
    free(tmp);
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
    float seqMergeTime, parMergeTime;

    srand(time(NULL));
    int n_m = 200;
    int *m, *mseq, *mref, *mGPU;

    if(argc==2)
    {
      n_m = atoi(argv[1]);
    }

    printf("========== Path Sort : =========\n");
    printf("* Size of array : %d\n\n", n_m);
    //int* mseq;
    m = (int*)malloc(n_m*sizeof(int));
    init_array_no_order(m, n_m, n_m*10);
    gpuCheck(cudaMalloc(&mGPU, n_m*sizeof(int)));
    gpuCheck(cudaMemcpy(mGPU, m, n_m*sizeof(int), cudaMemcpyHostToDevice));

    print_array(m, n_m);

    mseq = (int*)malloc(n_m*sizeof(int)); //copie de m
    copy_array(m, mseq, n_m);
    mref = (int*)malloc(n_m*sizeof(int)); //copie de m
    copy_array(m, mref, n_m);
    //Partie des calculs1024
    //================ Paral1024lel : =======================\\
    //Etape de prétraitement :
    startS = std::clock();
    fusionSort(mGPU, n_m);
    cudaDeviceSynchronize();
    endS = std::clock();
    parMergeTime = (endS - startS) / (float) CLOCKS_PER_SEC;
    gpuCheck(cudaMemcpy(m, mGPU, n_m*sizeof(int), cudaMemcpyDeviceToHost));

    //Etape du tri fusion :

    startS = std::clock();
    fusionSortSeq(mseq, n_m);
    endS = std::clock();
    seqMergeTime = (endS - startS) / (float) CLOCKS_PER_SEC;

    printf("========= Parallel sort : =============\n");
    printf("Total time elapsed : %f s\n", parMergeTime);
    assertSorted(mref, m, n_m);
    printf("Parrallel algorithm is %f times faster than sequential merge !\n", seqMergeTime/parMergeTime);
    printf("Parrallel merge is %f times faster than sequential merge !\n", seqMergeTime/parMergeTime);

    printf("========= Sequential sort : =============\n");
    printf("Total time elapsed : %f s\n", seqMergeTime);
    // assertSorted(mref, mseq, n_m);




    return 0;
}
