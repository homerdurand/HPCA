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

//Globall version of parallel merge of a and b in m with |m|<1024
__global__ void mergeSmall_k(int* d_a, int* d_b, int* d_m, int n_a, int n_b, int n_m){
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

//Device version of parallel merge of a and b in m with |m|<1024
__device__ void mergeParallel(int* d_a, int* d_b, int* d_m, int n_a, int n_b, int n_m){
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
    mergeParallel(a_k, b_k, m_k, dimx, dimy, dimx+dimy);
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

//Function that decide wether to use mergeSmall_k or mergeBig_k giving the size of the pieces
void mergeAll(int* m, int n_m, int* a, int n_a, int* b, int n_b){
    int pas = 1024; // <1024
    int nbPartitions = n_m/pas+(n_m%pas!=0); // On ajoute 1 si n_m n'est pas un mutliple de p
    int n_path = (1 + nbPartitions); //1(pour (0,0)) + |m|/pas(nbr de morceau de taille pas) + 1(si dernier morceau de taille < pas))

    // printf("n_m : %d\nn_a : %d\nn_b : %d\n", n_m, n_a, n_b);
    int *mGPU, *aGPU, *bGPU;
    gpuCheck(cudaMalloc(&mGPU, n_m*sizeof(int)));
    gpuCheck(cudaMalloc(&aGPU, n_a*sizeof(int)));
    gpuCheck(cudaMalloc(&bGPU, n_b*sizeof(int)));

    //Declaration et allocation de path
    int2 *pathGPU;
    gpuCheck(cudaMalloc(&pathGPU, n_path*sizeof(int2)));

    gpuCheck(cudaMemcpy(aGPU, a, n_a*sizeof(int), cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(bGPU, b, n_b*sizeof(int), cudaMemcpyHostToDevice));

    // printf("n_12 : %d, n_1 : %d, n_2 : %d\n", n_m, n_a, n_b);
    if(n_m <= 1024){
        // printf("case mergeSmall\n");
        // printf("n_m = %d, n_a = %d, n_b = %d\n", n_a, n_b, n_m);
        mergeSmall_k<<<1, 1024>>>(aGPU, bGPU, mGPU, n_a, n_b, n_m);
        gpuCheck( cudaPeekAtLastError() );
        //gpuCheck(cudaDeviceSynchronize());
        gpuCheck(cudaMemcpy(m, mGPU, n_m*sizeof(int), cudaMemcpyDeviceToHost));
    }
    else{
        //================ Parallel : =======================\\
        // printf("\n");
        // printf("case mergeBig\n");
        pathBig_k<<<nbPartitions/1024+1, 1024>>>(pas, pathGPU, n_path, aGPU, n_a, bGPU, n_b);
        gpuCheck( cudaPeekAtLastError() );
        gpuCheck( cudaDeviceSynchronize() );
        mergeBig_k<<<nbPartitions, pas>>>(mGPU, n_m, aGPU, n_a, bGPU, n_b, pathGPU, n_path, nbPartitions);
        gpuCheck( cudaDeviceSynchronize() );

        gpuCheck(cudaMemcpy(m, mGPU, n_m*sizeof(int), cudaMemcpyDeviceToHost));

        gpuCheck( cudaDeviceSynchronize() );
    }
    gpuCheck(cudaFree(aGPU));
    gpuCheck(cudaFree(bGPU));
    gpuCheck(cudaFree(pathGPU));
}

//Fonction de prétraitement qui trie chaque paire contigüe d'éléments d'un tableau m
__global__ void pretraitementFusionSort(int* m, int* mGPU, int n){
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


//Function that sort any array
void fusionSort(int *m, int n_m)
{
  //L1 : indice du premier élément de m_part1
  //R1 : indice du dernier élément de m_part1
  //L2 : indice du premier élément de m_part2
  //R2 : indice du dernier élément de m_part2
  int size = 1;
  int L1, R1, L2, R2;
  int i;

  while (size <= n_m)
  {
    i = 0;
    while(i < n_m){
      // printf("m : ");
      // print_array(m, n_m);
      L1 = i;
      R1 = i + size-1;
      L2 = i + size;
      R2 = i + 2*size-1;
      if(L2 >= n_m){
          break;
      }
      if(R2 >= n_m){
          R2 = n_m-1;
      }
      int size_1 = R1-L1+1;
      int size_2 = R2-L2+1;
      int size_12 = size_1+size_2;
      int *m_12, *m_1, *m_2;
      m_12 = (int*)malloc(size_12*sizeof(int));
      m_1 = (int*)malloc(size_1*sizeof(int));
      m_2 = (int*)malloc(size_2*sizeof(int));

      //Remplir les tableaux m_1 et m_2
      for (int j = 0; j < size_1; j++)
      {
        m_1[j] = m[L1 + j];
      }
      for (int j = 0; j < size_2; j++)
      {
        m_2[j] = m[L2 + j];
      }

      mergeAll(m_12, size_12, m_1, size_1, m_2, size_2);

      for (int j = 0; j < size_12; j++)
      {
        m[L1+j] = m_12[j];
      }
      free(m_12);
      free(m_1);
      free(m_2);
      i = i + 2*size;
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
    int n_m = 100000;
    int *m, *mseq, *mref;

    if(argc==2)
    {
      n_m = atoi(argv[1]);
    }

    printf("========== Path Sort : =========\n");
    printf("* Size of array : %d\n\n", n_m);
    //int* mseq;
    m = (int*)malloc(n_m*sizeof(int));
    init_array_no_order(m, n_m, n_m*10);
    mseq = (int*)malloc(n_m*sizeof(int)); //copie de m
    copy_array(m, mseq, n_m);
    mref = (int*)malloc(n_m*sizeof(int)); //copie de m
    copy_array(m, mref, n_m);
    //Partie des calculs1024
    //================ Paral1024lel : =======================\\
    //Etape de prétraitement :
    startS = std::clock();
    fusionSort(m, n_m);
    gpuCheck(cudaDeviceSynchronize());
    endS = std::clock();
    parMergeTime = (endS - startS) / (float) CLOCKS_PER_SEC;
    //Etape du tri fusion :

    startS = std::clock();
    fusionSortSeq(mseq, n_m);
    endS = std::clock();
    seqMergeTime = (endS - startS) / (float) CLOCKS_PER_SEC;

    printf("========= Parallel sort : =============\n");
    printf("Total time elapsed : %f s\n", parMergeTime);
    // assertSorted(mref, m, n_m);
    printf("Parrallel algorithm is %f times faster than sequential merge !\n", seqMergeTime/parMergeTime);
    printf("Parrallel merge is %f times faster than sequential merge !\n", seqMergeTime/parMergeTime);

    printf("========= Sequential sort : =============\n");
    printf("Total time elapsed : %f s\n", seqMergeTime);
    // assertSorted(mref, mseq, n_m);




    return 0;
}
