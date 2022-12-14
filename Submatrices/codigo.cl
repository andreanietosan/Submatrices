typedef struct
{
	int x, // Coordenada x de la submatriz
		y, // Coordenada y de la submatriz
		t; // Tamaño de la submatriz
} terna_t;

__kernel void submatriz( int N, __global double *vecA, __global terna_t *ternas, __global double *cA)
{
    size_t id = get_global_id(0);

    //desplazamiento de vecA que le correcponde a cada wi
    __global double *A = &vecA[N*N*id];

    for (int i = 0; i < ternas[id].t; i++)
		for (int j = 0; j < ternas[id].t; j++){
            A[((ternas[id].y + i) % N) * N + (ternas[id].x + j) % N] = 0;
			for (int a = 0; a < ternas[id].t; a++)
				A[((ternas[id].y + i) % N) * N + (ternas[id].x + j) % N] += cA[((ternas[id].y + i) % N) * N + (ternas[id].x + a) % N] * cA[((ternas[id].y + a) % N) * N + (ternas[id].x + j) % N];
        }
}