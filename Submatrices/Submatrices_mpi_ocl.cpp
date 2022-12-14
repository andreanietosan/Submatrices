#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <string.h>
#include <CL/cl.h>
#include <mpi.h>

#include "funciones.h"

typedef struct
{
	int x, // Coordenada x de la submatriz
		y, // Coordenada y de la submatriz
		t; // Tama�o de la submatriz
} terna_t;

void initializedouble(int t, double *a, double lv, double uv)
{
	int i;

	for (i = 0; i < t; i++)
		// Valores generados entre lv y uv con 2 decimales
		a[i] = ((int)((((1. * rand()) / RAND_MAX) * (uv - lv) + lv) * 100.)) / 100.;
}

void initialize(int t, double *a, terna_t *ternas, int r)
{
	int i;

	initializedouble(t * t, a, -10., 10.);

	for (i = 0; i < r; i++)
	{
		ternas[i].x = (int)(((1. * rand()) / RAND_MAX) * t);
		ternas[i].y = (int)(((1. * rand()) / RAND_MAX) * t);
		ternas[i].t = (int)(((1. * rand()) / RAND_MAX) * (t - 2) + 2);
	}
}

void escribir(int t, double *a)
{
	int i, j;

	for (i = 0; i < t; i++)
	{
		for (j = 0; j < t; j++)
		{
			printf("%.2f ", a[i * t + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void escribirt(terna_t *a, int t)
{
	int i;

	for (i = 0; i < t; i++)
		printf("%d ", a[i].x);
	printf("\n");
	for (i = 0; i < t; i++)
		printf("%d ", a[i].y);
	printf("\n");
	for (i = 0; i < t; i++)
		printf("%d ", a[i].t);
	printf("\n");
}

/*
c
c     mseconds - returns elapsed milliseconds since Jan 1st, 1970.
c
*/
long long mseconds()
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec * 1000 + t.tv_usec / 1000;
}

int ObtenerParametros(int argc, char *argv[], int *debug, int *num_workitems, int *workitems_por_workgroups)
{
	int i;
	*debug = 0;
	*num_workitems = 0;
	*workitems_por_workgroups = 0;
	if (argc < 2)
		return 0;
	for (i = 2; i < argc;)
	{
		if (strcmp(argv[i], "-d") == 0)
		{
			*debug = 1;
			i++;
		}
		else if (strcmp(argv[i], "-wi") == 0)
		{
			i++;
			if (i == argc)
				return 0;
			*num_workitems = atoi(argv[i]);
			i++;
			if (*num_workitems <= 0)
				return 0;
		}
		else if (strcmp(argv[i], "-wi_wg") == 0)
		{
			i++;
			if (i == argc)
				return 0;
			*workitems_por_workgroups = atoi(argv[i]);
			i++;
			if (*workitems_por_workgroups <= 0)
				return 0;
		}
		else
			return 0;
	}
	return 1;
}

typedef struct
{
	cl_uint num_plataformas, num_dispositivos;
	cl_platform_id *plataformas;
	cl_device_id *dispositivos;
	cl_context contexto;
	cl_command_queue cola;
	cl_program programa;
	cl_kernel kernel; // Se pueden definir mas campos de tipo cl_kernel si es necesario

	cl_int error;
	cl_mem buffInTernas, buffInCA, buffOutVecA;
} EntornoOCL_t;

// **************************************************************************
// ***************************** IMPLEMENTACION *****************************
// **************************************************************************
cl_int InicializarEntornoOCL(EntornoOCL_t *entorno)
{
	ObtenerPlataformas(entorno->plataformas, entorno->num_plataformas);
	ObtenerDispositivos(entorno->plataformas[0], CL_DEVICE_TYPE_ALL, entorno->dispositivos, entorno->num_dispositivos);
	CrearContexto(entorno->plataformas[0], entorno->dispositivos, entorno->num_dispositivos, entorno->contexto);
	entorno->cola = clCreateCommandQueue(entorno->contexto, entorno->dispositivos[0], CL_QUEUE_PROFILING_ENABLE, &entorno->error);
	if (entorno->error != CL_SUCCESS)
	{
		printf("Error al crear la cola de comandos.");
		CodigoError(entorno->error);
		return entorno->error;
	}
	CrearPrograma(entorno->programa, entorno->contexto, entorno->num_dispositivos, entorno->dispositivos, " ", "codigo.cl");
	entorno->kernel = clCreateKernel(entorno->programa, "submatriz", &entorno->error);
	if (entorno->error != CL_SUCCESS)
	{
		printf("Error al crear el kernel. ");
		CodigoError(entorno->error);
		return entorno->error;
	}
}

cl_int LiberarEntornoOCL(EntornoOCL_t *entorno)
{
	clReleaseContext(entorno->contexto);
	clReleaseCommandQueue(entorno->cola);
	clReleaseProgram(entorno->programa);
	clReleaseKernel(entorno->kernel);
	clReleaseDevice(entorno->dispositivos[0]);
}

/*
N -> Tama�o de la matriz (NxN)
A -> Matriz
ternas -> Vector de ternas con los tama�os y las coordenadas de las submatrices
num_sb -> N�mero de submatrices
num_workitems -> N�mero de work items que se usar�n para lanzar el kernel. Es opcional, se puede usar o no dentro de la funci�n
workitems_por_workgroups -> N�mero de work items que se lanzar�n en cada work group. Es opcional, se puede usar o no dentro de la funci�n
*/
cl_int ocl(int N, double *A, terna_t *ternas, int num_sb, EntornoOCL_t *entorno, int num_workitems, int workitems_por_workgroups)
{
	int myrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	// vector de matrices para cada work items
	double *vecA = (double *)malloc(N * N * num_sb * sizeof(double));
	for (int i = 0; i < N * N * num_sb; i++)
	{
		vecA[i] = 0;
	}

	// creacion de buffers
	entorno->buffInTernas = clCreateBuffer(entorno->contexto, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(terna_t *), ternas, &entorno->error);
	if (entorno->error != CL_SUCCESS)
	{
		printf("Error al crear el primer buffer de entrada. ");
		CodigoError(entorno->error);
		return entorno->error;
	}

	entorno->buffInCA = clCreateBuffer(entorno->contexto, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(double *), A, &entorno->error);
	if (entorno->error != CL_SUCCESS)
	{
		printf("Error al crear el segundo buffer de entrada. ");
		CodigoError(entorno->error);
		return entorno->error;
	}

	entorno->buffOutVecA = clCreateBuffer(entorno->contexto, CL_MEM_USE_HOST_PTR, sizeof(double *), vecA, &entorno->error);
	if (entorno->error != CL_SUCCESS)
	{
		printf("Error al crear el buffer de salida. ");
		CodigoError(entorno->error);
		return entorno->error;
	}

	entorno->error = clSetKernelArg(entorno->kernel, 0, sizeof(int), &N);
	if (entorno->error != CL_SUCCESS)
	{
		printf("Error al asignar el primer parámetro. ");
		CodigoError(entorno->error);
		return entorno->error;
	}
	entorno->error = clSetKernelArg(entorno->kernel, 1, sizeof(cl_mem), &entorno->buffOutVecA);
	if (entorno->error != CL_SUCCESS)
	{
		printf("Error al asignar el segundo parámetro. ");
		CodigoError(entorno->error);
		return entorno->error;
	}
	entorno->error = clSetKernelArg(entorno->kernel, 2, sizeof(cl_mem), &entorno->buffInTernas);
	if (entorno->error != CL_SUCCESS)
	{
		printf("Error al asignar el tercer parámetro. ");
		CodigoError(entorno->error);
		return entorno->error;
	}

	entorno->error = clSetKernelArg(entorno->kernel, 3, sizeof(cl_mem), &entorno->buffInCA);
	if (entorno->error != CL_SUCCESS)
	{
		printf("Error al asignar el cuarto parámetro. ");
		CodigoError(entorno->error);
		return entorno->error;
	}

	cl_event EventoExec;
	size_t NumWI = num_sb;
	entorno->error = clEnqueueNDRangeKernel(entorno->cola, entorno->kernel, 1, NULL, &NumWI, NULL, 0, NULL, &EventoExec);
	if (entorno->error != CL_SUCCESS)
	{
		printf("Error al encolar la ejecución del kernel: ");
		CodigoError(entorno->error);
		return entorno->error;
	}

	clFinish(entorno->cola);

	// recorrer el vecA para sumar cada submatriz en A
	// bucle que recorra el num_sb, otro fila y colum
	// y dentro hago la suma
	for (int k = 0; k < num_sb; k++)
	{
		double *auxA = &vecA[k * N * N];
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				// para no acumular las A
				if (k == 0 && myrank != 0)
				{
					A[i * N + j] = auxA[i * N + j];
				}
				else
				{
					if (auxA[i * N + j] != 0)
					{
						A[i * N + j] += auxA[i * N + j];
					}
				}
	}
}
// **************************************************************************
// *************************** FIN IMPLEMENTACION ***************************
// **************************************************************************

/*
Recibir� los siguientes par�metros (los par�metros entre corchetes son opcionales): fichEntrada [-d]
fichEntrada -> Obligatorio. Fichero de entrada con los par�metros de lanzamiento de los experimentos
-d -> Opcional. Si se indica, se mostrar�n por pantalla los valores iniciales, finales y tiempo de cada experimento
-wi work_items -> Opcional. Si se indica, se lanzar�n tantos work items como se indique en work_items (para OpenCL)
-wi_wg workitems_por_workgroup -> Opcional. Si se indica, se lanzar�n tantos work items en cada work group como se indique en WorkItems_por_WorkGroup (para OpenCL)
*/
int main(int argc, char *argv[])
{
	int i,
		debug = 0,					  // Indica si se desean mostrar los tiempos y resultados parciales de los experimentos
		num_workitems = 0,			  // Numero de work items que se utilizaran
		workitems_por_workgroups = 0, // Numero de work items por cada work group que se utilizaran
		num_problems,				  // Numero de experimentos
		matrix_size,				  // Tamaño de la matriz
		seed,						  // Semilla
		num_random,					  // Numero de submatrices
		myrank,						  // Identificador del proceso
		size;						  // Numero de procesos lanzados
	double *A;						  // Matriz de datos. Se representa en forma de vector. Para acceder a la fila f y la columna c: A[f*N+c]
	terna_t *ternas;				  // Vector de ternas con los tamaños y las coordenadas de las submatrices
	long long ti,					  // Tiempo inicial
		tf,							  // Tiempo final
		tt = 0;						  // Tiempo acumulado de los tiempos parciales de todos los experimentos realizados
	FILE *f;						  // Fichero con los datos de entrada
	EntornoOCL_t entorno;			  // Entorno para el control de OpenCL

	if (!ObtenerParametros(argc, argv, &debug, &num_workitems, &workitems_por_workgroups))
	{
		printf("Ejecuci�n incorrecta\nEl formato correcto es %s fichEntrada [-d] [-wi work_items] [-wi_wg workitems_por_workgroup]\n", argv[0]);
		return 0;
	}

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	InicializarEntornoOCL(&entorno);

	// Se leen el numero de experimentos a realizar
	if (myrank == 0)
	{ // Solo el proceso 0 tiene acceso al fichero y, por tanto, a los datos
		f = fopen(argv[1], "r");
		fscanf(f, "%d", &num_problems);
	}

	ti = mseconds();
	// **************************************************************************
	// ***************************** IMPLEMENTACION *****************************
	// **************************************************************************

	// Se debe enviar el número de experimentos (num_problems) a todos los procesos

	MPI_Bcast(&num_problems, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// **************************************************************************
	// *************************** FIN IMPLEMENTACION ***************************
	// **************************************************************************
	tf = mseconds();
	tt += tf - ti;

	for (i = 0; i < num_problems; i++)
	{
		if (myrank == 0)
		{ // S�lo el proceso 0 tiene acceso al fichero y, por tanto, a los datos
			// Por cada experimento se leen
			fscanf(f, "%d", &matrix_size); // Tama�o de la matriz (cuadrada)
			fscanf(f, "%d", &seed);		   // Semilla para la inicializaci�n de n�meros aleatorios
			fscanf(f, "%d", &num_random);  // N�mero de submatrices a generar
			// Reserva de memoria para la matriz de datos y las ternas de las submatrices
			A = (double *)malloc(sizeof(double) * matrix_size * matrix_size);
			ternas = (terna_t *)malloc(sizeof(terna_t) * num_random);

			srand(seed);
			initialize(matrix_size, A, ternas, num_random);

			if (debug)
			{
				printf("Matriz original del experimento %d:\n", i);
				escribir(matrix_size, A);
				printf("Submatrices del experimento %d:\n", i);
				escribirt(ternas, num_random);
			}
		}

		ti = mseconds();
		// **************************************************************************
		// ***************************** IMPLEMENTACION *****************************
		// **************************************************************************

		// 1. enviar todos los tam matrix_size y num_random (bcast)
		MPI_Bcast(&matrix_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&num_random, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// 2. enviar la matriz entera: primero reservamos memoria en los procesos de 1 a x y luego enviamos A (bcast)
		if (myrank != 0)
		{
			A = (double *)malloc(sizeof(double) * matrix_size * matrix_size);
		}
		MPI_Bcast(A, matrix_size * matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// 3. repartir las ternas: reservar memoria en todos los procesos y luego se reparten las ternas (scatter)
		// el tam del trozo que se envia y el que se recibe es el mismo(tam es num_random/size)
		// ternas es un struct, para enviarlas la opcion sencilla es coger tam y multpx3
		int tam = num_random / size;
		int repTernas = tam * 3;

		terna_t *myterna = (terna_t *)malloc(sizeof(terna_t) * tam);
		// MPI_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype,void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,MPI_Comm comm)
		/*sendbuf	Dirección inicial del buffer de salida, solo útil para el proceso raíz, el resto de procesos ignoran este parámetro.
		sendcount	Número de elementos que se envía a cada proceso del comunicador (entero que sólo tiene sentido en el raíz).
		sendtype	Tipo de dato que se va a enviar, solo lo tendrá en cuenta la raíz (Como por ejemplo MPI_INT).
		recvbuf		Direción del buffer de recepción (para todos los procesos, incluido el proceso raíz).
		recvcount	Número de elementos que espera recibir cada proceso (int).
		recvtype	Tipo de datos de los elementos a recibir (Como por ejemplo MPI_INT).
		root		Rango (rank) del proceso raíz (el que realizará el envío).
		comm		Comunicador por el que realizar la comunicación.
		*/
		MPI_Scatter(ternas, repTernas, MPI_INT, myterna, repTernas, MPI_INT, 0, MPI_COMM_WORLD);

		// 4. llamar a ocl
		ocl(matrix_size, A, myterna, tam, &entorno, num_workitems, workitems_por_workgroups);

		// 5. reduction de A: definir una nueva variable, reservarla y hacer las reducion de todas las  A
		// sobre esa nnueva variable y coger esa variable y copiarla en a (liberar A y asignar la nueva variable)
		double *copiaA;
		if (myrank == 0)
		{
			copiaA = (double *)malloc(matrix_size * matrix_size * sizeof(double));
		}

		// 	MPI_Reduce(void *sendbuf, void *recvbuf, int count,MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
		/*sendbuf	Dirección inicial del buffer en envío.
		recvbuf		Dirección inicial del buffer de recepción, útil únicamente para el proceso raíz.
		count		Número de elementos que se va a enviar del buffer de envío (int).
		datatype	Tipo de datos de los elementos del buffer de envío (por ejemplo MPI_INT).
		op			Operación de reducción, constante definida por MPI (Definidas en el apartado de descripción).
		root		Rango del proceso raíz, el proceso receptor (int).
		comm		Comunicador por el que se realiza la comunicación.
		*/
		MPI_Reduce(A, copiaA, matrix_size * matrix_size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		free(A);
		A = copiaA;
		// Deberan crearse las estructuras que se consideren necesarias para almacenar las partes de la informacion de cada proceso
		// El proceso 0 debe repartir la informacion a procesar entre todos los procesos (incluido al mismo)

		// El proceso 0 debe recolectar la informacion procesada por todos los procesos (incluida la suya)
		// Deberan liberarse todas las estructuras creadas para almacenar las partes de la informacion de cada proceso

		// **************************************************************************
		// *************************** FIN IMPLEMENTACION ***************************
		// **************************************************************************
		tf = mseconds();
		tt += tf - ti;

		if (myrank == 0)
		{
			if (debug)
			{
				printf("Tiempo del experimento %d: %Ld ms\n", i, tf - ti);
				printf("Matriz resultado del experimento %d:\n", i);
				escribir(matrix_size, A);
			}
			free(A);
			free(ternas);
		}
	}

	LiberarEntornoOCL(&entorno);
	MPI_Finalize();
	if (myrank == 0)
	{
		printf("Tiempo total de %d experimentos: %Ld ms\n", num_problems, tt);
		fclose(f);
	}

	return 0;
}
