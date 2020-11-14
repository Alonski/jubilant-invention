build:	
	mpicxx -fopenmp -c histogram.c -o histogram.o
	nvcc -I./inc -c histogramCUDA.cu -o histogramCUDA.o
	mpicxx -fopenmp -o histogram histogram.o histogramCUDA.o /usr/local/cuda-9.1/lib64/libcudart_static.a -ldl -lrt


clean:	
	rm *.o histogram	

run:	
	mpiexec -np 2 ./histogram