build:	
		mpicc -fopenmp -o histogram histogram.c	

clean:	
	rm *.o histogram	

run:	
	mpiexec -np $(N) ./histogram