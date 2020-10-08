build:	
		mpicc -c histogram.c	
		mpicc -o histogram histogram.o -lm	

clean:	
	rm *.o histogram	

run:	
	mpiexec -np $(N) ./histogram