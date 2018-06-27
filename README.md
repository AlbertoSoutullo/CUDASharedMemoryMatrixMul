# CUDASharedMemoryMatrixMul

This kernel was used to study diferent computation times with diferent matrix sizes.

Multiplying 2 matrix with size of 10000x10000, we obtained the following results:

Trying to do a secuencial multiplication witha simple for loop:<br>
  -Unable to computate
  
Doing an 8 threads static division multiplication:<br>
  -1595'099 sec

Using CUDA with shared memory:<br>
  -18'914
  
With CUDA we obtained a speedup of 84'334302 compared with static division.

The results were obtained with an Intel Xenon and a nVidia GTX 560.
