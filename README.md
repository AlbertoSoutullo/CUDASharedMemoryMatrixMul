# CUDASharedMemoryMatrixMul

<b>Matrix structure used:</b><br>
We used .bin files with raw numbers, being the first one and the second one the number
of rows and the number of columns respectively.

In order to creathe those matrix as easy as possible, a .cpp file is added. In it we can
create 2 matrix, the first one with random numbers and a given size, and the second
one will be an identity matrix, with a given size aswell.


<b>Kernel:</b><br>
This kernel was used to study diferent computation times with diferent matrix sizes.<br>
The multiplication is done like the following image:
<img src="https://s3.amazonaws.com/i.seelio.com/6f/fd/6ffd44cf043d8c0e80e4652da28bffb6ae1e.png">

Multiplying 2 matrix with size of 10000x10000, we obtained the following results:

Trying to do a <u>secuencial multiplication with a simple FOR loop</u>:<br>
  -Unable to computate
  
Doing an <u>8 threads static division multiplication</u>:<br>
  -1595'099 sec

Using <u>CUDA with shared memory</u>:<br>
  -18'914 sec
  
With CUDA we obtained a speedup of 84'334302 compared with static division.

The results were obtained with an <i>Intel Xenon</i> and a <i>nVidia GTX 560</i>.
