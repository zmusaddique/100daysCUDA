### Log

It is still a buggy kernel

What were the issues?
- Functions signatures weren't detected - There was a mismatch in function signatures between host function and the python bindings
- Memory out of bounds when computing dK and dV - Had to insert padding in the shared memory and access pattern of indexing each tile in shared mem 
- dQ is calculated wrong hugely - The access patterns seem right and also introduced smem padding. The computation operations also seems right. There might be a basic issue that I'm overlooking. I need to analyze the inputs and the overall functionality of compute_dQ.

That's it for today
