Improvements from prev kernel:

- Adding padded dimensions for reduced bank conflicts
- warp-level reductions for max_val
- reduced memory footprint 
- more const vars

### New learnings 

__shfl_down_sync(0xffffffff, max_val, offset)
- gets a value from a thread that is offset positions away
- 0xffffffff - is a mask that indicates all threads in warp are participating

Broadcasting the Result:
__shfl_sync(0xffffffff, max_val, 0) broadcasts the value from thread 0 to all other threads

## Reading
- Started learning backward pass from Umar sensei's vid  
