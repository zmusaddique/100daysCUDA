// Attention = softmax((Q*Kt)/sqrt(dk))*V 
// We are interested in optimizing softmax, not matMULs
//  

// Safe softmax = e^(Zi - k)/ sum(Ezi - k), k - max val

// steps in softmax -
// 1. Find max val - 1 iteration
// 2. Calculate normalization factor - 1 iteration
// 3. Apply softmax - 1 iteration

// All of these are sequential and require previous steps to be executed

// However by being stubborn to reduce the passes we arrive to just 2 passes
// by using  online softmax and max in one iter 

// 1. m= -inf
//    l0 =  

// for i = 1 to N
//    mi = max(mi-1, Xi)
//    li = li-1* e^(mi-1 - mi) + e^(Xi-mi)

// for k=1 to N
//  softmax with current values
