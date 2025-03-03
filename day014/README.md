# Thought Journal

Writing this just so that I don't end up regretting taking this so much time simply because I forgot the effort to reach this point in understanding.

What am I currently stuck with? - The backward pass of the kernel.  
What are the topics covered afresh? - Jacobian matrix, expressing jacobian in terms of transpose of an input element, jacobian through the softmax.

What is a jacobian? - It is the differentiation of each $$y_{k}$$ wrt each input row of X - This constitutes a single row of the matrix. Each row in the matrix is a representation of an output element of Y differentiated wrt to an input row vector$$x_{1}$$

Consideriing a matmul $$Y = XW$$, where $$X \in \mathbb{R}^{2 \times3},\ W \in \mathbb{R}^{3 \times 4},\ \ Y \in \mathbb{R}^{2 \times 4}$$

Now,
$$\frac{\partial \phi}{\partial X} = \frac{\partial \phi}{\partial Y} \cdot \frac{\partial Y}{\partial X}$$

The jacobian $$\frac{\partial \phi}{\partial Y}$$ is given by Pytorch (assuming there is a feed-forward NN after Y) and

$$
\frac{\partial Y}{\partial X} = \begin{pmatrix}
W^T & 0 \\
0 & W^T
\end{pmatrix}
$$

Thus, $$\frac{\partial \phi}{\partial X} =  \frac{\partial \phi}{\partial Y} \cdot W^T$$ and similarly, $$\frac{\partial \phi}{\partial W} = X^T \cdot \frac{\partial \phi}{\partial Y}$$ (trick is to match product dimensions to the expected input result dimensions)

This working out shows that we can use existing input matrices to compute the jacobian without having to materializing it and saving huge amounts of memory.

### Gradient through the softmax

We know that $$S = QK^T,\ \ P_{i} = sofmax(S_{i}),\ \ O = PV$$

$$
\[
S_{i} = S_{i, :}\in \mathbb{R}^N\\
P_{i} = Softmax(S_{i}) \in \mathbb{R}^N\\
P_{iJ} = \frac{e^{S_{i,J}}}{\sum_{l=1}^{N}e^{S_{il}}}\\
\frac{\partial \phi}{\partial S_{i,J}} = \frac{\partial \phi}{\partial P_{i}} \cdot \frac{\partial P_{i}}{\partial S_{i}}
\]
$$

For each element in output vector wrt each element in input vector - Oh, wait this is what a jacobian is. I misunderstood it before forgetting this partial differentiation was a ~row op~ It's one output element wrt an input element - This is an element in the row of the jacobian.

The Jacobian matrix of a vector function$$ \( \mathbf{f} : \mathbb{R}^n \to \mathbb{R}^m \), where \( \mathbf{f}(\mathbf{x}) = (y_1, y_2, \dots, y_m) \) and \( \mathbf{x} = (x_1, x_2, \dots, x_n) \)$$, is given by:

$$
\[
\mathbf{J} =
\begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & \dots & \frac{\partial y_1}{\partial x_n} \\
\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & \dots & \frac{\partial y_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial y_m}{\partial x_1} & \frac{\partial y_m}{\partial x_2} & \dots & \frac{\partial y_m}{\partial x_n}
\end{bmatrix}
\]
$$

Looking at each element of the jacobian $$\frac{\partial P_{i,J}}{\partial S_{i,k}}$$,

$$
\[
\frac{\partial P_{i,J}}{\partial S_{i,k}} = \frac{\partial}{\partial S_{i,k}} \left( \frac{e^{S_{i,J}}}{\sum_{l=1}^{N} e^{S_{i,l}}} \right)
\]
$$

While differentiating, we have 2 cases, 1. J=k and J!=k in the jacobian.

1. When J=k
   We can find $$\frac{\partial P_{i,J}}{\partial S_{i,k}} = P_{i,J}\cdot (1 - P_{i,k})$$
2. When J!=k
   $$\frac{\partial P_{i,J}}{\partial S_{i,k}} = -P_{i,k} \cdot P_{i,J}$$

Writing down the jacobian, we find the jacobian for softmax can be written as $$diag(P_{i}) - P_{i} \cdot P_{i}^T$$

### Coming to the backward pass algo
