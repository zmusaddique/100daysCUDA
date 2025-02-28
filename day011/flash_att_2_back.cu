__global__ void compute_dV(float *dV, const float *P, const float *dO, int N, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < d) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += P[row * N + k] * dO[k * d + col];
        }
        dV[row * d + col] = sum;
    }
}


__global__ void compute_dP(float *dP, const float *dO, const float *V, int N, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < d; ++k) {
            sum += dO[row * d + k] * V[col * d + k];
        }
        dP[row * N + col] = sum;
    }
}

// WIP
