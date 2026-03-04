# HPC-Matrix-Multiplication-CUDA1
# Matrix Multiplication using CUDA

This project demonstrates matrix multiplication using GPU parallel computing with CUDA.

Matrix multiplication formula:

C[i][j] = Σ A[i][k] * B[k][j]

CUDA uses thousands of threads to compute matrix multiplication efficiently.

Advantages:
- Faster execution for large matrices
- Parallel processing

Compilation:

nvcc matrix_multiplication.cu -o matrix

Execution:

./matrix
