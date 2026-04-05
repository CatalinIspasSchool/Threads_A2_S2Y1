//Matrix Multiplication using different Memory Management Model
// CMP202
// j.zarrin@abertay.ac.uk
#include <sycl/sycl.hpp>
#include <iostream>
#include <algorithm>
#include <chrono>
using namespace sycl;
constexpr size_t M1 = 25;// For performance results -- try large scale matrices
constexpr size_t N1 = 10;// For performance results -- try large scale matrices
constexpr size_t P1 = 4;// For performance results -- try large scale matrices
constexpr size_t TILE_SIZE1 = 2; // Set an appropriate tile size

constexpr size_t M2 = 50;// For performance results -- try large scale matrices
constexpr size_t N2 = 30;// For performance results -- try large scale matrices
constexpr size_t P2 = 20;// For performance results -- try large scale matrices
constexpr size_t TILE_SIZE2 = 5; // Set an appropriate tile size

constexpr size_t M3 = 125;// For performance results -- try large scale matrices
constexpr size_t N3 = 60;// For performance results -- try large scale matrices
constexpr size_t P3 = 80;// For performance results -- try large scale matrices
constexpr size_t TILE_SIZE3 = 5; // Set an appropriate tile size

constexpr size_t M4 = 400;// For performance results -- try large scale matrices
constexpr size_t N4 = 300;// For performance results -- try large scale matrices
constexpr size_t P4 = 250;// For performance results -- try large scale matrices
constexpr size_t TILE_SIZE4 = 10; // Set an appropriate tile size

constexpr size_t M5 = 2000;// For performance results -- try large scale matrices
constexpr size_t N5 = 1000;// For performance results -- try large scale matrices
constexpr size_t P5 = 500;// For performance results -- try large scale matrices
constexpr size_t TILE_SIZE5 = 20; // Set an appropriate tile size


void tiled_matrix_multiplication(const float* A, const float* B, float* C, queue& q) {
    buffer<float, 2> bufA(A, range<2>(M, N));
    buffer<float, 2> bufB(B, range<2>(N, P));
    buffer<float, 2> bufC(C, range<2>(M, P));
    q.submit([&](handler& h) {
        sycl::stream out(1028, 256, h);

        auto accA = bufA.get_access<access::mode::read>(h);
        auto accB = bufB.get_access<access::mode::read>(h);
        auto accC = bufC.get_access<access::mode::write>(h);
        accessor<float, 2, access::mode::read_write, access::target::local> tileA(range<2>(TILE_SIZE, TILE_SIZE), h);
        accessor<float, 2, access::mode::read_write, access::target::local> tileB(range<2>(TILE_SIZE, TILE_SIZE), h);

        h.parallel_for<class TiledMatrixMulKernel>(nd_range<2>(range<2>(M, P), range<2>(TILE_SIZE, TILE_SIZE)), [=](nd_item<2> item) {
            const int globalRow = item.get_global_id(0);
            const int globalCol = item.get_global_id(1);
            const int localRow = item.get_local_id(0);
            const int localCol = item.get_local_id(1);

            float temp = 0.0f;
            for (int t = 0; t < N; t += TILE_SIZE) {
                // Load tiles into local memory
                tileA[localRow][localCol] = accA[globalRow][t + localCol];
                tileB[localRow][localCol] = accB[t + localRow][globalCol];
                item.barrier(access::fence_space::local_space);

                // Perform tile multiplication
                for (int k = 0; k < TILE_SIZE; ++k) {
                    temp += tileA[localRow][k] * tileB[k][localCol];
                }
                item.barrier(access::fence_space::local_space); // Wait for all work-items to finish
            }
            accC[globalRow][globalCol] = temp;
            });
        });
}

void matrix_multiplication(const float* A, const float* B, float* C, queue& q) {
    buffer<float, 2> bufA(A, range<2>(M, N));
    buffer<float, 2> bufB(B, range<2>(N, P));
    buffer<float, 2> bufC(C, range<2>(M, P));
    q.submit([&](handler& h) {
        auto accA = bufA.get_access<access::mode::read>(h);
        auto accB = bufB.get_access<access::mode::read>(h);
        auto accC = bufC.get_access<access::mode::write>(h);

        h.parallel_for<class MatrixMulKernel>(range<2>(M, P), [=](id<2> idx) {
            const int i = idx[0];
            const int j = idx[1];
            float temp = 0.0f;
            for (int k = 0; k < N; ++k) {
                temp += accA[i][k] * accB[k][j];
            }
            accC[i][j] = temp;
            });
        });
}

//void i_usm_matrix_multiplication(const float* A, const float* B, float* C, queue& q) {
//    // The kernel now directly uses the pointers A, B, and C
//    q.submit([&](handler& h) {
//        h.parallel_for<class MatrixMulKernelUSMi>(range<2>(N, N), [=](id<2> idx) {
//            const int i = idx[0];
//            const int j = idx[1];
//            float temp = 0.0f;
//            for (int k = 0; k < N; ++k) {
//                temp += A[i * N + k] * B[k * N + j];
//            }
//            C[i * N + j] = temp;
//            });
//        });
//}

void e_usm_matrix_multiplication(const float* A_host, const float* B_host, float* C_host, queue& q) {
    // Allocate memory on the device
    float* A = malloc_device<float>(M * N, q);
    float* B = malloc_device<float>(N * P, q);
    float* C = malloc_device<float>(M * P, q);

    // Copy data from host to device
    q.memcpy(A, A_host, sizeof(float) * M * N).wait();
    q.memcpy(B, B_host, sizeof(float) * N * P).wait();

    // Ensure the data is copied before starting computation
    //q.wait();

    // Perform the matrix multiplication on the device
    q.submit([&](handler& h) {
        //sycl::stream out(1028, 256, h);
        h.parallel_for<class MatrixMulKernelUSMe>(range<2>(M, P), [=](id<2> idx) {
            const int i = idx[0];
            const int j = idx[1];
            float temp = 0.0f;
            for (int k = 0; k < N; ++k) {
                temp += A[i * N + k] * B[k * P + j];
                //out << i << " " << j << " " << k << " " << temp << endl;
            }
            C[i * P + j] = temp;
        });
    });
    q.wait();

    // Copy the result back to host memory
    q.memcpy(C_host, C, sizeof(float) * M * P).wait();

    // Free device memory
    free(A, q);
    free(B, q);
    free(C, q);
}

int main() {

    float* A = new float[M1 * N1];
    float* B = new float[N1 * P1];

    float* C = new float[M1 * P1];


    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = i;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++) {
            B[i * P + j] = j;
        }
    }


    queue q(default_selector_v);            // set to gpu_selector_v or cpu_selector_v to set to cpu or gpu, but for now it defaults to cpu because IT STILL ISN'T WORKING



    // Querying local memory size and maximum work-group size
    auto deviceName = q.get_device().get_info<info::device::name>();
    auto localMemSize = q.get_device().get_info<info::device::local_mem_size>();
    auto maxWorkGroupSize = q.get_device().get_info<info::device::max_work_group_size>();

    std::cout << "Device Name:" << deviceName << '\n';
    std::cout << "Local Memory Size: " << localMemSize << " bytes\n";
    std::cout << "Max Work-Group Size: " << maxWorkGroupSize << std::endl;
    //float* A = malloc_shared<float>(N * N, q);
    //float* B = malloc_shared<float>(N * N, q);
    //float* C = malloc_shared<float>(N * N, q);


    //for (int i = 0; i < 30; i++)
    //{
    //}




    auto start = std::chrono::high_resolution_clock::now();
    //matrix_multiplication(A, B, C, q);
    e_usm_matrix_multiplication(A, B, C, q);
    //tiled_matrix_multiplication(A, B, C, q);
    q.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "For the Data: " << N << "x" << N << "-Matrix multiplication took " << duration.count() << " Milliseconds.\n";
    std::cout << "Only part of the matrices is printed. AxB=C\n";
    int Z = std::min(static_cast<int>(N), 3);
    for (int i = 0; i < Z; i++) {
        for (int j = 0; j < Z; j++) {
            std::cout << A[i * N + j] << "\t";
        }
        std::cout << "\t\t";
        for (int j = 0; j < Z; j++) {
            std::cout << B[i * P + j] << "\t";
        }
        std::cout << "\t\t";
        for (int j = 0; j < Z; j++) {
            std::cout << C[i * P + j] << "\t";
        }
        std::cout << std::endl;
    }
    //free(A, q); // Free the allocated USM memory
    //free(B, q);
    //free(C, q);
    return 0;
}
