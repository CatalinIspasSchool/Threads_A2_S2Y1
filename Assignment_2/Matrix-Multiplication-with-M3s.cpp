//Matrix Multiplication using different Memory Management Model
// CMP202
// j.zarrin@abertay.ac.uk
#include <sycl/sycl.hpp>
#include <iostream>
#include <algorithm>
#include <chrono>
using namespace sycl;
constexpr size_t M[5] = {25, 50, 125, 400, 2000};
constexpr size_t N[5] = {10, 30, 60, 300, 1000};
constexpr size_t P[5] = {4, 20, 80, 250, 500};
constexpr size_t TILE_SIZE[5] = {2, 5, 10, 50, 50};



void tiled_matrix_multiplication(const float* A, const float* B, float* C, queue& q, const int testCase) {
    buffer<float, 2> bufA(A, range<2>(M[testCase], N[testCase]));
    buffer<float, 2> bufB(B, range<2>(N[testCase], P[testCase]));
    buffer<float, 2> bufC(C, range<2>(M[testCase], P[testCase]));
    q.submit([&](handler& h) {
        sycl::stream out(1028, 256, h);

        auto accA = bufA.get_access<access::mode::read>(h);
        auto accB = bufB.get_access<access::mode::read>(h);
        auto accC = bufC.get_access<access::mode::write>(h);
        accessor<float, 2, access::mode::read_write, access::target::local> tileA(range<2>(TILE_SIZE[testCase], TILE_SIZE[testCase]), h);
        accessor<float, 2, access::mode::read_write, access::target::local> tileB(range<2>(TILE_SIZE[testCase], TILE_SIZE[testCase]), h);

        h.parallel_for<class TiledMatrixMulKernel>(nd_range<2>(range<2>(M[testCase], P[testCase]), range<2>(TILE_SIZE[testCase], TILE_SIZE[testCase])), [=](nd_item<2> item) {
            const int globalRow = item.get_global_id(0);
            const int globalCol = item.get_global_id(1);
            const int localRow = item.get_local_id(0);
            const int localCol = item.get_local_id(1);

            float temp = 0.0f;
            for (int t = 0; t < N[testCase]; t += TILE_SIZE[testCase]) {
                // Load tiles into local memory
                tileA[localRow][localCol] = accA[globalRow][t + localCol];
                tileB[localRow][localCol] = accB[t + localRow][globalCol];
                item.barrier(access::fence_space::local_space);

                // Perform tile multiplication
                for (int k = 0; k < TILE_SIZE[testCase]; ++k) {
                    temp += tileA[localRow][k] * tileB[k][localCol];
                }
                item.barrier(access::fence_space::local_space); // Wait for all work-items to finish
            }
            accC[globalRow][globalCol] = temp;
            });
        });
}

void matrix_multiplication(const float* A, const float* B, float* C, queue& q, const int testCase) {
    buffer<float, 2> bufA(A, range<2>(M[testCase], N[testCase]));
    buffer<float, 2> bufB(B, range<2>(N[testCase], P[testCase]));
    buffer<float, 2> bufC(C, range<2>(M[testCase], P[testCase]));
    q.submit([&](handler& h) {
        auto accA = bufA.get_access<access::mode::read>(h);
        auto accB = bufB.get_access<access::mode::read>(h);
        auto accC = bufC.get_access<access::mode::write>(h);

        h.parallel_for<class MatrixMulKernel>(range<2>(M[testCase], P[testCase]), [=](id<2> idx) {
            const int i = idx[0];
            const int j = idx[1];
            float temp = 0.0f;
            for (int k = 0; k < N[testCase]; ++k) {
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

void e_usm_matrix_multiplication(const float* A_host, const float* B_host, float* C_host, queue& q, const int testCase) {
    // Allocate memory on the device
    float* A = malloc_device<float>(M[testCase] * N[testCase], q);
    float* B = malloc_device<float>(N[testCase] * P[testCase], q);
    float* C = malloc_device<float>(M[testCase] * P[testCase], q);

    // Copy data from host to device
    q.memcpy(A, A_host, sizeof(float) * M[testCase] * N[testCase]).wait();
    q.memcpy(B, B_host, sizeof(float) * N[testCase] * P[testCase]).wait();

    // Ensure the data is copied before starting computation
    //q.wait();

    // Perform the matrix multiplication on the device
    q.submit([&](handler& h) {
        //sycl::stream out(1028, 256, h);
        h.parallel_for<class MatrixMulKernelUSMe>(range<2>(M[testCase], P[testCase]), [=](id<2> idx) {
            const int i = idx[0];
            const int j = idx[1];
            float temp = 0.0f;
            for (int k = 0; k < N[testCase]; ++k) {
                temp += A[i * N[testCase] + k] * B[k * P[testCase] + j];
                //out << i << " " << j << " " << k << " " << temp << endl;
            }
            C[i * P[testCase] + j] = temp;
        });
    });
    q.wait();

    // Copy the result back to host memory
    q.memcpy(C_host, C, sizeof(float) * M[testCase] * P[testCase]).wait();

    // Free device memory
    free(A, q);
    free(B, q);
    free(C, q);
}

int main() {



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

    //1
    { 
        float* A = new float[M[0] * N[0]];
        float* B = new float[N[0] * P[0]];
        float* C = new float[M[0] * P[0]];
        int loopCount = 30;

        for (int i = 0; i < M[0]; i++) {
            for (int j = 0; j < N[0]; j++) {
                A[i * N[0] + j] = i;
            }
        }
        for (int i = 0; i < N[0]; i++) {
            for (int j = 0; j < P[0]; j++) {
                B[i * P[0] + j] = j;
            }
        }
        matrix_multiplication(A, B, C, q, 0);
        q.wait();
        e_usm_matrix_multiplication(A, B, C, q, 0);
        q.wait();
        tiled_matrix_multiplication(A, B, C, q, 0);
        q.wait();


        float totalBuffTime = 0, totalExpTime = 0, totalTiledTime = 0;
        for (int i = 0; i < loopCount; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            matrix_multiplication(A, B, C, q, 0);
            q.wait();
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            totalBuffTime += duration.count();

            start = std::chrono::high_resolution_clock::now();
            e_usm_matrix_multiplication(A, B, C, q, 0);
            q.wait();
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            totalExpTime += duration.count();

            start = std::chrono::high_resolution_clock::now();
            tiled_matrix_multiplication(A, B, C, q, 0);
            q.wait();
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            totalTiledTime += duration.count();
        }
        std::cout << "100 elements average: Buffer/Accessor - " << totalBuffTime/loopCount/1000000 << "ms \t Explicit - " << totalExpTime/loopCount / 1000000 << "ms \t Tiled - " << totalTiledTime/loopCount / 1000000 << "ms \n";
    }

    //2
    {
        float* A = new float[M[1] * N[1]];
        float* B = new float[N[1] * P[1]];
        float* C = new float[M[1] * P[1]];
        int loopCount = 30;

        for (int i = 0; i < M[1]; i++) {
            for (int j = 0; j < N[1]; j++) {
                A[i * N[1] + j] = i;
            }
        }
        for (int i = 0; i < N[1]; i++) {
            for (int j = 0; j < P[1]; j++) {
                B[i * P[1] + j] = j;
            }
        }

        float totalBuffTime = 0, totalExpTime = 0, totalTiledTime = 0;
        for (int i = 0; i < loopCount; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            matrix_multiplication(A, B, C, q, 1);
            q.wait();
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            totalBuffTime += duration.count();

            start = std::chrono::high_resolution_clock::now();
            e_usm_matrix_multiplication(A, B, C, q, 1);
            q.wait();
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            totalExpTime += duration.count();

            start = std::chrono::high_resolution_clock::now();
            tiled_matrix_multiplication(A, B, C, q, 1);
            q.wait();
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            totalTiledTime += duration.count();
        }
        std::cout << "1000 elements average: Buffer/Accessor - " << totalBuffTime / loopCount / 1000000 << "ms \t Explicit - " << totalExpTime / loopCount / 1000000 << "ms \t Tiled - " << totalTiledTime / loopCount / 1000000 << "ms \n";
    }

    //3
    {
        float* A = new float[M[2] * N[2]];
        float* B = new float[N[2] * P[2]];
        float* C = new float[M[2] * P[2]];
        int loopCount = 20;

        for (int i = 0; i < M[2]; i++) {
            for (int j = 0; j < N[2]; j++) {
                A[i * N[2] + j] = i;
            }
        }
        for (int i = 0; i < N[2]; i++) {
            for (int j = 0; j < P[2]; j++) {
                B[i * P[2] + j] = j;
            }
        }

        float totalBuffTime = 0, totalExpTime = 0, totalTiledTime = 0;
        for (int i = 0; i < loopCount; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            matrix_multiplication(A, B, C, q, 2);
            q.wait();
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            totalBuffTime += duration.count();

            start = std::chrono::high_resolution_clock::now();
            e_usm_matrix_multiplication(A, B, C, q, 2);
            q.wait();
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            totalExpTime += duration.count();

            start = std::chrono::high_resolution_clock::now();
            tiled_matrix_multiplication(A, B, C, q, 2);
            q.wait();
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            totalTiledTime += duration.count();
        }
        std::cout << "10000 elements average: Buffer/Accessor - " << totalBuffTime / loopCount / 1000000 << "ms \t Explicit - " << totalExpTime / loopCount / 1000000 << "ms \t Tiled - " << totalTiledTime / loopCount / 1000000 << "ms \n";
    }

    //4
    {
        float* A = new float[M[3] * N[3]];
        float* B = new float[N[3] * P[3]];
        float* C = new float[M[3] * P[3]];
        int loopCount = 10;

        for (int i = 0; i < M[3]; i++) {
            for (int j = 0; j < N[3]; j++) {
                A[i * N[3] + j] = i;
            }
        }
        for (int i = 0; i < N[3]; i++) {
            for (int j = 0; j < P[3]; j++) {
                B[i * P[3] + j] = j;
            }
        }

        float totalBuffTime = 0, totalExpTime = 0, totalTiledTime = 0;
        for (int i = 0; i < loopCount; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            matrix_multiplication(A, B, C, q, 3);
            q.wait();
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            totalBuffTime += duration.count();

            start = std::chrono::high_resolution_clock::now();
            e_usm_matrix_multiplication(A, B, C, q, 3);
            q.wait();
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            totalExpTime += duration.count();

            start = std::chrono::high_resolution_clock::now();
            tiled_matrix_multiplication(A, B, C, q, 3);
            q.wait();
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            totalTiledTime += duration.count();
        }
        std::cout << "100000 elements average: Buffer/Accessor - " << totalBuffTime / loopCount / 1000000 << "ms \t Explicit - " << totalExpTime / loopCount / 1000000 << "ms \t Tiled - " << totalTiledTime / loopCount / 1000000 << "ms \n";
    }

    //5
    {
        float* A = new float[M[4] * N[4]];
        float* B = new float[N[4] * P[4]];
        float* C = new float[M[4] * P[4]];
        int loopCount = 5;

        for (int i = 0; i < M[4]; i++) {
            for (int j = 0; j < N[4]; j++) {
                A[i * N[4] + j] = i;
            }
        }
        for (int i = 0; i < N[4]; i++) {
            for (int j = 0; j < P[4]; j++) {
                B[i * P[4] + j] = j;
            }
        }

        float totalBuffTime = 0, totalExpTime = 0, totalTiledTime = 0;
        for (int i = 0; i < loopCount; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            matrix_multiplication(A, B, C, q, 4);
            q.wait();
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            totalBuffTime += duration.count();

            start = std::chrono::high_resolution_clock::now();
            e_usm_matrix_multiplication(A, B, C, q, 4);
            q.wait();
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            totalExpTime += duration.count();

            start = std::chrono::high_resolution_clock::now();
            tiled_matrix_multiplication(A, B, C, q, 4);
            q.wait();
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            totalTiledTime += duration.count();
        }
        std::cout << "1000000 elements average: Buffer/Accessor - " << totalBuffTime / loopCount / 1000000 << "ms \t Explicit - " << totalExpTime / loopCount / 1000000 << "ms \t Tiled - " << totalTiledTime / loopCount / 1000000 << "ms \n";
    }

    
    //float* A = new float[M[0] * N[0]];
    //float* B = new float[N[0] * P[0]];
    //float* C = new float[M[0] * P[0]];
    //int loopCount = 30;

    //for (int i = 0; i < M[0]; i++) {
    //    for (int j = 0; j < N[0]; j++) {
    //        A[i * N[0] + j] = i;
    //    }
    //}
    //for (int i = 0; i < N[0]; i++) {
    //    for (int j = 0; j < P[0]; j++) {
    //        B[i * P[0] + j] = j;
    //    }
    //}

    //auto start = std::chrono::high_resolution_clock::now();
    //matrix_multiplication(A, B, C, q, 0);
    //q.wait();
    //auto stop = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    //std::cout << "For the Data: " << N[0] << "x" << N[0] << "-Matrix multiplication took " << duration.count() << " Milliseconds.\n";
    //std::cout << "Only part of the matrices is printed. AxB=C\n";
    //int Z = std::min(static_cast<int>(N[0]), 3);
    //for (int i = 0; i < Z; i++) {
    //    for (int j = 0; j < Z; j++) {
    //        std::cout << A[i * N[0] + j] << "\t";
    //    }
    //    std::cout << "\t\t";
    //    for (int j = 0; j < Z; j++) {
    //        std::cout << B[i * P[0] + j] << "\t";
    //    }
    //    std::cout << "\t\t";
    //    for (int j = 0; j < Z; j++) {
    //        std::cout << C[i * P[0] + j] << "\t";
    //    }
    //    std::cout << std::endl;
    //}

    //free(A, q); // Free the allocated USM memory
    //free(B, q);
    //free(C, q);
    return 0;
}
