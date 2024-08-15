extern "C" {
#include <components/mc/ucc_mc.h>
#include <pthread.h>
}
#include <common/test.h>
#include <common/test_ucc.h>
#include <cuda_runtime.h>
#include <vector>

class test_mc_memcpy : public ::testing::Test {
protected:
    void SetUp() override {
        ucc_mc_params_t mc_params = {
            .thread_mode = UCC_THREAD_SINGLE,
        };

        ucc_constructor();
        ucc_mc_init(&mc_params);

        if (UCC_OK != ucc_mc_available(UCC_MEMORY_TYPE_CUDA)) {
            GTEST_SKIP();
        }
        cudaMalloc(&src, max_size);
        cudaMalloc(&dst, max_size);
    }

    void TearDown() override {
        cudaFree(src);
        cudaFree(dst);
        ucc_mc_finalize();
    }

    void* src;
    void* dst;
    size_t max_size = 1 << 20; // 1 MB
};

TEST_F(test_mc_memcpy, test_device_to_device_performance) {
    std::vector<size_t> sizes = {1, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576}; // 1 byte to 1 MB
    int num_iterations = 1;

    for (size_t size : sizes) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_iterations; i++) {
            ucc_mc_memcpy(dst, src, size, UCC_MEMORY_TYPE_CUDA, UCC_MEMORY_TYPE_CUDA);
        }

        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        double average_latency = (diff.count() * 1e6) / num_iterations;
        printf("ucc_mc_memcpy of %zu bytes (%d iterations): %.2f µs (average)\n", size, num_iterations, average_latency);
    }
}

