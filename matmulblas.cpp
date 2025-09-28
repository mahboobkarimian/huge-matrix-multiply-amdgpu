#define ROCM_MATHLIBS_API_USE_HIP_COMPLEX
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                                                        \
    do                                                                                \
    {                                                                                 \
        hipError_t _status = (error);                                                 \
        if(_status != hipSuccess)                                                     \
        {                                                                             \
            std::cerr << "hip error: '" << hipGetErrorString(_status) << "' ("        \
                      << static_cast<int>(_status) << ") at " << __FILE__ << ':'      \
                      << __LINE__ << std::endl;                                       \
            std::exit(EXIT_FAILURE);                                                  \
        }                                                                             \
    } while(false)
#endif

#ifndef CHECK_ROCBLAS_STATUS
#define CHECK_ROCBLAS_STATUS(status)                                                  \
    do                                                                                \
    {                                                                                 \
        rocblas_status _status = (status);                                            \
        if(_status != rocblas_status_success)                                         \
        {                                                                             \
            const char* _msg = nullptr;                                               \
            /* Prefer library string helper if available to avoid ambiguity */        \
            _msg = ::rocblas_status_to_string(_status);                               \
            if(!_msg) _msg = pretty_rocblas_status(_status);                          \
            std::cerr << "rocBLAS error: '" << _msg << "' ("                         \
                      << static_cast<int>(_status) << ") at " << __FILE__ << ':'      \
                      << __LINE__ << std::endl;                                       \
            std::exit(EXIT_FAILURE);                                                  \
        }                                                                             \
    } while(false)
#endif

namespace
{
    void print_hex_bytes(const void* data, std::size_t byte_count)
    {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        for(std::size_t i = 0; i < byte_count; ++i)
        {
            std::cout << std::hex << std::setfill('0') << std::setw(2)
                      << static_cast<unsigned>(bytes[i]) << ' ';
        }
        std::cout << std::dec << '\n';
    }

    void print_complex(const hipFloatComplex& value)
    {
        std::cout << '(' << value.x << " + " << value.y << "i)";
    }

    const char* pretty_rocblas_status(rocblas_status status)
    {
        switch(status)
        {
        case rocblas_status_success:
            return "rocblas_status_success";
        case rocblas_status_invalid_handle:
            return "rocblas_status_invalid_handle";
        case rocblas_status_not_implemented:
            return "rocblas_status_not_implemented";
        case rocblas_status_invalid_pointer:
            return "rocblas_status_invalid_pointer";
        case rocblas_status_invalid_size:
            return "rocblas_status_invalid_size";
        case rocblas_status_memory_error:
            return "rocblas_status_memory_error";
        case rocblas_status_internal_error:
            return "rocblas_status_internal_error";
        case rocblas_status_perf_degraded:
            return "rocblas_status_perf_degraded";
        case rocblas_status_size_query_mismatch:
            return "rocblas_status_size_query_mismatch";
        case rocblas_status_size_increased:
            return "rocblas_status_size_increased";
        case rocblas_status_size_unchanged:
            return "rocblas_status_size_unchanged";
        case rocblas_status_invalid_value:
            return "rocblas_status_invalid_value";
        case rocblas_status_continue:
            return "rocblas_status_continue";
        case rocblas_status_check_numerics_fail:
            return "rocblas_status_check_numerics_fail";
        case rocblas_status_excluded_from_build:
            return "rocblas_status_excluded_from_build";
        case rocblas_status_arch_mismatch:
            return "rocblas_status_arch_mismatch";
        }
        return "rocblas_status_unknown";
    }
} // namespace

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return EXIT_FAILURE;
    }

    const int input_size = std::atoi(argv[1]);
    if(input_size <= 0)
    {
        std::cerr << "Matrix size must be positive." << std::endl;
        return EXIT_FAILURE;
    }

    const rocblas_int m = input_size;
    const rocblas_int n = input_size;
    const rocblas_int k = input_size;
    const std::size_t element_count = static_cast<std::size_t>(m) * n;
    const std::size_t bytes = element_count * sizeof(hipFloatComplex);
    const std::size_t sample_count = std::min<std::size_t>(16, element_count);

    std::cout << "matrix rows: " << m << " cols: " << n << " size: "
              << element_count << " ram: "
              << (static_cast<double>(bytes) / 1'000'000.0) << " MB" << std::endl;

    rocblas_handle handle;
    CHECK_ROCBLAS_STATUS(rocblas_create_handle(&handle));

    CHECK_ROCBLAS_STATUS(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    hipFloatComplex* d_A = nullptr;
    hipFloatComplex* d_B = nullptr;
    hipFloatComplex* d_C = nullptr;
    // Use managed (unified) memory so host writes/reads are valid on these pointers.
    CHECK_HIP_ERROR(hipMallocManaged(reinterpret_cast<void**>(&d_A), bytes, hipMemAttachGlobal));
    CHECK_HIP_ERROR(hipMallocManaged(reinterpret_cast<void**>(&d_B), bytes, hipMemAttachGlobal));
    CHECK_HIP_ERROR(hipMallocManaged(reinterpret_cast<void**>(&d_C), bytes, hipMemAttachGlobal));

    // UNSAFE: mimic Rust program behavior by writing directly to device memory from host.
    // This relies on HMM/fine-grain mappings and is not portable or recommended for production.
    const hipFloatComplex init_value = hipFloatComplex{2.0f, 0.0f};
    for (std::size_t i = 0; i < element_count; ++i) {
        d_A[i] = init_value;
        d_B[i] = init_value;
        d_C[i] = hipFloatComplex{0.0f, 0.0f};
    }

    std::cout << "Matrix A (input):" << std::endl;
    for(std::size_t i = 0; i < sample_count; ++i)
    {
        print_complex(d_A[i]);
        std::cout << ' ';
    }
    std::cout << std::endl;
    if(element_count > 0)
    {
        print_hex_bytes(&d_A[0], sizeof(hipFloatComplex));
    }

    std::cout << "Matrix B (input):" << std::endl;
    for(std::size_t i = 0; i < sample_count; ++i)
    {
        print_complex(d_B[i]);
        std::cout << ' ';
    }
    std::cout << std::endl;
    if(element_count > 0)
    {
        print_hex_bytes(&d_B[0], sizeof(hipFloatComplex));
    }

    std::cout << "Matrix C (input):" << std::endl;
    for(std::size_t i = 0; i < sample_count; ++i)
    {
        print_complex(d_C[i]);
        std::cout << ' ';
    }
    std::cout << std::endl;

    // No hipMemcpy: we wrote directly to device pointers above (unsafe, HMM-dependent).

    const hipFloatComplex alpha = hipFloatComplex{1.0f, 0.0f};
    const hipFloatComplex beta = hipFloatComplex{0.0f, 0.0f};

    // Prefetch managed memory to the active GPU to reduce initial page faults.
    int active_device = 0;
    CHECK_HIP_ERROR(hipGetDevice(&active_device));
    CHECK_HIP_ERROR(hipMemPrefetchAsync(d_A, bytes, active_device, 0));
    CHECK_HIP_ERROR(hipMemPrefetchAsync(d_B, bytes, active_device, 0));
    CHECK_HIP_ERROR(hipMemPrefetchAsync(d_C, bytes, active_device, 0));
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    const auto start_gemm = std::chrono::steady_clock::now();
    CHECK_ROCBLAS_STATUS(rocblas_cgemm(handle,
                                       rocblas_operation_none,
                                       rocblas_operation_none,
                                       m,
                                       n,
                                       k,
                                       &alpha,
                                       d_A,
                                       m,
                                       d_B,
                                       k,
                                       &beta,
                                       d_C,
                                       m));
    const auto end_gemm = std::chrono::steady_clock::now();

    const auto start_sync = std::chrono::steady_clock::now();
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    const auto end_sync = std::chrono::steady_clock::now();

    // Prefetch results to CPU for printing to avoid on-demand migration stalls.
    CHECK_HIP_ERROR(hipMemPrefetchAsync(d_C, bytes, hipCpuDeviceId, 0));
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    // Read results directly from managed memory (host-accessible).
    std::cout << "Matrix C (output):" << std::endl;
    for(std::size_t i = 0; i < sample_count; ++i)
    {
        print_complex(d_C[i]);
        std::cout << ' ';
    }
    std::cout << std::endl;

    const auto gemm_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end_gemm - start_gemm);
    const auto sync_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end_sync - start_sync);

    std::cout << "calc1:       " << gemm_duration.count() << " us" << std::endl;
    std::cout << "sync:        " << sync_duration.count() << " us" << std::endl;

    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_B));
    CHECK_HIP_ERROR(hipFree(d_C));
    CHECK_ROCBLAS_STATUS(rocblas_destroy_handle(handle));

    return EXIT_SUCCESS;
}
