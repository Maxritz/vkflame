// rocblas_shim.cpp — minimal stub satisfying ggml-hip's rocblas_initialize import.
// ggml-hip.dll imports exactly ONE rocblas symbol; all GEMM goes through hipblas.
// We don't need any rocblas compute — just provide the symbol so the DLL loads.

extern "C"
{
    // rocblas_status_success = 0
    int rocblas_initialize()
    {
        return 0;
    }
} // extern "C"
