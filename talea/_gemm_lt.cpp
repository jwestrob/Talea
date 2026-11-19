// Compare cuBLASLt's alternative kernels with a stock PyTorch linear layer.
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <new>

struct Plan {
    cublasLtHandle_t handle{};
    cublasLtMatmulDesc_t op{};
    cublasLtMatrixLayout_t weight{}, inputs{}, output{};
    cublasLtMatmulPreference_t prefs{};
    cublasLtMatmulHeuristicResult_t algorithms[16]{};
    int count{};
    size_t workspace{};
    ~Plan() {
        if (prefs) cublasLtMatmulPreferenceDestroy(prefs);
        if (output) cublasLtMatrixLayoutDestroy(output);
        if (inputs) cublasLtMatrixLayoutDestroy(inputs);
        if (weight) cublasLtMatrixLayoutDestroy(weight);
        if (op) cublasLtMatmulDescDestroy(op);
        if (handle) cublasLtDestroy(handle);
    }
};

extern "C" int setup(int rows, int inner, int cols, size_t workspace_bytes, Plan **out) {
    Plan *p = new (std::nothrow) Plan;
    if (!p) return -100;
    *out = p;
    p->workspace = workspace_bytes;
    // The row-major F.linear result Y[rows,cols] is the column-major
    // transpose W^T[cols,inner] * X[inner,rows].
    const cublasOperation_t transpose = CUBLAS_OP_T;
#define CHECK(expr) { auto s = (expr); if (s != CUBLAS_STATUS_SUCCESS) return -int(s); }
    CHECK(cublasLtCreate(&p->handle));
    CHECK(cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
    CHECK(cublasLtMatrixLayoutCreate(&p->weight, CUDA_R_16F, inner, cols, inner));
    CHECK(cublasLtMatrixLayoutCreate(&p->inputs, CUDA_R_16F, inner, rows, inner));
    CHECK(cublasLtMatrixLayoutCreate(&p->output, CUDA_R_16F, cols, rows, cols));
    CHECK(cublasLtMatmulPreferenceCreate(&p->prefs));
    CHECK(cublasLtMatmulPreferenceSetAttribute(p->prefs, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &workspace_bytes, sizeof(workspace_bytes)));
    CHECK(cublasLtMatmulAlgoGetHeuristic(p->handle, p->op, p->weight, p->inputs, p->output,
                                         p->output, p->prefs, 16, p->algorithms, &p->count));
    return p->count;
}
extern "C" int execute(Plan *p, int index, const void *w, const void *x, void *y, void *workspace, cudaStream_t stream) {
    if (index < 0 || index >= p->count) return -1;
    float alpha=1.0f, beta=0.0f;
    return int(cublasLtMatmul(p->handle, p->op, &alpha, w, p->weight, x, p->inputs, &beta, y, p->output,
                              y, p->output, &p->algorithms[index].algo, workspace, p->workspace, stream));
}
extern "C" void details(Plan *p, int index, int *out) {
    auto *algo = &p->algorithms[index].algo;
    cublasLtMatmulAlgoConfigAttributes_t attributes[] = {CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID,
        CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CUBLASLT_ALGO_CONFIG_STAGES_ID, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION};
    for (int i=0; i<7; ++i) {
        out[i] = -1;
        size_t written=0;
        if (cublasLtMatmulAlgoConfigGetAttribute(algo, attributes[i], out+i, sizeof(*out), &written) != CUBLAS_STATUS_SUCCESS || written != sizeof(*out)) out[i] = -1;
    }
}
extern "C" void close_plan(Plan *p) { delete p; }
