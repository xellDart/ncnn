// Copyright 2024 ncnn contributors
// SPDX-License-Identifier: BSD-3-Clause

#include "reduction_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif
#endif

#include <float.h>
#include <string.h>

namespace ncnn {

Reduction_x86::Reduction_x86()
{
#if __SSE2__
    support_packing = true;
#endif
}

// AVX-512/AVX/SSE vectorized column-wise sum reduction for 2D matrices.
// Reduces [w, h] -> [w] by summing along h dimension.
// Uses row-major sequential access (cache-friendly) instead of strided per-column access.
template<typename AccumOp>
static void reduction_reduce_h_sum_avx(const float* src, float* dst, int w, int h, int hstep)
{
    AccumOp op;
    memset(dst, 0, w * sizeof(float));

    for (int j = 0; j < h; j++)
    {
        const float* row = src + j * hstep;
        int i = 0;

#if __AVX__
#if __AVX512F__
        for (; i + 15 < w; i += 16)
        {
            __m512 _sum = _mm512_loadu_ps(dst + i);
            __m512 _val = _mm512_loadu_ps(row + i);
            _mm512_storeu_ps(dst + i, op.apply512(_sum, _val));
        }
#endif // __AVX512F__
        for (; i + 7 < w; i += 8)
        {
            __m256 _sum = _mm256_loadu_ps(dst + i);
            __m256 _val = _mm256_loadu_ps(row + i);
            _mm256_storeu_ps(dst + i, op.apply256(_sum, _val));
        }
#endif // __AVX__
#if __SSE2__
        for (; i + 3 < w; i += 4)
        {
            __m128 _sum = _mm_loadu_ps(dst + i);
            __m128 _val = _mm_loadu_ps(row + i);
            _mm_storeu_ps(dst + i, op.apply128(_sum, _val));
        }
#endif // __SSE2__
        for (; i < w; i++)
        {
            dst[i] = op.apply(dst[i], row[i]);
        }
    }
}


// Accumulator ops with SIMD variants
struct accum_add
{
    float apply(float a, float b) const { return a + b; }
#if __SSE2__
    __m128 apply128(__m128 a, __m128 b) const { return _mm_add_ps(a, b); }
#if __AVX__
    __m256 apply256(__m256 a, __m256 b) const { return _mm256_add_ps(a, b); }
#if __AVX512F__
    __m512 apply512(__m512 a, __m512 b) const { return _mm512_add_ps(a, b); }
#endif
#endif
#endif
};

struct accum_max
{
    float apply(float a, float b) const { return a > b ? a : b; }
#if __SSE2__
    __m128 apply128(__m128 a, __m128 b) const { return _mm_max_ps(a, b); }
#if __AVX__
    __m256 apply256(__m256 a, __m256 b) const { return _mm256_max_ps(a, b); }
#if __AVX512F__
    __m512 apply512(__m512 a, __m512 b) const { return _mm512_max_ps(a, b); }
#endif
#endif
#endif
};

struct accum_min
{
    float apply(float a, float b) const { return a < b ? a : b; }
#if __SSE2__
    __m128 apply128(__m128 a, __m128 b) const { return _mm_min_ps(a, b); }
#if __AVX__
    __m256 apply256(__m256 a, __m256 b) const { return _mm256_min_ps(a, b); }
#if __AVX512F__
    __m512 apply512(__m512 a, __m512 b) const { return _mm512_min_ps(a, b); }
#endif
#endif
#endif
};

struct accum_asum
{
    float apply(float a, float b) const { return a + fabsf(b); }
#if __SSE2__
    __m128 apply128(__m128 a, __m128 b) const
    {
        __m128 _mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
        return _mm_add_ps(a, _mm_and_ps(b, _mask));
    }
#if __AVX__
    __m256 apply256(__m256 a, __m256 b) const
    {
        __m256 _mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        return _mm256_add_ps(a, _mm256_and_ps(b, _mask));
    }
#if __AVX512F__
    __m512 apply512(__m512 a, __m512 b) const
    {
        __m512 _mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
        return _mm512_add_ps(a, _mm512_and_ps(b, _mask));
    }
#endif
#endif
#endif
};

struct accum_sumsq
{
    float apply(float a, float b) const { return a + b * b; }
#if __SSE2__
    __m128 apply128(__m128 a, __m128 b) const { return _mm_add_ps(a, _mm_mul_ps(b, b)); }
#if __AVX__
    __m256 apply256(__m256 a, __m256 b) const { return _mm256_add_ps(a, _mm256_mul_ps(b, b)); }
#if __AVX512F__
    __m512 apply512(__m512 a, __m512 b) const { return _mm512_add_ps(a, _mm512_mul_ps(b, b)); }
#endif
#endif
#endif
};

int Reduction_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;
    int axes_flag[4] = {0};
    bool reduce_w = false;
    bool reduce_h = false;
    bool reduce_d = false;
    bool reduce_c = false;

    if (reduce_all)
    {
        reduce_w = true;
        reduce_h = true;
        reduce_d = true;
        reduce_c = true;
    }
    else
    {
        const int* axes_ptr = axes;
        int reduced_axes_num = axes.w;

        for (int i = 0; i < reduced_axes_num; i++)
        {
            int axis = axes_ptr[i];
            if (axis < 0)
                axis += dims;
            axes_flag[axis] = 1;
        }

        if (dims == 1)
        {
            reduce_w = true;
        }
        else if (dims == 2)
        {
            if (axes_flag[0] == 1) reduce_h = true;
            if (axes_flag[1] == 1) reduce_w = true;
        }
        else if (dims == 3)
        {
            if (axes_flag[0] == 1) reduce_c = true;
            if (axes_flag[1] == 1) reduce_h = true;
            if (axes_flag[2] == 1) reduce_w = true;
        }
        else if (dims == 4)
        {
            if (axes_flag[0] == 1) reduce_c = true;
            if (axes_flag[1] == 1) reduce_d = true;
            if (axes_flag[2] == 1) reduce_h = true;
            if (axes_flag[3] == 1) reduce_w = true;
        }
    }

    // Check if we can handle this case with optimized SIMD path
    const int op = operation;
    const int elempack = bottom_blob.elempack;

    if (dims == 2 && !reduce_w && reduce_h
        && (op == ReductionOp_SUM || op == ReductionOp_MEAN))
    {
        // [w, h*elempack] -> [w] — HOT PATH
        // Native elempack support: no convert_packing needed
        // Cache-friendly: process rows sequentially, accumulate with SIMD
        const int w = bottom_blob.w;
        const int h = bottom_blob.h;
        const int real_h = h * elempack;
        const int row_stride = w * elempack; // floats per packed row

        if (keepdims)
            top_blob.create(w, 1, 4u, 1, opt.blob_allocator);
        else
            top_blob.create(w, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        float* outptr = (float*)top_blob;
        const float* srcptr = (const float*)bottom_blob;

        if (elempack == 1)
        {
            // Simple case: sequential rows, accumulate directly
            reduction_reduce_h_sum_avx<accum_add>(srcptr, outptr, w, h, w);
        }
        else
        {
            // elempack > 1: data layout is [h groups][w columns][elempack values]
            // Each position (g, c) stores elempack values from consecutive rows
            // We need to sum ALL values for each column c across all groups and elempack

            // Phase 1: accumulate across h groups into a temp buffer [w * elempack]
            // This is a sequential scan — very cache friendly
            Mat acc_buf(row_stride, 4u, 1, opt.workspace_allocator);
            if (acc_buf.empty())
                return -100;
            float* acc = (float*)acc_buf;
            memset(acc, 0, row_stride * sizeof(float));

            for (int g = 0; g < h; g++)
            {
                const float* row = srcptr + g * row_stride;
                int i = 0;
#if __AVX__
#if __AVX512F__
                for (; i + 15 < row_stride; i += 16)
                {
                    __m512 _s = _mm512_loadu_ps(acc + i);
                    __m512 _v = _mm512_loadu_ps(row + i);
                    _mm512_storeu_ps(acc + i, _mm512_add_ps(_s, _v));
                }
#endif
                for (; i + 7 < row_stride; i += 8)
                {
                    __m256 _s = _mm256_loadu_ps(acc + i);
                    __m256 _v = _mm256_loadu_ps(row + i);
                    _mm256_storeu_ps(acc + i, _mm256_add_ps(_s, _v));
                }
#endif
#if __SSE2__
                for (; i + 3 < row_stride; i += 4)
                {
                    __m128 _s = _mm_loadu_ps(acc + i);
                    __m128 _v = _mm_loadu_ps(row + i);
                    _mm_storeu_ps(acc + i, _mm_add_ps(_s, _v));
                }
#endif
                for (; i < row_stride; i++)
                    acc[i] += row[i];
            }

            // Phase 2: horizontal sum each group of elempack values → one output per column
            if (elempack == 8)
            {
                for (int c = 0; c < w; c++)
                {
#if __AVX__
                    __m256 _v = _mm256_loadu_ps(acc + c * 8);
                    // horizontal sum: 8 → 1
                    __m128 _lo = _mm256_castps256_ps128(_v);
                    __m128 _hi = _mm256_extractf128_ps(_v, 1);
                    __m128 _sum = _mm_add_ps(_lo, _hi);
                    _sum = _mm_add_ps(_sum, _mm_movehl_ps(_sum, _sum));
                    _sum = _mm_add_ps(_sum, _mm_shuffle_ps(_sum, _sum, 1));
                    outptr[c] = _mm_cvtss_f32(_sum);
#else
                    float s = 0;
                    for (int e = 0; e < 8; e++)
                        s += acc[c * 8 + e];
                    outptr[c] = s;
#endif
                }
            }
            else if (elempack == 4)
            {
                for (int c = 0; c < w; c++)
                {
#if __SSE2__
                    __m128 _v = _mm_loadu_ps(acc + c * 4);
                    _v = _mm_add_ps(_v, _mm_movehl_ps(_v, _v));
                    _v = _mm_add_ps(_v, _mm_shuffle_ps(_v, _v, 1));
                    outptr[c] = _mm_cvtss_f32(_v);
#else
                    float s = 0;
                    for (int e = 0; e < 4; e++)
                        s += acc[c * 4 + e];
                    outptr[c] = s;
#endif
                }
            }
            else
            {
                for (int c = 0; c < w; c++)
                {
                    float s = 0;
                    for (int e = 0; e < elempack; e++)
                        s += acc[c * elempack + e];
                    outptr[c] = s;
                }
            }
        }

        // Post-processing for MEAN
        if (op == ReductionOp_MEAN)
        {
            float scale = 1.f / real_h;
            int i = 0;
#if __AVX__
#if __AVX512F__
            __m512 _scale512 = _mm512_set1_ps(scale);
            for (; i + 15 < w; i += 16)
                _mm512_storeu_ps(outptr + i, _mm512_mul_ps(_mm512_loadu_ps(outptr + i), _scale512));
#endif
            __m256 _scale256 = _mm256_set1_ps(scale);
            for (; i + 7 < w; i += 8)
                _mm256_storeu_ps(outptr + i, _mm256_mul_ps(_mm256_loadu_ps(outptr + i), _scale256));
#endif
            for (; i < w; i++)
                outptr[i] *= scale;
        }

        if (coeff != 1.f)
        {
            for (int i = 0; i < w; i++)
                outptr[i] *= coeff;
        }

        return 0;
    }

    // Flatten elempack for remaining optimized paths
    Mat a = bottom_blob;
    if (a.elempack != 1)
    {
        Mat a_flat;
        convert_packing(a, a_flat, 1, opt);
        a = a_flat;
    }

    bool handled = false;

    if (dims == 2 && reduce_w && !reduce_h)
    {
        // [w, h] -> [h] — row-wise reduction with SIMD horizontal sum
        const int w = a.w;
        const int h = a.h;

        if (keepdims)
            top_blob.create(1, h, 4u, 1, opt.blob_allocator);
        else
            top_blob.create(h, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        float* outptr = (float*)top_blob;
        const float* srcptr = (const float*)a;

        if (op == ReductionOp_SUM || op == ReductionOp_MEAN)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = 0; j < h; j++)
            {
                const float* row = srcptr + j * w;
                float sum = 0.f;
                int i = 0;
#if __AVX__
#if __AVX512F__
                __m512 _sum512 = _mm512_setzero_ps();
                for (; i + 15 < w; i += 16)
                    _sum512 = _mm512_add_ps(_sum512, _mm512_loadu_ps(row + i));
                sum += _mm512_reduce_add_ps(_sum512);
#else
                __m256 _sum256 = _mm256_setzero_ps();
                for (; i + 7 < w; i += 8)
                    _sum256 = _mm256_add_ps(_sum256, _mm256_loadu_ps(row + i));
                __m128 _lo = _mm256_castps256_ps128(_sum256);
                __m128 _hi = _mm256_extractf128_ps(_sum256, 1);
                __m128 _sum128 = _mm_add_ps(_lo, _hi);
                _sum128 = _mm_add_ps(_sum128, _mm_movehl_ps(_sum128, _sum128));
                _sum128 = _mm_add_ps(_sum128, _mm_shuffle_ps(_sum128, _sum128, 1));
                sum += _mm_cvtss_f32(_sum128);
#endif
#elif __SSE2__
                __m128 _sum128 = _mm_setzero_ps();
                for (; i + 3 < w; i += 4)
                    _sum128 = _mm_add_ps(_sum128, _mm_loadu_ps(row + i));
                _sum128 = _mm_add_ps(_sum128, _mm_movehl_ps(_sum128, _sum128));
                _sum128 = _mm_add_ps(_sum128, _mm_shuffle_ps(_sum128, _sum128, 1));
                sum += _mm_cvtss_f32(_sum128);
#endif
                for (; i < w; i++)
                    sum += row[i];

                outptr[j] = (op == ReductionOp_MEAN) ? sum / w : sum;
            }
            handled = true;
        }

        if (handled)
        {
            if (coeff != 1.f)
            {
                for (int i = 0; i < h; i++)
                    outptr[i] *= coeff;
            }
            return 0;
        }
    }

    if (dims == 2 && reduce_w && reduce_h)
    {
        // [w, h] -> scalar
        const int w = a.w;
        const int h = a.h;

        if (keepdims)
            top_blob.create(1, 1, 4u, 1, opt.blob_allocator);
        else
            top_blob.create(1, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (op == ReductionOp_SUM || op == ReductionOp_MEAN)
        {
            float total = 0.f;
            const float* srcptr = (const float*)a;
            const int total_size = w * h;
            int i = 0;
#if __AVX__
#if __AVX512F__
            __m512 _sum512 = _mm512_setzero_ps();
            for (; i + 15 < total_size; i += 16)
                _sum512 = _mm512_add_ps(_sum512, _mm512_loadu_ps(srcptr + i));
            total += _mm512_reduce_add_ps(_sum512);
#endif
#endif
            for (; i < total_size; i++)
                total += srcptr[i];

            if (op == ReductionOp_MEAN)
                total /= (w * h);

            ((float*)top_blob)[0] = total * coeff;
            return 0;
        }
    }

    if (dims == 1)
    {
        // [w] -> scalar — SIMD horizontal reduction
        const int w = a.w;

        top_blob.create(1, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (op == ReductionOp_SUM || op == ReductionOp_MEAN)
        {
            float sum = 0.f;
            const float* ptr = (const float*)a;
            int i = 0;
#if __AVX__
#if __AVX512F__
            __m512 _sum512 = _mm512_setzero_ps();
            for (; i + 15 < w; i += 16)
                _sum512 = _mm512_add_ps(_sum512, _mm512_loadu_ps(ptr + i));
            sum += _mm512_reduce_add_ps(_sum512);
#endif
#endif
            for (; i < w; i++)
                sum += ptr[i];

            ((float*)top_blob)[0] = (op == ReductionOp_MEAN ? sum / w : sum) * coeff;
            return 0;
        }
    }

    // Fallback: delegate to generic Reduction implementation
    return Reduction::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
