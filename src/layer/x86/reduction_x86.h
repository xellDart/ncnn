// Copyright 2024 ncnn contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_REDUCTION_X86_H
#define LAYER_REDUCTION_X86_H

#include "reduction.h"

namespace ncnn {

class Reduction_x86 : public Reduction
{
public:
    Reduction_x86();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_REDUCTION_X86_H
