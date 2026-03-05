#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>
#include "aie_kernel_utils.h"


//===- passThrough.cc -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// #define __AIENGINE__ 1
#define NOCPP

#include <stdint.h>
#include <stdlib.h>



template <typename T, int N>
__attribute__((noinline)) void passThrough_aie(T *restrict in, T *restrict out,
                                               const int32_t height,
                                               const int32_t width) {
  event0();

  v64uint8 *restrict outPtr = (v64uint8 *)out;
  v64uint8 *restrict inPtr = (v64uint8 *)in;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(6)
  for (int j = 0; j < (height * width); j += N) // Nx samples per loop
  {
    *outPtr++ = *inPtr++;
  }

  event1();
}

extern "C" {



void passThroughLine(int32_t *in, int32_t *out, int32_t lineWidth) {
  passThrough_aie<int32_t, 16>(in, out, 1, lineWidth);
}

void passThroughTile(int32_t *in, int32_t *out, int32_t tileHeight,
                     int32_t tileWidth) {
  passThrough_aie<int32_t, 16>(in, out, tileHeight, tileWidth);
}



} // extern "C"




extern "C" {

void odd_even(
   int32_t *input,
   int32_t *input1,
   int32_t *object_fifo_buf,
   int32_t * join_count_fifo,
   int32_t *buf,
   int32_t * join_count_buf,
   int32_t N) {
  event0();



  int join_count = 0;
  for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        if(input[i] == input1[j]){
            object_fifo_buf[join_count] = input[i];

        }else{
            object_fifo_buf[join_count] = 0;
        }
        join_count++;

  }
  event1();
}
join_count_fifo[0] = join_count_fifo[0] + join_count;
//return 7;
}

} // extern "C"
