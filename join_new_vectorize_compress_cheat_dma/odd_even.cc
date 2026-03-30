#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>
#include "aie_kernel_utils.h"



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


void odd_even(int32_t * restrict input, int32_t * restrict input1,  int32_t * restrict value,const int32_t N,int32_t * restrict elems_produced) {
  event0();



   int join_count = 0;


   int32_t *__restrict valuev = value;

   int32_t *__restrict inputv = input;

  AIE_PREPARE_FOR_PIPELINING
  //AIE_LOOP_UNROLL(2)
  AIE_LOOP_UNROLL_FULL
  for (int i = 0; i < 4; i++) {
        aie::vector<int32_t, 16> A0 = aie::load_v<16>(inputv);
       //
         //AIE_LOOP_UNROLL_FULL
         //AIE_LOOP_UNROLL(2)
         for (int z = 0; z < 16; z++) {
            int32_t *__restrict input1v = input1;
            AIE_LOOP_UNROLL_FULL
           for (int j = 0; j < 4; j++) {

            aie::vector<int32_t, 16> A1 = aie::load_v<16>(input1v);
            auto mask =  aie::eq(A1,A0[z]);


            aie::vector<int32_t, 16> comp_vec = aie::broadcast(-1);
            int k = 0;
            AIE_LOOP_UNROLL_FULL
            for (int t = 0; t < 16; ++t) {
                /*if (mask.test(t)) {
                    comp_vec[k] = A1[t];
                    k++;
                }*/
                comp_vec[k] = mask.test(t) ? A1[t] : -1 ;
                k = k + mask.test(t);
            }
            aie::store_unaligned_v(valuev,comp_vec);
            //aie::store_v(valuev,comp_vec);
            //auto newvec = aie::select(-1,A1,mask);
            //aie::store_v(valuev,newvec);
            valuev +=k;

            join_count +=k;

            input1v += 16;
       }
       }
       inputv +=16;

}
//todo vectorize this
 for (auto vv = valuev; vv < value + 4096;vv++) {
    *vv= -1;
 }
 //*elems_produced = value + 4096 - valuev;
 *elems_produced = join_count;


event1();
}

} // extern "C"
