#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>
#include "aie_kernel_utils.h"

extern "C" {

void odd_even(int32_t * restrict input, int32_t * restrict input1,  int32_t * restrict value,const int32_t N) {
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
         AIE_LOOP_UNROLL_FULL
         for (int z = 0; z < 16; z++) {
            int32_t *__restrict input1v = input1;
            AIE_LOOP_UNROLL_FULL
           for (int j = 0; j < 4; j++) {

            aie::vector<int32_t, 16> A1 = aie::load_v<16>(input1v);
            auto mask =  aie::eq(A1,A0[z]);
            auto newvec = aie::select(-1,A1,mask);
            aie::store_v(valuev,newvec);
            valuev +=16;
            input1v += 16;
       }
       }
       inputv +=16;



}
event1();
}

} // extern "C"
