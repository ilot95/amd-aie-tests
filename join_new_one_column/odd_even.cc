#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>
#include "aie_kernel_utils.h"

extern "C" {

void odd_even(int32_t *input, int32_t * restrict input1,  int32_t * restrict value,const int32_t N) {
  event0();



   int join_count = 0;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_UNROLL(2)
  for (int i = 0; i < 64; i++) {

      AIE_LOOP_UNROLL_FULL
      for (int j = 0; j < 64; j++) {
        if(input[i] == input1[j]){
            value[join_count] = input[i];

        }else{
            value[join_count] = -1;
        }
        join_count++;
  }
  event1();
}
}

} // extern "C"
