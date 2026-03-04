#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>

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
