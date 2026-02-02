#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

extern "C" {

void odd_even(int32_t *input,int32_t *input1, int32_t *index,  int32_t *value,int32_t N) {
  event0();

  int join_count = 0;
  for (int i = 0; i < N; i++) {
  for (int j = 0; j < N; j++) {
    if(input[i] == input1[j]){
        value[join_count] = input[i];

        index[join_count] =i;
        join_count++;
    }
  }


  }
  event1();
}

} // extern "C"
