#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

extern "C" {

void odd_even(int32_t *input, int32_t *odd,  int32_t *even,int32_t N) {
  event0();
  int odd_count = 0;
  int even_count = 0;
  for (int i = 0; i < N; i++) {
    if(input[i] % 2){
        even[even_count] = input[i];
        even_count++;
    }else{
        odd[odd_count] = input[i];
        odd_count++;
    }

  }
  event1();
}

} // extern "C"
