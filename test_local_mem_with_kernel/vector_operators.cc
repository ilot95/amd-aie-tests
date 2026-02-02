#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

extern "C" {

void vector_plus_one(int32_t *a, int32_t *c, int32_t N) {
  event0();
  for (int i = 0; i < N; i++) {
    c[i] = a[i] + 1;
  }
  event1();
}

} // extern "C"
