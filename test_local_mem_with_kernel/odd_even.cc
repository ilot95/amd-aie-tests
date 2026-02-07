#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>

extern "C" {

void odd_even(int32_t *input, int32_t *odd,  int32_t *even,int32_t N) {
  event0();
  int odd_count = 0;
  int even_count = 0;
  //This might work
  //::acquire_equal(0,1);
  //::release(0,1);
  //This works shift by 16 to get value
  int32_t core_id = ::get_coreid();

  //This should work too
  aie::vector<int16, 32> v;
  v[0]= 444;
  v[1]= 555;
  v[2]= 666;
  v[3]= 777;
  v[4]= 888;
  v[5]= 999;
  //linking fails for this as there is no printt in kernel
  //aie::print(v,true,"Hello" );
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
