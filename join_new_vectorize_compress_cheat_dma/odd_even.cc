#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>
#include "aie_kernel_utils.h"
#include "aie_objectfifo.h"



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


void writeout(
            int32_t * restrict in_buf0, int32_t * restrict in_buf1,
            int32_t * restrict in_of_numer0, int32_t * in_of_numer1,
            int32_t * restrict out_buf0,int32_t * restrict out_buf1,
            int64_t in_acq_lock,int64_t in_rel_lock,
            int64_t in_of_numer_acq_lock,int64_t in_of_numer_rel_lock,
            int64_t out_acq_lock, int64_t out_rel_lock,
            int32_t * restrict elems_produced,
            const int32_t iters_outer,
            const int32_t iters_inner
            ) {
            *elems_produced =0;

            objectfifo_t of_in = {(int32_t)in_acq_lock, (int32_t)in_rel_lock, -1, 1, 2,
                                {in_buf0, in_buf1}};
            objectfifo_t of_in_of_numer = {(int32_t)in_of_numer_acq_lock, (int32_t)in_of_numer_rel_lock, -1, 1, 2,
                                {in_of_numer0, in_of_numer1}};

            objectfifo_t of_out = {(int32_t)out_acq_lock, (int32_t)out_rel_lock, -1, 1, 2,
                                 {out_buf0, out_buf1}};
            event0();

            objectfifo_acquire(&of_out);
            int32_t *out = (int32_t *)objectfifo_get_buffer(&of_out, 0);
            int freeOutBuf = 4096;
            int outCount = 0;
            int count_out_ac = 1;

            //262144
            //for (int i = 0; i < 65536; i++) {
            //todo why are two loops not possible
            for (int64_t i = 0; i < ((int64_t)iters_outer)*(int64_t)iters_inner; i++) {
            //for (int i = 0; i < 512; i++) {
            //for (int z = 0; z < 512; z++) {
                objectfifo_acquire(&of_in);
                int32_t *input = (int32_t *)objectfifo_get_buffer(&of_in, i);

                objectfifo_acquire(&of_in_of_numer);
                int32_t *numer_el = (int32_t *)objectfifo_get_buffer(&of_in_of_numer, i);
                *elems_produced += *numer_el;

                auto to_copy = std::min(*numer_el,freeOutBuf);



              for (int j = 0; j < to_copy; j += 1) // Nx samples per loop
              {
                out[j+outCount] = input[j];
              }
              freeOutBuf = freeOutBuf - to_copy;
              outCount = outCount + to_copy;

              if(freeOutBuf == 0){

                objectfifo_release(&of_out);
                objectfifo_acquire(&of_out);
                out = (int32_t *)objectfifo_get_buffer(&of_out, count_out_ac);
                count_out_ac ++;

                freeOutBuf = 4096;
                outCount =0;
                for (int j = 0; j < ((*numer_el) - to_copy); j += 1) // Nx samples per loop
                {
                out[j] = input[j+to_copy];
                }
                freeOutBuf = freeOutBuf -((*numer_el) - to_copy);
                outCount = outCount + ((*numer_el) - to_copy);
              }


                objectfifo_release(&of_in_of_numer);
                objectfifo_release(&of_in);

            }//}
            for (int j = outCount; j < 4096; j += 1){
            out[j] = -1;
            }
            objectfifo_release(&of_out);
             event1();
         }


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
