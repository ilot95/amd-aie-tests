#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>
#include "aie_kernel_utils.h"
#include "aie_objectfifo.h"

extern "C" {

void odd_even(
            int32_t *in_buf0, int32_t *in_buf1,
            int32_t *in1_buf0, int32_t *in1_buf1,
            int32_t *out_buf0,int32_t *out_buf1,
            int64_t in_acq_lock,int64_t in_rel_lock,
            int64_t in1_acq_lock,int64_t in1_rel_lock,
            int64_t out_acq_lock,int64_t out_rel_lock) {
  event0();
    objectfifo_t of_in = {(int32_t)in_acq_lock, (int32_t)in_rel_lock, -1, 1, 2,
                        {in_buf0, in_buf1}};
    objectfifo_t of_in1 = {(int32_t)in1_acq_lock, (int32_t)in1_rel_lock, -1, 1, 2,
                        {in1_buf0, in1_buf1}};
   objectfifo_t of_out = {(int32_t)out_acq_lock, (int32_t)out_rel_lock, -1, 1, 2,
                         {out_buf0, out_buf1}};
    //todo make this settable
    auto const iters_outer = 256;
    auto const iters_inner = 256;
    while(true){
            //for _ in range_(iters_outer):
            for (int i = 0; i < iters_outer; i++) {
                //elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)
                objectfifo_acquire(&of_in);
                int32_t *input = (int32_t *)objectfifo_get_buffer(&of_in, i);
                //for _ in range_(iters_inner):
                for (int j = 0; j < iters_inner; j++) {
                    //elem_inner = of_in_inner.acquire(ObjectFifoPort.Consume, 1)
                    //out = of_out1.acquire(ObjectFifoPort.Produce, 1)
                    objectfifo_acquire(&of_in1);
                    objectfifo_acquire(&of_out);
                    int32_t *input1 = (int32_t *)objectfifo_get_buffer(&of_in1, j);
                    int32_t *out = (int32_t *)objectfifo_get_buffer(&of_out, j);

                    int join_count = 0;
                    AIE_PREPARE_FOR_PIPELINING
                    AIE_LOOP_UNROLL(2)
                    for (int i = 0; i < 64; i++) {
                      AIE_LOOP_UNROLL_FULL
                      for (int j = 0; j < 64; j++) {
                        if(input[i] == input1[j]){
                            out[join_count] = input[i];

                        }else{
                            out[join_count] = -1;
                        }
                        join_count++;
                      }
                   }


                    //of_out1.release(ObjectFifoPort.Produce, 1)
                    //of_in_inner.release(ObjectFifoPort.Consume, 1)
                    objectfifo_release(&of_in1);
                    objectfifo_release(&of_out);
                }
                //of_in1.release(ObjectFifoPort.Consume, 1)
                objectfifo_release(&of_in);
            }






    }






event1();
}

} // extern "C"
