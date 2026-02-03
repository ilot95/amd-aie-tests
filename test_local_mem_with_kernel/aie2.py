from pkgutil import extend_path

import numpy as np
import sys
import aie.utils.trace as trace_utils

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_, if_, else_
from aie.extras.context import mlir_mod_ctx
from setuptools.archive_util import extraction_drivers

if len(sys.argv) > 1:
    if sys.argv[1] == "npu":
        dev = AIEDevice.npu1
    elif sys.argv[1] == "npu2":
        dev = AIEDevice.npu2
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

def external_mem_to_core():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            elements = 64
            tile_ty_size = 1
            iters = elements // tile_ty_size


            tile_ty = np.ndarray[(tile_ty_size,), np.dtype[np.int32]]

            buffer_ty = np.ndarray[(elements,), np.dtype[np.int32]]
            #trace_size = 8192

            # External, binary kernel definition
            odd_even = external_func(
                "odd_even",
                inputs=[buffer_ty, buffer_ty,buffer_ty, np.int32]
            )

            # Tile declarations
            ShimTile00 = tile(0, 0)
            #ShimTile10 = tile(1, 0)
            #ShimTile20 = tile(2, 0)
            #MemTile01 = tile(0, 1)
            #MemTile11 = tile(1, 1)
            ComputeTile02 = tile(0, 2)
            ComputeTile12 = tile(1, 2)

            # AIE-array data movement with object fifos
            # Input
            #of_in = object_fifo("in", ShimTile00, MemTile01, 2, tile_ty)
            of_in1 = object_fifo("in1", ShimTile00, ComputeTile02, 2, tile_ty)
            #object_fifo_link(of_in, of_in1)



            # Output
            #of_out1 = object_fifo("out1", ComputeTile02, MemTile01, 2, tile_ty)
            of_out1 = object_fifo("out", ComputeTile02, ShimTile00, 2, tile_ty)
            #object_fifo_link(of_out1, of_out)

            #of_out1_odd = object_fifo("outodd", ComputeTile02, MemTile01, 2, tile_ty)
            of_out1_odd = object_fifo("odd", ComputeTile02, ShimTile00, 2, tile_ty)
            #object_fifo_link(of_out1_odd, of_out_odd)


            even_buffer = aie.buffer(
                tile=ComputeTile02,
                datatype=np.ndarray[(elements,), np.dtype[np.int32]],
                name=f"evenbuffer",
                initial_value=np.array(0, dtype=np.int32)
            )
            odd_buffer = aie.buffer(
                tile=ComputeTile02,
                datatype=np.ndarray[(elements,), np.dtype[np.int32]],
                name=f"oddbuffer",
                initial_value=np.array(0, dtype=np.int32)
            )
            input_buffer = aie.buffer(
                tile=ComputeTile02,
                datatype=np.ndarray[(elements,), np.dtype[np.int32]],
                name=f"inputbuffer0",
                initial_value=np.array(0, dtype=np.int32)
            )


            # Set up compute tiles
            # Compute tile
            @core(ComputeTile02, "odd_even.o")
            def core_body_02():
                for i in range_(iters):
                    elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)

                    input_buffer[i] = elem_in[0]

                    of_in1.release(ObjectFifoPort.Consume, 1)

                call(odd_even, [input_buffer, odd_buffer,even_buffer, elements])

                for i in range_(iters):

                    elemOut_even = of_out1.acquire(ObjectFifoPort.Produce, 1)
                    elemOut_odd = of_out1_odd.acquire(ObjectFifoPort.Produce, 1)

                    elemOut_even[0] = even_buffer[i]
                    elemOut_odd[0] = odd_buffer[i]

                    of_out1_odd.release(ObjectFifoPort.Produce, 1)
                    of_out1.release(ObjectFifoPort.Produce, 1)






            # To/from AIE-array data movement
            data_ty = np.ndarray[(elements,), np.dtype[np.int32]]

            #tiles_to_trace = [ComputeTile02, MemTile01, ShimTile00]
            #if trace_size > 0:
            #    trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile20)

            @runtime_sequence(data_ty, data_ty,data_ty)
            def sequence(inTensor,outOddTensor, outTensor,):


                # npu_dma_memcpy_nd(
                #     metadata=of_in, bd_id=2, mem=inTensor, sizes=[1, 1, 1, elements],issue_token=True
                # )
                # npu_dma_memcpy_nd(
                #     metadata=of_out_odd, bd_id=1, mem=outOddTensor, sizes=[1, 1, 1, elements],issue_token=True
                # )
                # npu_dma_memcpy_nd(
                #     metadata=of_out, bd_id=0, mem=outTensor, sizes=[1, 1, 1, elements],issue_token=True
                # )
                # dma_wait(of_out,of_out_odd,of_in)
                in_task = shim_dma_single_bd_task(of_in1, inTensor, sizes=[1, 1, 1, elements])
                out_task = shim_dma_single_bd_task(
                    of_out1_odd, outOddTensor, sizes=[1, 1, 1, elements], issue_token=True
                )
                out_task1 = shim_dma_single_bd_task(
                    of_out1, outTensor, sizes=[1, 1, 1, elements], issue_token=True
                )

                dma_start_task(in_task, out_task,out_task1)
                dma_await_task(out_task,out_task1)
                dma_free_task(in_task)



    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


external_mem_to_core()
