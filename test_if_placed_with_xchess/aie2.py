from pkgutil import extend_path

import numpy as np
import sys
import aie.utils.trace as trace_utils

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_

from aie.helpers.dialects.scf import if_,else_

from aie.extras.context import mlir_mod_ctx
from setuptools.archive_util import extraction_drivers

if len(sys.argv) > 1:
    if sys.argv[1] == "npu":
        dev = AIEDevice.npu1.npu1
    elif sys.argv[1] == "npu2":
        dev = AIEDevice.npu2
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

def external_mem_to_core():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            tile_ty = np.ndarray[(256,), np.dtype[np.int32]]
            trace_size = 8192

            # External, binary kernel definition
            vector_plus_one = external_func(
                "vector_plus_one",
                inputs=[tile_ty, tile_ty, np.int32]
            )

            # Tile declarations
            ShimTile00 = tile(0, 0)
            ShimTile10 = tile(1, 0)
            ShimTile20 = tile(2, 0)
            MemTile01 = tile(0, 1)
            MemTile11 = tile(1, 1)
            ComputeTile02 = tile(0, 2)
            ComputeTile12 = tile(1, 2)

            # AIE-array data movement with object fifos
            # Input
            of_in = object_fifo("in", ShimTile00, MemTile01, 2, tile_ty)
            of_in1 = object_fifo("in1", MemTile01, ComputeTile02, 2, tile_ty)
            object_fifo_link(of_in, of_in1)

            # Computation teil connector
            of_02_12 = object_fifo("_02to12", ComputeTile02, ComputeTile12, 2, tile_ty)

            # Output
            of_out1 = object_fifo("out1", ComputeTile12, MemTile11, 2, tile_ty)
            of_out = object_fifo("out", MemTile11, ShimTile10, 2, tile_ty)
            object_fifo_link(of_out1, of_out)

            # Set up compute tiles
            # Compute tile
            @core(ComputeTile02, "vector_operators.o")
            def core_body_02():
                # Effective while(1)
                for _ in range_(16):
                    elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_02_12.acquire(ObjectFifoPort.Produce, 1)
                    call(vector_plus_one, [elem_in, elem_out, 256])
                    of_in1.release(ObjectFifoPort.Consume, 1)
                    of_02_12.release(ObjectFifoPort.Produce, 1)

            # To/from AIE-array data movement
            data_ty = np.ndarray[(4096,), np.dtype[np.int32]]

            @core(ComputeTile12, "vector_operators.o")
            def core_body_12():
                # Effective while(1)
                for _ in range_(16):
                    elem_in = of_02_12.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out1.acquire(ObjectFifoPort.Produce, 1)

                    with if_(elem_in[0] % 2 == 0, hasElse=True) as if_op:
                        # pass

                        elem_out[0] = elem_in[0]

                    with else_(if_op):
                        elem_out[0] = elem_in[0]

                    call(vector_plus_one, [elem_in, elem_out, 256])



                    of_02_12.release(ObjectFifoPort.Consume, 1)
                    of_out1.release(ObjectFifoPort.Produce, 1)




            # To/from AIE-array data movement
            data_ty = np.ndarray[(4096,), np.dtype[np.int32]]

            tiles_to_trace = [ComputeTile02, MemTile01, ShimTile00]
            if trace_size > 0:
                trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile20)

            @runtime_sequence(data_ty, data_ty)
            def sequence(inTensor, outTensor):
                if trace_size > 0:
                    trace_utils.configure_packet_tracing_aie2(
                        tiles_to_trace=tiles_to_trace,
                        shim=ShimTile20,
                        trace_size=trace_size,
                    )
                npu_dma_memcpy_nd(
                    metadata=of_in, bd_id=1, mem=inTensor, sizes=[1, 1, 1, 4096]
                )
                npu_dma_memcpy_nd(
                    metadata=of_out, bd_id=0, mem=outTensor, sizes=[1, 1, 1, 4096]
                )
                # of_out will only complete after of_in completes, so we can just wait on of_out instead of both
                dma_wait(of_out)
                trace_utils.gen_trace_done_aie2(ShimTile20)

    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


external_mem_to_core()
