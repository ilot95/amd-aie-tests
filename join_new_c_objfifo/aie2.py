from pkgutil import extend_path

import numpy as np
import sys
import aie.utils.trace as trace_utils

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_, if_, else_
from aie.extras.context import mlir_mod_ctx
from setuptools.archive_util import extraction_drivers

#use stderr so the mlir output does not break
#These are don't have to be errors
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

if len(sys.argv) > 1:
    if sys.argv[1] == "npu":
        dev = AIEDevice.npu1
    elif sys.argv[1] == "npu2":
        dev = AIEDevice.npu2
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

trace_size = 0
if len(sys.argv) > 2:
    if sys.argv[2].isdigit():
        trace_size = int(sys.argv[2])
        eprint("[INFO] trace_size: {}".format(trace_size))
    else:
        eprint("[Info] sys.argv[2] (trace_size):{} is not a positive number falling back to trace_size = 0".format(sys.argv[2]))

host_elements = 1024
if len(sys.argv) > 3:
    if sys.argv[3].isdigit():
        host_elements = int(sys.argv[3])
        eprint("[INFO] host_elements: {}".format(host_elements))
    else:
        eprint("[Info] sys.argv[3] (host_elements):{} is not a positive number falling back to host_elements = 1024".format(sys.argv[3]))



def external_mem_to_core():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():





            tranfer_size_elemnts_in = host_elements
            tranfer_size_elemnts_out = (host_elements*host_elements)


            eprint("[INFO] tranfer_size_elemnts_in: {}".format(tranfer_size_elemnts_in))
            eprint("[INFO] tranfer_size_elemnts_out: {}".format(tranfer_size_elemnts_out))


            #eprint("[INFO] transfer size in KB: {}".format(tranfer_size_elemnts_in*4/1024))


            #elements = 4096

            tile_ty_size_in = 64
            tile_ty_size_out = tile_ty_size_in * tile_ty_size_in

            eprint("[INFO] tile_ty_size_in: {}".format(tile_ty_size_in))
            eprint("[INFO] tile_ty_size_out: {}".format(tile_ty_size_out))

            #todo fix for A B different sizes
            iters_outer = host_elements // tile_ty_size_in
            #one relation needs to be pushed several times
            transfers_inner = iters_outer

            # todo fix for A B different sizes
            iters_inner = host_elements // tile_ty_size_in

            eprint("[INFO] iters_outer: {}".format(iters_outer))
            eprint("[INFO] iters_inner: {}".format(iters_inner))

            eprint("[INFO] transfers_inner: {}".format(transfers_inner))


            tile_ty_in = np.ndarray[(tile_ty_size_in,), np.dtype[np.int32]]
            tile_ty_out = np.ndarray[(tile_ty_size_out,), np.dtype[np.int32]]

            #buffer_ty = np.ndarray[(elements,), np.dtype[np.int32]]

            data_ty_in = np.ndarray[(tranfer_size_elemnts_in,), np.dtype[np.int32]]
            data_ty_out = np.ndarray[(tranfer_size_elemnts_out,), np.dtype[np.int32]]


            # External, binary kernel definition
            odd_even = external_func(
                "odd_even",
                inputs=[
                    tile_ty_in,  # in buffer 0
                    tile_ty_in,  # in buffer 1
                    tile_ty_in,  # in1 buffer 0
                    tile_ty_in,  # in1 buffer 1
                    tile_ty_out,  # out buffer 0
                    tile_ty_out,  # out buffer 1
                    T.index(),  # in acq_lock
                    T.index(),  # in rel_lock
                    T.index(),  # in1 acq_lock
                    T.index(),  # in1 rel_lock
                    T.index(),  # out acq_lock
                    T.index(),  # out rel_lock
                ]
            )

            # Tile declarations
            ShimTile00 = tile(0, 0)
            #ShimTile10 = tile(1, 0)
            ShimTile20 = tile(2, 0)
            MemTile01 = tile(0, 1)
            #MemTile11 = tile(1, 1)
            ComputeTile02 = tile(0, 2)
            #ComputeTile12 = tile(1, 2)

            # AIE-array data movement with object fifos
            # Input
            tile_ty = np.ndarray[(host_elements,), np.dtype[np.int32]]
            of_in_sh = object_fifo("in", ShimTile00, MemTile01, 2, tile_ty)
            of_in = object_fifo("in1", MemTile01, ComputeTile02, 2, tile_ty_in)
            object_fifo_link(of_in_sh, of_in)

            of_in_inner_sh = object_fifo("in_inner", ShimTile00, MemTile01, 2, tile_ty)
            of_in1 = object_fifo("in1_inner", MemTile01, ComputeTile02, 2, tile_ty_in)
            object_fifo_link(of_in_inner_sh, of_in1)



            # Output
            tile_ty_out_mem = np.ndarray[(tile_ty_size_out,), np.dtype[np.int32]]
            of_out = object_fifo("out", ComputeTile02, MemTile01, 2, tile_ty_out)
            of_out_sh = object_fifo("out1", MemTile01, ShimTile00, 2, tile_ty_out_mem)
            object_fifo_link( of_out,of_out_sh)





            # Set up compute tiles
            # Compute tile
            @core(ComputeTile02, "odd_even.o",dynamic_objfifo_lowering=False)
            def core_body_02():
                for _ in range_(0xFFFFFFFF):
                    in_buf0 = of_in.get_buffer(0)
                    in_buf1 = of_in.get_buffer(1)
                    in_acq, in_rel = of_in.get_lock(ObjectFifoPort.Consume)

                    in1_buf0 = of_in1.get_buffer(0)
                    in1_buf1 = of_in1.get_buffer(1)
                    in1_acq, in1_rel = of_in1.get_lock(ObjectFifoPort.Consume)

                    out_buf0 = of_out.get_buffer(0)
                    out_buf1 = of_out.get_buffer(1)
                    out_acq, out_rel = of_out.get_lock(ObjectFifoPort.Produce)

                    # C kernel owns the compute loop and buffer rotation
                    odd_even(
                        in_buf0,
                        in_buf1,
                        in1_buf0,
                        in1_buf1,
                        out_buf0,
                        out_buf1,
                        in_acq,
                        in_rel,
                        in1_acq,
                        in1_rel,
                        out_acq,
                        out_rel,
                    )







            tiles_to_trace = [ShimTile00,MemTile01,ComputeTile02 ]
            if trace_size > 0:
                trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile20)




            # To/from AIE-array data movement
            #d

            #tiles_to_trace = [ComputeTile02, MemTile01, ShimTile00]
            #if trace_size > 0:
            #    trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile20)

            @runtime_sequence(data_ty_in, data_ty_in,data_ty_out)
            def sequence(inTensor,innerinTensor,outOddTensor,):

                if trace_size > 0:
                    trace_utils.configure_packet_tracing_aie2( #todo is this method correct form every npu?
                        tiles_to_trace=tiles_to_trace,
                        shim=ShimTile20,
                        ddr_id=4,# 4 -> group_id(7)
                        trace_size=trace_size,
                    )




                in_task = shim_dma_single_bd_task(of_in_sh, inTensor, offset= 0 ,sizes=[1, 1, 1, tranfer_size_elemnts_in],issue_token=False)
                out_task = shim_dma_single_bd_task(
                    of_out_sh, outOddTensor, offset=0, sizes=[1, 1, 1, tranfer_size_elemnts_out], issue_token=True
                )

                dma_start_task(in_task,out_task)

                for i in range(transfers_inner):
                    inner_in_task1 = shim_dma_single_bd_task(of_in_inner_sh, innerinTensor, offset=0,
                                                      sizes=[1, 1, 1, tranfer_size_elemnts_in], issue_token=True)

                    dma_start_task(inner_in_task1)
                    dma_await_task(inner_in_task1)



                dma_await_task(out_task)
                dma_free_task(in_task)

                trace_utils.gen_trace_done_aie2(ShimTile20)





    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


external_mem_to_core()