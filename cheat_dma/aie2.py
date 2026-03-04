from pkgutil import extend_path

import numpy as np
import sys
import aie.utils.trace as trace_utils

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.types import i32
from aie.helpers.dialects.scf import _for as range_, if_, else_
from aie.extras.context import mlir_mod_ctx
from aie.extras.types import index

from aie.ir import *
from aie.dialects import aie, cf, scf, func, memref
from aie.extras.dialects import arith
from aie.dialects import cf
from setuptools.archive_util import extraction_drivers

from aie.dialects.scf import IfOp, ForOp, yield_

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

            data_ty_done = np.ndarray[(16,), np.dtype[np.int32]]

            data_ty_one_int = np.ndarray[(1,), np.dtype[np.int32]]

            # External, binary kernel definition
            odd_even = external_func(
                "odd_even",
                inputs=[tile_ty_in, tile_ty_in,tile_ty_out,data_ty_one_int,tile_ty_out,data_ty_one_int,np.int32]
            )

            # Tile declarations
            ShimTile00 = tile(0, 0)
            #ShimTile10 = tile(1, 0)
            ShimTile20 = tile(2, 0)
            #MemTile01 = tile(0, 1)
            #MemTile11 = tile(1, 1)
            ComputeTile02 = tile(0, 2)
            #ComputeTile12 = tile(1, 2)

            # AIE-array data movement with object fifos
            # Input
            #of_in = object_fifo("in", ShimTile00, MemTile01, 2, tile_ty)
            of_in1 = object_fifo("in1", ShimTile00, ComputeTile02, 2, tile_ty_in)
            of_in_inner = object_fifo("in1_inner", ShimTile00, ComputeTile02, 2, tile_ty_in)
            #object_fifo_link(of_in, of_in1)



            # Output

            of_out1 = object_fifo("out", ComputeTile02, ShimTile00, 2, tile_ty_out)

            of_done = object_fifo("outdone", ComputeTile02, ShimTile00, 2, data_ty_done)

            output_buffer = aie.buffer(
                tile=ComputeTile02,
                datatype=tile_ty_out,
                name=f"outputbuffer",
                initial_value=np.array(0, dtype=np.int32)
            )

            ty_one_int = np.ndarray[(1,), np.dtype[np.int32]]
            join_cnt_fifo = aie.buffer(
                tile=ComputeTile02,
                datatype=ty_one_int,
                name=f"join_cnt_fifo",
                initial_value=np.array(0, dtype=np.int32)
            )
            join_cnt_buffer = aie.buffer(
                tile=ComputeTile02,
                datatype=ty_one_int,
                name=f"join_cnt_buffer",
                initial_value=np.array(0, dtype=np.int32)
            )

            # Set up compute tiles
            # Compute tile
            @core(ComputeTile02, "odd_even.o")
            def core_body_02():
                for _ in range_(0xFFFFFFFF):

                    #stack allocation
                    #counter = memref.alloca([1], i32())
                    #probably needed
                    join_cnt_fifo[0] = 0
                    for _ in range_(iters_outer):
                        elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)

                        for i in range_(iters_inner,insert_yield=True):
                            elem_inner = of_in_inner.acquire(ObjectFifoPort.Consume, 1)

                            out = of_out1.acquire(ObjectFifoPort.Produce, 1)

                            #with if_(join_cnt_fifo[0]==0, hasElse=False) as if_op:
                            #    yield ()



                            cm1 = arith.constant(0)
                            join_cnt_buffer[0] = arith.constant(100)

                            init_running = arith.constant(1)
                            wh = scf.WhileOp([i32()],[init_running])
                            bf = wh.before.blocks.append(init_running.type)
                            af = wh.after.blocks.append(init_running.type)

                            with InsertionPoint(bf):
                                running = bf.arguments[0]

                                # condition: running != 0
                                cond = arith.cmpi("ne", running, arith.constant( 0))

                                # scf.condition returns the condition + loop-carried values
                                scf.condition(cond, [init_running])

                            with InsertionPoint(af):
                                #running = af.arguments[0]

                                # Acquire FIFO element


                                idx0 = arith.constant( 0,index=True)
                                val = memref.load(join_cnt_buffer, [idx0])

                                # Compute next running: break if sentinel -1
                                is_stop = arith.cmpi("eq", val, cm1)
                                next_running = arith.select(is_stop, arith.constant( 0), arith.constant(1))

                                join_cnt_buffer[0] = join_cnt_buffer[0] - 1

                                call(odd_even, [elem_in, elem_inner, out, join_cnt_fifo, output_buffer, join_cnt_buffer,
                                                tile_ty_size_in])

                                # Increment value and store
                                # dont know what this does
                                #result = arith.addi(val, c1)
                                #memref.store(result, elem, [idx0])

                                # Release FIFO


                                # Yield updated loop-carried values
                                scf.yield_([next_running])

                            of_in_inner.release(ObjectFifoPort.Consume, 1)
                            of_out1.release(ObjectFifoPort.Produce, 1)


                            # out = of_out1.acquire(ObjectFifoPort.Produce, 1)
                            #
                            # elem_inner = of_in_inner.acquire(ObjectFifoPort.Consume, 1)
                            # call(odd_even, [elem_in, elem_inner, out,join_cnt_fifo,output_buffer,join_cnt_buffer, tile_ty_size_in])
                            # of_in_inner.release(ObjectFifoPort.Consume, 1)
                            #
                            #
                            # of_out1.release(ObjectFifoPort.Produce, 1)



                        of_in1.release(ObjectFifoPort.Consume, 1)

                    elem_done = of_done.acquire(ObjectFifoPort.Produce,1)
                    for i in range_(16):
                        elem_done[i] = 5
                    of_done.release(ObjectFifoPort.Produce, 1)






            tiles_to_trace = [ComputeTile02, ShimTile00]
            if trace_size > 0:
                trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile20)
                #todo use other shimtile to trace?



            # To/from AIE-array data movement
            #d

            #tiles_to_trace = [ComputeTile02, MemTile01, ShimTile00]
            #if trace_size > 0:
            #    trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile20)

            @runtime_sequence(data_ty_in, data_ty_in,data_ty_out,data_ty_done)
            def sequence(inTensor,innerinTensor,outOddTensor,doneTensor,):

                if trace_size > 0:
                    trace_utils.configure_packet_tracing_aie2( #todo is this method correct form every npu?
                        tiles_to_trace=tiles_to_trace,
                        shim=ShimTile20,
                        ddr_id=4,# 4 -> group_id(7)
                        trace_size=trace_size,
                    )




                in_task = shim_dma_single_bd_task(of_in1, inTensor, offset= 0 ,sizes=[1, 1, 1, tranfer_size_elemnts_in],issue_token=False)
                out_task = shim_dma_single_bd_task(
                    of_out1, outOddTensor, offset=0, sizes=[1, 1, 1, tranfer_size_elemnts_out], issue_token=False
                )

                done_task = shim_dma_single_bd_task(
                    of_done, doneTensor, offset=0, sizes=[1, 1, 1, 16], issue_token=True ,burst_length=64
                )

                dma_start_task(in_task,out_task,done_task)

                for i in range(transfers_inner):
                    inner_in_task1 = shim_dma_single_bd_task(of_in_inner, innerinTensor, offset=0,
                                                      sizes=[1, 1, 1, tranfer_size_elemnts_in], issue_token=True)

                    dma_start_task(inner_in_task1)
                    dma_await_task(inner_in_task1)



                dma_await_task(done_task)
                dma_free_task(in_task)
                dma_free_task(out_task)

                #trace_utils.gen_trace_done_aie2(ShimTile20)




    res = True
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


external_mem_to_core()