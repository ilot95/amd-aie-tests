from pkgutil import extend_path

import numpy as np
import sys
import aie.utils.trace as trace_utils

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.types import i32
from aie.extras.types import memref
import aie.extras.types as types
from aie.helpers.dialects.scf import _for as range_, if_, else_
from aie.extras.context import mlir_mod_ctx
from aie.extras.types import index

from aie.ir import *
from aie.dialects import aie, cf, scf, func, memref
from aie.extras.dialects import arith
from aie.dialects import cf
from setuptools.archive_util import extraction_drivers

from aie.dialects.scf import IfOp, ForOp, yield_


# use stderr so the mlir output does not break
# These are don't have to be errors
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
        eprint("[Info] sys.argv[2] (trace_size):{} is not a positive number falling back to trace_size = 0".format(
            sys.argv[2]))

host_elements = 1024
if len(sys.argv) > 3:
    if sys.argv[3].isdigit():
        host_elements = int(sys.argv[3])
        eprint("[INFO] host_elements: {}".format(host_elements))
    else:
        eprint(
            "[Info] sys.argv[3] (host_elements):{} is not a positive number falling back to host_elements = 1024".format(
                sys.argv[3]))


def external_mem_to_core():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():

            tranfer_size_elemnts_in = host_elements
            tranfer_size_elemnts_out = (host_elements * host_elements)

            eprint("[INFO] tranfer_size_elemnts_in: {}".format(tranfer_size_elemnts_in))
            eprint("[INFO] tranfer_size_elemnts_out: {}".format(tranfer_size_elemnts_out))

            # eprint("[INFO] transfer size in KB: {}".format(tranfer_size_elemnts_in*4/1024))

            # elements = 4096

            tile_ty_size_in = 64
            tile_ty_size_out = tile_ty_size_in*tile_ty_size_in

            max_outer_aquire = tranfer_size_elemnts_out // tile_ty_size_out
            eprint("[INFO] max_outer_aquire: {}".format(max_outer_aquire))


            eprint("[INFO] tile_ty_size_in: {}".format(tile_ty_size_in))
            eprint("[INFO] tile_ty_size_out: {}".format(tile_ty_size_out))

            # todo fix for A B different sizes
            iters_outer = host_elements // tile_ty_size_in
            # one relation needs to be pushed several times
            transfers_inner = iters_outer

            # todo fix for A B different sizes
            iters_inner = host_elements // tile_ty_size_in

            eprint("[INFO] iters_outer: {}".format(iters_outer))
            eprint("[INFO] iters_inner: {}".format(iters_inner))

            eprint("[INFO] transfers_inner: {}".format(transfers_inner))

            tile_ty_in = np.ndarray[(tile_ty_size_in,), np.dtype[np.int32]]
            tile_ty_out = np.ndarray[(tile_ty_size_out,), np.dtype[np.int32]]

            # buffer_ty = np.ndarray[(elements,), np.dtype[np.int32]]

            data_ty_in = np.ndarray[(tranfer_size_elemnts_in,), np.dtype[np.int32]]
            data_ty_out = np.ndarray[(tranfer_size_elemnts_out,), np.dtype[np.int32]]

            data_ty_done = np.ndarray[(16,), np.dtype[np.int32]]

            data_ty_one_int = np.ndarray[(1,), np.dtype[np.int32]]

            # External, binary kernel definition
            odd_even = external_func(
                "odd_even",
                inputs=[tile_ty_in, tile_ty_in, tile_ty_out, data_ty_one_int, tile_ty_out, data_ty_one_int, np.int32]
            )

            passThroughLine = external_func(
                "passThroughLine",
                inputs=[tile_ty_out, tile_ty_out, np.int32]
            )

            # Tile declarations
            ShimTile00 = tile(0, 0)
            # ShimTile10 = tile(1, 0)
            ShimTile20 = tile(2, 0)
            # MemTile01 = tile(0, 1)
            # MemTile11 = tile(1, 1)
            ComputeTile02 = tile(0, 2)
            # ComputeTile12 = tile(1, 2)

            # AIE-array data movement with object fifos
            # Input
            # of_in = object_fifo("in", ShimTile00, MemTile01, 2, tile_ty)
            of_in1 = object_fifo("in1", ShimTile00, ComputeTile02, 2, tile_ty_in)
            of_in_inner = object_fifo("in1_inner", ShimTile00, ComputeTile02, 2, tile_ty_in)
            # object_fifo_link(of_in, of_in1)

            # Output

            of_out1 = object_fifo("out", ComputeTile02, ShimTile00, 2, tile_ty_out)

            of_done = object_fifo("outdone", ComputeTile02, ShimTile00, 2, data_ty_done)

            # output_buffer = aie.buffer(
            #     tile=ComputeTile02,
            #     datatype=tile_ty_out,
            #     name=f"outputbuffer",
            #     initial_value=np.array(0, dtype=np.int32)
            # )

            ty_one_int = np.ndarray[(1,), np.dtype[np.int32]]
            one_index_int_memref_type = types.memref(1, IndexType.get())
            # join_cnt = aie.buffer(
            #     tile=ComputeTile02,
            #     datatype=one_index_int_memref_type,
            #     name=f"join_cnt",
            #     #initial_value=np.array(0, dtype=np.int32)
            # )
            # aquire_out_cnt = aie.buffer(
            #     tile=ComputeTile02,
            #     datatype=ty_one_int,
            #     name=f"global_join_cnt",
            #     initial_value=np.array(0, dtype=np.int32)
            # )

            # Set up compute tiles
            # Compute tile
            @core(ComputeTile02, "odd_even.o",dynamic_objfifo_lowering=True)
            def core_body_02():
                for _ in range_(0xFFFFFFFF):

                    # stack allocation
                    # counter = memref.alloca([1], i32())
                    # probably needed
                    #memref_type for output fifo
                    elem_memref_type = types.memref(tile_ty_size_out, i32())
                    eprint(elem_memref_type)
                    join_cnt_init = arith.constant(0, type=i32(),index=True)

                    firstout = of_out1.acquire(ObjectFifoPort.Produce, 1)

                    aquire_out_cnt_init = arith.constant(1, type=i32())

                    # only for safety
                    #for z in range_(0, tile_ty_size_out, 1):
                    #    firstout[z] = 0

                    for _,(current_out_outer,jc_o,ac_o),final_final_out in range_(iters_outer,iter_args=[firstout,join_cnt_init,aquire_out_cnt_init],insert_yield=False):
                        elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)


                        for _, (current_out,jc_i,ac_i), final_out in range_(iters_inner, iter_args=[current_out_outer,jc_o,ac_o], insert_yield=False):
                            elem_inner = of_in_inner.acquire(ObjectFifoPort.Consume, 1)

                            init_i = arith.constant(0, index=True)
                            init_j = arith.constant(0, index=True)

                            wh = scf.WhileOp([init_i.type, init_j.type, elem_memref_type,jc_i.type,ac_i.type],
                                             [init_i, init_j, current_out,jc_i,ac_i])
                            bf = wh.before.blocks.append(init_i.type, init_j.type, elem_memref_type,jc_i.type,ac_i.type)
                            af = wh.after.blocks.append(init_i.type, init_j.type, elem_memref_type,jc_i.type,ac_i.type)

                            with InsertionPoint(bf):
                                running_i = bf.arguments[0]
                                running_j = bf.arguments[1]
                                running_out = bf.arguments[2]
                                jc_b = bf.arguments[3]
                                ac_b = bf.arguments[4]

                                # condition: running != 0
                                #
                                cond = arith.cmpi("ne", running_i,
                                                  arith.constant(tile_ty_size_in, type=i32(), index=True))

                                # scf.condition returns the condition + loop-carried values
                                scf.condition(cond, [running_i, running_j, running_out,jc_b,ac_b])

                            with InsertionPoint(af):
                                i = af.arguments[0]
                                j = af.arguments[1]
                                out = af.arguments[2]
                                jc_a = af.arguments[3]
                                ac_a = af.arguments[4]
                                # Acquire FIFO element

                                # idx0 = arith.constant( 0,index=True)
                                # val = memref.load(join_cnt_buffer, [idx0])

                                # Compute next running: break if sentinel 0
                                # is_stop = arith.cmpi("eq", val, cm1)
                                # next_running = arith.select(is_stop, arith.constant( 0), arith.constant(1))

                                # This seems to work
                                # join_cnt_buffer[0] = join_cnt_buffer[0] - 1

                                elem_in_i = elem_in[i]

                                cmpjoin = arith.cmpi("eq", elem_in_i,elem_inner[j])
                                #jcc = join_cnt[0]
                                out[jc_a] = elem_in_i
                                jc_a_after_join = arith.select(cmpjoin, jc_a + 1,jc_a)


                                #todo also test with scf
                                # with if_(elem_in_i == elem_inner[j]):
                                #     jcc = join_cnt[0]
                                #     out[jcc] = elem_in_i
                                #     join_cnt[0] = jcc + 1
                                    # global_join_cnt[0] = global_join_cnt[0] + 1

                                # chek if buffer full
                                # last iteration is handled later

                                # elem_memref_type = MemRefType.get([tile_ty_size_out], i32())


                                # Build the condition
                                cond = arith.cmpi("eq", jc_a_after_join,
                                                  arith.constant(tile_ty_size_out, type=i32(),index=True))

                                cond1 = arith.cmpi("slt", ac_a,
                                                  arith.constant(max_outer_aquire, type=i32()))

                                combined = arith.AndIOp(cond, cond1)

                                # Create an scf.if that RETURNS a memref result
                                if_op = scf.IfOp(combined, [elem_memref_type,jc_a.type,ac_a.type], hasElse=True)

                                # ---- THEN block: buffer is full → release old, acquire new ----
                                with InsertionPoint(if_op.then_block):
                                    of_out1.release(ObjectFifoPort.Produce, 1)
                                    jc_if_new = arith.constant(0, type=i32(), index=True)
                                    ac_if_new = ac_a + 1
                                    new_out = of_out1.acquire(ObjectFifoPort.Produce, 1)
                                    #only for safety
                                    #for z in range_(0, tile_ty_size_out, 1):
                                    #    new_out[z] = 0
                                    yield_([new_out,jc_if_new,ac_if_new])

                                # ---- ELSE block: buffer not full → keep current buffer ----
                                with InsertionPoint(if_op.else_block):
                                    yield_([out, jc_a_after_join, ac_a])
                                    #yield_([out,jc_a,ac_a])  # pass through the existing buffer unchanged

                                # The result of the if_op is the (possibly new) output buffer
                                ifres = if_op.results[0]
                                #join_cnt[0] = if_op.results[1]
                                #aquire_out_cnt[0] = if_op.results[2]

                                jc_if_res = if_op.results[1]
                                ac_if_res = if_op.results[2]


                                # Yield updated loop-carried values
                                next_running_j = arith.addi(j, arith.constant(1, type=i32(), index=True))
                                # next_running_i_maybe = arith.addi(i, arith.constant(1))
                                cmp = arith.cmpi("eq", next_running_j,
                                                 arith.constant(tile_ty_size_in, type=i32(), index=True))
                                next_running_j = arith.select(cmp, arith.constant(0, type=i32(), index=True),
                                                              next_running_j)

                                next_running_i = arith.select(cmp, i + 1, i)
                                scf.yield_([next_running_i, next_running_j, ifres,jc_if_res,ac_if_res])
                            of_in_inner.release(ObjectFifoPort.Consume, 1)
                            scf.yield_([wh.results[2],wh.results[3],wh.results[4]])

                        of_in1.release(ObjectFifoPort.Consume, 1)
                        scf.yield_([final_out[0],final_out[1],final_out[2]])


                    #zero rest of tensor
                    for z in range_(final_final_out[1],tile_ty_size_out,1):
                        final_final_out[0][z] =0

                    # Always do this as the loop will skip if all aquires  were used
                    of_out1.release(ObjectFifoPort.Produce, 1)
                    #join_cnt[0] = 0

                    #some extra aquire releases
                    # for _ in range_(0, 200, 1):
                    #     oo = of_out1.acquire(ObjectFifoPort.Produce, 1)
                    #     for z in range_(0,tile_ty_size_out,1):
                    #         oo[z] =0
                    #     of_out1.release(ObjectFifoPort.Produce, 1)



                    elem_done = of_done.acquire(ObjectFifoPort.Produce, 1)
                    for i in range_(16):
                        elem_done[i] = final_final_out[2]
                    #elem_done[0] = join_cnt[0]
                    of_done.release(ObjectFifoPort.Produce, 1)

            tiles_to_trace = [ComputeTile02, ShimTile00]
            if trace_size > 0:
                trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile20)
                # todo use other shimtile to trace?

            # To/from AIE-array data movement
            # d

            # tiles_to_trace = [ComputeTile02, MemTile01, ShimTile00]
            # if trace_size > 0:
            #    trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile20)

            @runtime_sequence(data_ty_in, data_ty_in, data_ty_out, data_ty_done)
            def sequence(inTensor, innerinTensor, outOddTensor, doneTensor, ):

                if trace_size > 0:
                    trace_utils.configure_packet_tracing_aie2(  # todo is this method correct form every npu?
                        tiles_to_trace=tiles_to_trace,
                        shim=ShimTile20,
                        ddr_id=4,  # 4 -> group_id(7)
                        trace_size=trace_size,
                    )

                in_task = shim_dma_single_bd_task(of_in1, inTensor, offset=0, sizes=[1, 1, 1, tranfer_size_elemnts_in],
                                                  issue_token=False)
                out_task = shim_dma_single_bd_task(
                    of_out1, outOddTensor, offset=0, sizes=[1, 1, 1, tranfer_size_elemnts_out], issue_token=False
                )

                done_task = shim_dma_single_bd_task(
                    of_done, doneTensor, offset=0, sizes=[1, 1, 1, 16], issue_token=True, burst_length=64
                )

                dma_start_task(in_task, out_task, done_task)

                for i in range(transfers_inner):
                    inner_in_task1 = shim_dma_single_bd_task(of_in_inner, innerinTensor, offset=0,
                                                             sizes=[1, 1, 1, tranfer_size_elemnts_in], issue_token=True)

                    dma_start_task(inner_in_task1)
                    dma_await_task(inner_in_task1)

                dma_await_task(done_task)
                dma_free_task(in_task)
                dma_free_task(out_task)

                # trace_utils.gen_trace_done_aie2(ShimTile20)

    res = True
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)


external_mem_to_core()
