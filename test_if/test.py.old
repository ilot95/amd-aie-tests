# test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys
from aie.utils.xrt import  AIE_Application
import aie.utils.test as test_utils

def setup_aie(
    xclbin_path,
    insts_path,
    in_0_shape,
    in_0_dtype,
    in_1_shape,
    in_1_dtype,
    out_buf_shape,
    out_buf_dtype,
    enable_trace=False,
    kernel_name="MLIR_AIE",
    trace_size=16384,
    verbosity=0,
    trace_after_output=False,
):
    app = AIE_Application(xclbin_path, insts_path, kernel_name)

    if in_0_shape and in_0_dtype:
        if verbosity >= 1:
            print("register 1st input to group_id 3")
        app.register_buffer(3, shape=in_0_shape, dtype=in_0_dtype)
    if in_1_shape and in_1_dtype:
        if verbosity >= 1:
            print("register 2nd input to group_id 4")
        app.register_buffer(4, shape=in_1_shape, dtype=in_1_dtype)

    if enable_trace:
        if trace_after_output:
            out_buf_len_bytes = (
                np.prod(out_buf_shape) * np.dtype(out_buf_dtype).itemsize
            )
            out_buf_shape = (out_buf_len_bytes + trace_size,)
            out_buf_dtype = np.uint8

    if in_1_shape and in_1_dtype:
        if verbosity >= 1:
            print("register output to group_id 5")
        app.register_buffer(5, shape=out_buf_shape, dtype=out_buf_dtype)
    else:
        if verbosity >= 1:
            print("register output to group_id 4")
        app.register_buffer(4, shape=out_buf_shape, dtype=out_buf_dtype)

        app.register_buffer(5, shape=out_buf_shape, dtype=out_buf_dtype)
        if verbosity >= 1:
            print("register placeholder buffer (32b) to group_id 5")
        app.register_buffer(
            6, shape=(1,), dtype=np.uint32
        )  # TODO Needed so register buf 7 succeeds (not needed in C/C++ host code)

    if enable_trace:
        if not trace_after_output:
            trace_buf_shape = (
                trace_size * 4,
            )  # 4x as workaround to avoid driver corruption
            trace_buf_dtype = np.uint8
            if verbosity >= 1:
                print("register placeholder buffer (32b) to group_id 6")
            app.register_buffer(
                6, shape=(1,), dtype=np.uint32
            )  # TODO Needed so register buf 7 succeeds (not needed in C/C++ host code)
            if verbosity >= 1:
                print(
                    "register trace on 7: size: "
                    + str(trace_buf_shape)
                    + ", dtype:"
                    + str(trace_buf_dtype)
                )
            app.register_buffer(7, shape=trace_buf_shape, dtype=trace_buf_dtype)

    return app


# Wrapper function to write buffer arguments into registered input buffers, then call
# `run` function for AIE Application, and finally return the output buffer data.
def execute(
    app, input_one=None, input_two=None, enable_trace=False, trace_after_output=False
):
    if not (input_one is None):
        app.buffers[3].write(input_one)
    if not (input_two is None):
        app.buffers[4].write(input_two)

    app.run()

    if trace_after_output or not enable_trace:
        if not (input_two is None):
            # return app.buffers[5].read(), 0
            return app.buffers[5].read()
        else:
            # return app.buffers[4].read(), 0
            return app.buffers[5].read() , app.buffers[4].read()
    else:
        if not (input_two is None):
            return app.buffers[5].read(), app.buffers[7].read()
        else:
            return app.buffers[4].read(), app.buffers[7].read()



def main(opts):
    print("Running...\n")

    data_size = int(opts.size)
    dtype = np.int32

    app = setup_aie(
        opts.xclbin,
        opts.instr,
        data_size,
        dtype,
        None,
        None,
        data_size,
        dtype,
    )
    input = np.arange(0, data_size, dtype=dtype)
    aie_output = execute(app, input)

    # Copy output results and verify they are correct
    errors = 0
    if opts.verify:
        if opts.verbosity >= 1:
            print("Verifying results ...")

        #print("Verifying results ...")
        #e = np.equal(input, aie_output)
        #errors = np.size(e) - np.count_nonzero(e)
        print(aie_output)

    if not errors:
        #print("\nPASS!\n")
        exit(0)
    else:
        print("\nError count: ", errors)
        print("\nFailed.\n")
        exit(-1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    p.add_argument(
        "-s", "--size", required=True, dest="size", help="Passthrough kernel size"
    )
    opts = p.parse_args(sys.argv[1:])
    main(opts)
