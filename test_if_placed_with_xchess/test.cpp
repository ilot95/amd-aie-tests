#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
using DATATYPE = std::uint32_t;
#endif

int main(int argc, const char *argv[]) {
  // Program arguments parsing
  cxxopts::Options options("section-3");
  test_utils::add_default_options(options);

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();
  std::string trace_file = vm["trace_file"].as<std::string>();

  // Declaring design constants
  constexpr bool VERIFY = true;
  constexpr int IN_SIZE = 4096;
  constexpr int OUT_SIZE = IN_SIZE;

  // Load instruction sequence
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT context and load the kernel
  xrt::device device;
  xrt::kernel kernel;

  test_utils::init_xrt_load_kernel(device, kernel, verbosity,
                                   vm["xclbin"].as<std::string>(),
                                   vm["kernel"].as<std::string>());

  // set up the buffer objects
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_inA = xrt::bo(device, IN_SIZE * sizeof(DATATYPE),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));

  auto bo_outOdd = xrt::bo(device, OUT_SIZE * sizeof(DATATYPE),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_outC = xrt::bo(device, OUT_SIZE * sizeof(DATATYPE),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  int tmp_trace_size = (trace_size > 0) ? trace_size * 4 : 1;
  auto bo_trace = xrt::bo(device, tmp_trace_size,
						XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Copy instruction stream to xrt buffer object
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Initialize buffer bo_inA
  DATATYPE *bufInA = bo_inA.map<DATATYPE *>();
  for (int i = 0; i < IN_SIZE; i++)
    bufInA[i] = i;

  // Zero out buffer bo_outC
  DATATYPE *bufOut = bo_outC.map<DATATYPE *>();
  memset(bufOut, 0, OUT_SIZE * sizeof(DATATYPE));

  DATATYPE *bufOutOdd = bo_outOdd.map<DATATYPE *>();
  memset(bufOutOdd, 0, OUT_SIZE * sizeof(DATATYPE));


  // Initialize buffer bo_trace
  char *bufTrace = bo_trace.map<char *>();
  memset(bufTrace, 0, trace_size);

  // sync host to device memories
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_outC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_outOdd.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Execute the kernel and wait to finish
  int errors = 0;
  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;
  for (unsigned iter = 0; iter < num_iter; iter++) {
    if (verbosity >= 1)
      std::cout << "Running Kernel.\n";

    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run =
        kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_outC, bo_outOdd, 0, bo_trace);
    run.wait();
	bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto stop = std::chrono::high_resolution_clock::now();

    // Sync device to host memories
    bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_outOdd.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // Accumulate run times
    /* Warmup iterations do not count towards average runtime. */
    if (iter < n_warmup_iterations) {
      continue;
    }
    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();
    std::cout << std::endl
              << "NPU time: " << npu_time << "us."
              << std::endl;
    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;

	// only output the trace with the first formal iteration
	if (trace_size > 0 && iter == n_warmup_iterations) {
		test_utils::write_out_trace((char *)bufTrace, trace_size, trace_file);
	}

    // Compare out to golden
    if (verbosity >= 1) {
      std::cout << "Verifying results ..." << std::endl;
    }
    for (uint32_t i = 0; i < IN_SIZE; i++) {
      int32_t test = bufOut[i];
      std::cout << test << " ";
    }
    std::cout  << "\n";
    for (uint32_t i = 0; i < IN_SIZE; i++) {
      int32_t test = bufOutOdd[i];
      std::cout << test << " ";
    }
    std::cout << std::endl;
    for (uint32_t i = 0; i < IN_SIZE; i++) {
      int32_t ref = bufInA[i] + 2;
      int32_t test = bufOut[i];
      if (test != ref) {
        if (verbosity >= 1)
          std::cout << "Error in output " << test << " != " << ref << std::endl;
        errors++;
      } else {
        if (verbosity >= 1)
          std::cout << "Correct output " << test << " == " << ref << std::endl;
      }
    }
  }

  // print out profiling result
  std::cout << std::endl
          << "Number of iterations: " << n_iterations
          << " (warmup iterations: " << n_warmup_iterations << ")"
          << std::endl;

  std::cout << std::endl
            << "Avg NPU time: " << npu_time_total / n_iterations << "us."
            << std::endl;


  float macs = 0;
  if (macs > 0)
    std::cout << "Avg NPU gflops: "
              << macs / (1000 * npu_time_total / n_iterations) << std::endl;

  std::cout << std::endl
            << "Min NPU time: " << npu_time_min << "us." << std::endl;
  if (macs > 0)
    std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min)
              << std::endl;

  std::cout << std::endl
            << "Max NPU time: " << npu_time_max << "us." << std::endl;
  if (macs > 0)
    std::cout << "Min NPU gflops: " << macs / (1000 * npu_time_max)
              << std::endl;


  // Print Pass/Fail result of our test
  if (!errors) {
    std::cout << std::endl << "PASS!" << std::endl << std::endl;
    return 0;
  } else {
    std::cout << std::endl
              << errors << " mismatches." << std::endl
              << std::endl;
    std::cout << std::endl << "fail." << std::endl << std::endl;
    return 1;
  }
}
