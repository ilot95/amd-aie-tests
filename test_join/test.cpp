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
  constexpr int IN_SIZE = 8;
  constexpr int OUT_SIZE = IN_SIZE*IN_SIZE;

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

   auto bo_inB = xrt::bo(device, IN_SIZE * sizeof(DATATYPE),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  auto bo_outOdd = xrt::bo(device, OUT_SIZE * sizeof(DATATYPE),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_outC = xrt::bo(device, OUT_SIZE * sizeof(DATATYPE),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
  int tmp_trace_size = (trace_size > 0) ? trace_size * 4 : 1;


  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Copy instruction stream to xrt buffer object
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Initialize buffer bo_inA
  DATATYPE *bufInA = bo_inA.map<DATATYPE *>();
  for (int i = 0; i < IN_SIZE; i++)
    bufInA[i] = i+1;

  /*for (int i = 0; i < IN_SIZE; i++)
    bufInA[i] = 3;*/


  DATATYPE *bufInB = bo_inB.map<DATATYPE *>();
  for (int i = 0; i < IN_SIZE; i++)
    bufInB[i] = i+1;

  /*for (int i = 0; i < IN_SIZE; i++)
    bufInB[i] = 3;*/

  // Zero out buffer bo_outC
  DATATYPE *bufOut = bo_outC.map<DATATYPE *>();
  memset(bufOut, 0, OUT_SIZE * sizeof(DATATYPE));

  DATATYPE *bufOutOdd = bo_outOdd.map<DATATYPE *>();
  memset(bufOutOdd, 0, OUT_SIZE * sizeof(DATATYPE));




  // sync host to device memories
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_outC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_outOdd.sync(XCL_BO_SYNC_BO_TO_DEVICE);


  // Execute the kernel and wait to finish
  int errors = 0;
  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;


    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run =
        kernel(opcode, bo_instr, instr_v.size(), bo_inA,bo_inB, bo_outC, bo_outOdd);
    run.wait2();

    auto stop = std::chrono::high_resolution_clock::now();

    // Sync device to host memories
    bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_outOdd.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();
    std::cout << std::endl
              << "NPU time: " << npu_time << "us."
              << std::endl;


    for (uint32_t i = 0; i < OUT_SIZE; i++) {
      int32_t test = bufOut[i];
      std::cout << test << " ";
    }
    std::cout  << "\n";
    for (uint32_t i = 0; i < OUT_SIZE; i++) {
      int32_t test = bufOutOdd[i];
      std::cout << test << " ";
    }
    std::cout << std::endl;


for (int i = 0; i < IN_SIZE; i++)
    bufInA[i] = 3;


  for (int i = 0; i < IN_SIZE; i++)
    bufInB[i] = i+1;


  memset(bufOut, 0, OUT_SIZE * sizeof(DATATYPE));

  memset(bufOutOdd, 0, OUT_SIZE * sizeof(DATATYPE));




  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_outC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_outOdd.sync(XCL_BO_SYNC_BO_TO_DEVICE);


sleep(3);


     start = std::chrono::high_resolution_clock::now();

     //run = kernel(opcode, bo_instr, instr_v.size(), bo_inA,bo_inB, bo_outC, bo_outOdd);
     run.start();
    run.wait2();

     stop = std::chrono::high_resolution_clock::now();

    // Sync device to host memories
    bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_outOdd.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

     npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();
    std::cout << std::endl
              << "NPU time: " << npu_time << "us."
              << std::endl;


    for (uint32_t i = 0; i < OUT_SIZE; i++) {
      int32_t test = bufOut[i];
      std::cout << test << " ";
    }
    std::cout  << "\n";
    for (uint32_t i = 0; i < OUT_SIZE; i++) {
      int32_t test = bufOutOdd[i];
      std::cout << test << " ";
    }
    std::cout << std::endl;


      memset(bufOut, 0, OUT_SIZE * sizeof(DATATYPE));

  memset(bufOutOdd, 0, OUT_SIZE * sizeof(DATATYPE));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_outC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_outOdd.sync(XCL_BO_SYNC_BO_TO_DEVICE);




     start = std::chrono::high_resolution_clock::now();
    opcode = 3;
     //run = kernel(opcode, bo_instr, instr_v.size(), bo_inA,bo_inB, bo_outC, bo_outOdd);
     run.start();
    run.wait2();

     stop = std::chrono::high_resolution_clock::now();

    // Sync device to host memories
    bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_outOdd.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

     npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();
    std::cout << std::endl
              << "NPU time: " << npu_time << "us."
              << std::endl;


    for (uint32_t i = 0; i < OUT_SIZE; i++) {
      int32_t test = bufOut[i];
      std::cout << test << " ";
    }
    std::cout  << "\n";
    for (uint32_t i = 0; i < OUT_SIZE; i++) {
      int32_t test = bufOutOdd[i];
      std::cout << test << " ";
    }
    std::cout << std::endl;





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
