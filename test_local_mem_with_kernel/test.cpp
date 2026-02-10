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
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"


#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
using DATATYPE = std::uint32_t;
#endif



uint32_t getParity(uint32_t n) {
  int count = 0;
  while (n > 0) {
    if (n & 1) { // Check if the least significant bit is 1
      count++;
    }
    n >>= 1; // Right shift to check the next bit
  }
  return (count % 2 == 0) ? 0 : 1; // 0 for even parity, 1 for odd parity
}

uint32_t create_ctrl_pkt(int operation, int beats, int addr,
                         int ctrl_pkt_read_id = 28) {
  uint32_t ctrl_pkt = ((ctrl_pkt_read_id & 0xFF) << 24) |
                      ((operation & 0x3) << 22) | ((beats & 0x3) << 20) |
                      (addr & 0x7FFFF);
  ctrl_pkt |= (0x1 ^ getParity(ctrl_pkt)) << 31;
  return ctrl_pkt;
}

int main(int argc, const char *argv[]) {
  // Program arguments parsing
  cxxopts::Options options("odd_even Kernel");
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
  constexpr bool PRINT_OUT_BUFFERS = false;
  constexpr int IN_SIZE = 4096;
  constexpr int OUT_SIZE = IN_SIZE;
  bool enable_ctrl_pkts = false;


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

  // If we enable control packets, then this is the input xrt buffer for that.
  // Otherwise, this is a dummy placedholder buffer.
    //todo why do we need this?
  auto bo_ctrlpkts =
      xrt::bo(device, 8, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  // Workaround so we declare a really small trace buffer when one is not used
  // Second workaround for driver issue. Allocate large trace buffer *4
  // This includes the 8 bytes needed for control packet response.
  //todo why 4*
  int tmp_trace_size = (trace_size > 0) ? trace_size * 4  : 1;
  auto bo_trace = xrt::bo(device, tmp_trace_size, XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(7));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Copy instruction stream to xrt buffer object
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Initialize buffer bo_inA

  DATATYPE *bufInA = bo_inA.map<DATATYPE *>();
  memset(bufInA, 0, IN_SIZE * sizeof(DATATYPE));
  //this will happen later in the loop
  /*for (int i = 0; i < IN_SIZE; i++)
    bufInA[i] = i;*/

  // Zero out buffer bo_outC
  DATATYPE *bufOut = bo_outC.map<DATATYPE *>();
  memset(bufOut, 0, OUT_SIZE * sizeof(DATATYPE));

  DATATYPE *bufOutOdd = bo_outOdd.map<DATATYPE *>();
  memset(bufOutOdd, 0, OUT_SIZE * sizeof(DATATYPE));

  char *bufTrace = bo_trace.map<char *>();
  uint32_t *bufCtrlPkts = bo_ctrlpkts.map<uint32_t *>();


    // Set control packet values
  if (trace_size > 0 && enable_ctrl_pkts) {
    bufCtrlPkts[0] = create_ctrl_pkt(1, 0, 0x32004); // core status
    bufCtrlPkts[1] = create_ctrl_pkt(1, 0, 0x320D8); // trace status
    if (verbosity >= 1) {
      std::cout << "bufCtrlPkts[0]:" << std::hex << bufCtrlPkts[0] << std::endl;
      std::cout << "bufCtrlPkts[1]:" << std::hex << bufCtrlPkts[1] << std::endl;
    }
  }

  // sync host to device memories
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_outC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_outOdd.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    if (trace_size > 0) {
    bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    if (enable_ctrl_pkts)
      bo_ctrlpkts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }



  // Execute the kernel and wait to finish
  int errors = 0;
  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;

  auto run = xrt::run(kernel);
  unsigned int opcode = 3;
  run.set_arg(0,opcode);
  run.set_arg(1,bo_instr);
  run.set_arg(2,instr_v.size());

  run.set_arg(3,bo_inA);
  run.set_arg(4,bo_outC);
  run.set_arg(5,bo_outOdd);
  //not sure about this one
  run.set_arg(6,bo_ctrlpkts);
  run.set_arg(7,bo_trace);


  for (int iter = 0; iter < num_iter; iter++) {
    //todo put back warmup iterrations
     std::cout << "iter: " << iter <<"\n";

      if (verbosity >= 1) {
      std::cout << "Setting inputs and zero out out buffers ..." << std::endl;
    }


      for (int i = 0; i < IN_SIZE; i++)
        bufInA[i] = i + iter;

      // Zero out buffer bo_outC
      memset(bufOut, 0, OUT_SIZE * sizeof(DATATYPE));
      memset(bufOutOdd, 0, OUT_SIZE * sizeof(DATATYPE));


      //this should not be needed
      //bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      bo_outC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      bo_outOdd.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "Running Kernel.\n";

    auto start = std::chrono::high_resolution_clock::now();

    /*auto run =
        kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_outC, bo_outOdd, 0, bo_trace);*/
    run.start();


    ert_cmd_state r = run.wait();

    auto stop = std::chrono::high_resolution_clock::now();

     if(r != ERT_CMD_STATE_COMPLETED){
        std::cout << "run.wait() did not return ERT_CMD_STATE_COMPLETED: " << r<<"\n";
    }

    // Sync device to host memories
    bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_outOdd.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (trace_size > 0)
      bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    //todo should tmp_trace_size be used here?
    if (trace_size > 0 ) {
      test_utils::write_out_trace(((char *)bufTrace), trace_size,
                                  trace_file);
    }
    // Accumulate run times
    /* Warmup iterations do not count towards average runtime. */

    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();
    std::cout << ""
              << "NPU time: " << npu_time << "us."
              << std::endl;
    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;



     if (PRINT_OUT_BUFFERS >= 1) {
      std::cout << "OutBuffers:" << std::endl;
      std::cout << "Even:" << std::endl;
      for (uint32_t i = 0; i < IN_SIZE; i++) {
      int32_t test = bufOut[i];
      std::cout << test << " ";
    }
    std::cout  << "\n";
    std::cout << "Odd:" << std::endl;
    for (uint32_t i = 0; i < IN_SIZE; i++) {
      int32_t test = bufOutOdd[i];
      std::cout << test << " ";
    }
    std::cout << std::endl;
    }

    // Compare out to golden

    if(VERIFY){
        if (verbosity >= 1) {
            std::cout << "Verifying results ..." << std::endl;
        }
        int cnt_odd = 0;
        int cnt_even =0;
        for (uint32_t i = 0; i < IN_SIZE; i++) {
            if(bufInA[i] % 2 ==0){
                if(!std::find(bufOut,bufOut+OUT_SIZE,bufInA[i])){
                  errors ++;
                }
                cnt_even++;
            }else{
                if(!std::find(bufOutOdd,bufOutOdd+OUT_SIZE,bufInA[i])){
                  errors ++;
                }else{
                }
                cnt_odd++;
            }

        }
         if (verbosity >= 1) {
             std::cout << "cnt_odd: " << cnt_odd << " cnt_even: " << cnt_even << "\n";
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
