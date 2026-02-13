srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../makefile-common

all: build_peano/final.xclbin build_peano/insts.bin build_xchesscc/final.xclbin build_xchesscc/insts.bin

targetname = vectorScalar
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

#todo make this settable
trace_size = 0


build_mlir/aie.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< ${devicename}  ${trace_size} > $@
	

#select the right kernel flags depending if you have npu aka NPU Phoenix aka aie_ml aka aie2
# or npu2 aka NPU Strix aka aie_2p ake aie2p
KERNEL_CC=xchesscc_wrapper
ifeq (${devicename}, npu)
KERNEL_CFLAGS=${CHESSCCWRAP2_FLAGS}
else ifeq (${devicename}, npu2)
KERNEL_CFLAGS=${CHESSCCWRAP2P_FLAGS}
endif

#a hacky way to use the right xchesscc there might be a better way
ifeq (${devicename}, npu)
	PATHVAR=${AIETOOLS_ROOT}/tps/lnx64/target_aie_ml/bin/LNa64bin
else ifeq (${devicename}, npu2)
	PATHVAR=${AIETOOLS_ROOT}/tps/lnx64/target_aie2p/bin/LNa64bin
endif


build_xchesscc/odd_even.o: ${srcdir}/odd_even.cc
	mkdir -p ${@D}
	cd ${@D} && ${KERNEL_CC} ${KERNEL_CFLAGS} ${KERNEL_DEFINES} -c $< -o ${@F}

build_xchesscc/final.xclbin: build_mlir/aie.mlir build_xchesscc/odd_even.o
	mkdir -p ${@D}
	cd ${@D}  &&  PATH=${PATHVAR}:$$PATH \
		  &&  aiecc.py  --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				--xchesscc --xbridge \
				--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)



build_peano/odd_even.o: ${srcdir}/odd_even.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}
else
	echo "Device type not supported"
endif

#--dynamic-objFifos  --no-xchesscc  --no-xbridge    --xchesscc --xbridge -v
build_peano/final.xclbin: build_mlir/aie.mlir build_peano/odd_even.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --dynamic-objFifos --no-compile-host --xclbin-name=${@F} \
				--no-xchesscc --no-xbridge  \
				--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf host_build
	mkdir -p host_build
	cd host_build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname}
	cd host_build && ${powershell} cmake --build . --config Release
	cp host_build/${targetname} $@

run_peano: ${targetname}.exe build_peano/final.xclbin
	${powershell} ./$< --verbosity=0 --iters=3 --warmup=0 --trace_sz=${trace_size} --trace_file=trace_peano.txt -x build_peano/final.xclbin -i build_peano/insts.bin -k MLIR_AIE
    #hacky works if iron env is inside mlir-aie repo
	${MLIR_AIE_DIR}/../../../../../python/utils/trace/parse.py --input trace_peano.txt --mlir build_mlir/aie.mlir --output trace_peano.json
	${MLIR_AIE_DIR}/../../../../../python/utils/trace/get_trace_summary.py --input trace_peano.json
    #no permission?
	#${MLIR_AIE_DIR}/python/aie/utils/trace/parse.py --input trace.txt --mlir build/aie.mlir --output trace.json
	#${MLIR_AIE_DIR}/python/aie/utils/trace/get_trace_summary.py --input trace.json

run_xchesscc: ${targetname}.exe build_xchesscc/final.xclbin
	${powershell} ./$< --verbosity=0 --iters=4 --warmup=0 --trace_sz=${trace_size} --trace_file=trace_xchesscc.txt -x build_xchesscc/final.xclbin -i build_xchesscc/insts.bin -k MLIR_AIE
	#hacky works if iron env is inside mlir-aie repo
	#${MLIR_AIE_DIR}/../../../../../python/utils/trace/parse.py --input trace_xchesscc.txt --mlir build_mlir/aie.mlir --output trace_xchesscc.json
	#${MLIR_AIE_DIR}/../../../../../python/utils/trace/get_trace_summary.py --input trace_xchesscc.json


run_all: run_peano run_xchesscc

clean: 
	rm -rf build_peano host_build build_mlir build_xchesscc ${targetname}.exe trace_peano.txt trace_peano.json trace_xchesscc.txt trace_xchesscc.json

