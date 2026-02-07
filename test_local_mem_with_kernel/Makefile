srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../makefile-common

all: build_peano/final.xclbin build_peano/insts.bin build_xchesscc/final.xclbin build_xchesscc/insts.bin

targetname = vectorScalar
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

build_mlir/aie.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< ${devicename} > $@
	

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
	${powershell} ./$< --verbosity=0 --iters=3 --warmup=1 --trace_sz=8192 --trace_file=trace.txt -x build_peano/final.xclbin -i build_peano/insts.bin -k MLIR_AIE
	#${srcdir}/../python/utils/parse_trace.py --input trace.txt --mlir build/aie.mlir --output trace.json
	#${srcdir}/../python/utils/get_trace_summary.py --input trace.json

run_xchesscc: ${targetname}.exe build_peano/final.xclbin
	${powershell} ./$< --verbosity=0 --iters=3 --warmup=1 --trace_sz=8192 --trace_file=trace.txt -x build_xchesscc/final.xclbin -i build_xchesscc/insts.bin -k MLIR_AIE
	#${srcdir}/../python/utils/parse_trace.py --input trace.txt --mlir build/aie.mlir --output trace.json
	#${srcdir}/../python/utils/get_trace_summary.py --input trace.json


run_all: run_peano run_xchesscc

clean: 
	rm -rf build_peano host_build build_mlir build_xchesscc ${targetname}.exe

