srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../makefile-common

all: build/final.xclbin build/insts.bin

targetname = vectorScalar
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

build/aie.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< ${devicename} > $@
	




#build/vector_operators.o: ${srcdir}/vector_operators.cc
#	mkdir -p ${@D}
#	cd ${@D} && xchesscc \
#		-f \
#		-p me \
#		-P $(AIETOOLS_ROOT)/data/aie_ml/lib \
#		-I $(AIETOOLS_ROOT)/include \
#		-c $(abspath $<) \
#		-o $(notdir $@)

#build/final.xclbin: build/aie.mlir build/vector_operators.o
#	mkdir -p ${@D}
#	cd ${@D} && aiecc.py --aie-generate-xclbin --dynamic-objFifos --no-compile-host --xclbin-name=${@F} \
#				--xchesscc --xbridge -v \
#				--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)



build/odd_even.o: ${srcdir}/odd_even.cc
	mkdir -p ${@D}
ifeq ($(devicename),npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}
else ifeq ($(devicename),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}
else
	echo "Device type not supported"
endif

#--dynamic-objFifos  --no-xchesscc  --no-xbridge    --xchesscc --xbridge -v
build/final.xclbin: build/aie.mlir build/odd_even.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --dynamic-objFifos --no-compile-host --xclbin-name=${@F} \
				--no-xchesscc --no-xbridge  \
				--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif 

run: ${targetname}.exe build/final.xclbin
	${powershell} ./$< --verbosity=0 --iters=3 --warmup=1 --trace_sz=8192 --trace_file=trace.txt -x build/final.xclbin -i build/insts.bin -k MLIR_AIE
	#${srcdir}/../python/utils/parse_trace.py --input trace.txt --mlir build/aie.mlir --output trace.json
	#${srcdir}/../python/utils/get_trace_summary.py --input trace.json

clean: 
	rm -rf build _build ${targetname}.exe

