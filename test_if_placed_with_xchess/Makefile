srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../makefile-common

.PHONY: all clean

targetname = main
devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

#all: build/final.xclbin build/insts.bin
all: build/final.xclbin ${targetname}.exe

# Copy mlir to build directory
# build/aie.mlir: ${srcdir}/aie.mlir
#	mkdir -p ${@D}
#	cp $< $@
build/aie.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

build/vector_operators.o: ${srcdir}/vector_operators.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc \
		-f \
		-p me \
		-P $(AIETOOLS_ROOT)/data/aie_ml/lib \
		-I $(AIETOOLS_ROOT)/include \
		-c $(abspath $<) \
		-o $(notdir $@)

build/final.xclbin: build/aie.mlir build/vector_operators.o
	mkdir -p ${@D}
	cd ${@D}  &&  PATH=${AIETOOLS_ROOT}/tps/lnx64/target_aie_ml/bin/LNa64bin:$$PATH \
		  &&  aiecc.py  --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				--xchesscc --xbridge \
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

clean:
	rm -rf build _build ${targetname}.exe
#------------------------------------------------------------------------------
# Additional make targets for tutorial exercises
# Note: AIECC_HOST_FLAGS is defined in the included tutorials/makefile-common.
#------------------------------------------------------------------------------
#tutorial-1.exe: ${srcdir}/test.cpp ${srcdir}/aie.mlir
#	aiecc.py -j4 $(AIECC_FLAGS) $(word 2,$^) $(AIECC_HOST_FLAGS) ./$< -o $@

run: ${targetname}.exe build/final.xclbin
	${powershell} ./$< --verbosity=0 --iters=10 --warmup=3 -x build/final.xclbin -i build/insts.bin -k MLIR_AIE
