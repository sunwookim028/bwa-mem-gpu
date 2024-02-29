MAKEFLAGS += -j64
CC=			gcc
CPPCC=		g++
#CC=			clang --analyze
CFLAGS=		-g -Wall -Wno-unused-function -O2
WRAP_MALLOC=-DUSE_MALLOC_WRAPPERS
AR=			ar
DFLAGS=		-DHAVE_PTHREAD $(WRAP_MALLOC)
LOBJS=		utils.o kthread.o kstring.o ksw.o bwt.o bntseq.o bwa.o bwamem.o bwamem_pair.o bwamem_extra.o malloc_wrap.o \
			QSufSort.o bwt_gen.o rope.o rle.o is.o bwtindex.o streams.o CUDAKernel_memmgnt.o bwt_CUDA.o bntseq_CUDA.o kbtree_CUDA.o ksw_CUDA.o bwa_CUDA.o kstring_CUDA.o bwamem_GPU.o GPULink.o minibatch_process.o superbatch_process.o
AOBJS=		bwashm.o bwase.o bwaseqio.o bwtgap.o bwtaln.o bamlite.o \
			bwape.o kopen.o pemerge.o maxk.o \
			bwtsw2_core.o bwtsw2_main.o bwtsw2_aux.o bwt_lite.o \
			bwtsw2_chain.o fastmap.o bwtsw2_pair.o loadKMerIndex.o
PROG=		bwa
INCLUDES=	
LIBS=		-lm -lz -lpthread
SUBDIRS=	.
CUDADIR=    ./cuda
CUDA_ARCH = sm_75
NVCC_OPTIM_FLAGS= -Xptxas -O4 -Xcompiler -O4 --device-c -arch=$(CUDA_ARCH)
NVCC_DEBUG_FLAGS= -g -G -O0 --device-c -arch=$(CUDA_ARCH)

ifeq ($(shell uname -s),Linux)
	LIBS += -lrt
endif

ifeq ($(debug), 1)
	NVCC_FLAGS = $(NVCC_DEBUG_FLAGS)
else
	NVCC_FLAGS = $(NVCC_OPTIM_FLAGS)
endif

.SUFFIXES:.c .o .cc

.c.o:
		$(CC) -c $(CFLAGS) $(DFLAGS) $(INCLUDES) $< -o $@

all:$(PROG)

bwa:libbwa.a $(AOBJS) main.o
		g++ $(CFLAGS) $(DFLAGS) $(AOBJS) main.o -o $@ -L. -lbwa -L/usr/local/cuda/lib64 -lcudart -lcudadevrt $(LIBS)

bwamem-lite:libbwa.a example.o
		$(CC) $(CFLAGS) $(DFLAGS) example.o -o $@ -L. -lbwa $(LIBS)

libbwa.a:$(LOBJS)
		$(AR) -csru $@ $(LOBJS)

clean:
		rm -f gmon.out *.o a.out $(PROG) *~ *.a

depend:
	( LC_ALL=C ; export LC_ALL; makedepend -Y -- $(CFLAGS) $(DFLAGS) -- *.c )

# DO NOT DELETE THIS LINE -- make depend depends on it.
CUDAKernel_memmgnt.o: $(CUDADIR)/CUDAKernel_memmgnt.cu $(CUDADIR)/CUDAKernel_memmgnt.cuh
	nvcc -I. $(NVCC_FLAGS) $(CUDADIR)/CUDAKernel_memmgnt.cu -o CUDAKernel_memmgnt.o
streams.o: $(CUDADIR)/streams.cu $(CUDADIR)/streams.cuh
	nvcc -I. $(NVCC_FLAGS) $(CUDADIR)/streams.cu -o streams.o
bwt_CUDA.o: $(CUDADIR)/bwt_CUDA.cu $(CUDADIR)/bwt_CUDA.cuh
	nvcc -I. $(NVCC_FLAGS) $(CUDADIR)/bwt_CUDA.cu -o bwt_CUDA.o
bntseq_CUDA.o: $(CUDADIR)/bntseq_CUDA.cuh $(CUDADIR)/bntseq_CUDA.cu
	nvcc -I. $(NVCC_FLAGS) $(CUDADIR)/bntseq_CUDA.cu -o bntseq_CUDA.o
kbtree_CUDA.o: $(CUDADIR)/kbtree_CUDA.cuh $(CUDADIR)/kbtree_CUDA.cu
	nvcc -I. $(NVCC_FLAGS) $(CUDADIR)/kbtree_CUDA.cu -o kbtree_CUDA.o
ksw_CUDA.o: $(CUDADIR)/ksw_CUDA.cuh $(CUDADIR)/ksw_CUDA.cu
	nvcc -I. $(NVCC_FLAGS) $(CUDADIR)/ksw_CUDA.cu -o ksw_CUDA.o
bwa_CUDA.o: $(CUDADIR)/bwa_CUDA.cuh $(CUDADIR)/bwa_CUDA.cu
	nvcc -I. $(NVCC_FLAGS) $(CUDADIR)/bwa_CUDA.cu -o bwa_CUDA.o
kstring_CUDA.o: $(CUDADIR)/kstring_CUDA.cuh $(CUDADIR)/kstring_CUDA.cu
	nvcc -I. $(NVCC_FLAGS) $(CUDADIR)/kstring_CUDA.cu -o kstring_CUDA.o
bwamem_GPU.o: bwamem.h bwa.h bntseq.h $(CUDADIR)/bwamem_GPU.cuh kbtree.h $(CUDADIR)/bwamem_GPU.cu $(CUDADIR)/CUDAKernel_memmgnt.cuh $(CUDADIR)/bwt_CUDA.cuh $(CUDADIR)/bntseq_CUDA.cuh $(CUDADIR)/kbtree_CUDA.cuh $(CUDADIR)/ksw_CUDA.cuh $(CUDADIR)/bwa_CUDA.cuh $(CUDADIR)/bwa_CUDA.cuh
	nvcc $(NVCC_FLAGS) $(CUDADIR)/bwamem_GPU.cu -o bwamem_GPU.o
GPULink.o: bwamem_GPU.o CUDAKernel_memmgnt.o bwt_CUDA.o bntseq_CUDA.o kbtree_CUDA.o ksw_CUDA.o bwa_CUDA.o kstring_CUDA.o streams.o
	nvcc -I. --device-link -arch=$(CUDA_ARCH) bwamem_GPU.o CUDAKernel_memmgnt.o bwt_CUDA.o bntseq_CUDA.o kbtree_CUDA.o ksw_CUDA.o bwa_CUDA.o kstring_CUDA.o streams.o --output-file GPULink.o
minibatch_process.o: cuda/minibatch_process.cpp cuda/minibatch_process.h
	nvcc -I. $(NVCC_FLAGS) -o minibatch_process.o cuda/minibatch_process.cpp
superbatch_process.o: cuda/superbatch_process.cpp cuda/superbatch_process.h
	nvcc -I. $(NVCC_FLAGS) -o superbatch_process.o cuda/superbatch_process.cpp
loadKMerIndex.o: kmers_index/loadKMerIndex.cpp kmers_index/hashKMerIndex.h
	g++ kmers_index/loadKMerIndex.cpp -c -o loadKMerIndex.o

QSufSort.o: QSufSort.h
bamlite.o: bamlite.h malloc_wrap.h
bntseq.o: bntseq.h utils.h kseq.h malloc_wrap.h khash.h
bwa.o: bntseq.h bwa.h bwt.h ksw.h utils.h kstring.h malloc_wrap.h kvec.h
bwa.o: kseq.h
bwamem.o: kstring.h malloc_wrap.h bwamem.h bwt.h bntseq.h bwa.h ksw.h kvec.h
bwamem.o: ksort.h utils.h kbtree.h
bwamem_extra.o: bwa.h bntseq.h bwt.h bwamem.h kstring.h malloc_wrap.h
bwamem_pair.o: kstring.h malloc_wrap.h bwamem.h bwt.h bntseq.h bwa.h kvec.h
bwamem_pair.o: utils.h ksw.h
bwape.o: bwtaln.h bwt.h kvec.h malloc_wrap.h bntseq.h utils.h bwase.h bwa.h
bwape.o: ksw.h khash.h
bwase.o: bwase.h bntseq.h bwt.h bwtaln.h utils.h kstring.h malloc_wrap.h
bwase.o: bwa.h ksw.h
bwaseqio.o: bwtaln.h bwt.h utils.h bamlite.h malloc_wrap.h kseq.h
bwashm.o: bwa.h bntseq.h bwt.h
bwt.o: utils.h bwt.h kvec.h malloc_wrap.h
bwt_gen.o: QSufSort.h malloc_wrap.h
bwt_lite.o: bwt_lite.h malloc_wrap.h
bwtaln.o: bwtaln.h bwt.h bwtgap.h utils.h bwa.h bntseq.h malloc_wrap.h
bwtgap.o: bwtgap.h bwt.h bwtaln.h malloc_wrap.h
bwtindex.o: bntseq.h bwa.h bwt.h utils.h rle.h rope.h malloc_wrap.h
bwtsw2_aux.o: bntseq.h bwt_lite.h utils.h bwtsw2.h bwt.h kstring.h
bwtsw2_aux.o: malloc_wrap.h bwa.h ksw.h kseq.h ksort.h
bwtsw2_chain.o: bwtsw2.h bntseq.h bwt_lite.h bwt.h malloc_wrap.h ksort.h
bwtsw2_core.o: bwt_lite.h bwtsw2.h bntseq.h bwt.h kvec.h malloc_wrap.h
bwtsw2_core.o: khash.h ksort.h
bwtsw2_main.o: bwt.h bwtsw2.h bntseq.h bwt_lite.h utils.h bwa.h
bwtsw2_pair.o: utils.h bwt.h bntseq.h bwtsw2.h bwt_lite.h kstring.h
bwtsw2_pair.o: malloc_wrap.h ksw.h
example.o: bwamem.h bwt.h bntseq.h bwa.h kseq.h malloc_wrap.h
fastmap.o: bwa.h bntseq.h bwt.h bwamem.h kvec.h malloc_wrap.h utils.h kseq.h
is.o: malloc_wrap.h
kopen.o: malloc_wrap.h
kstring.o: kstring.h malloc_wrap.h
ksw.o: ksw.h malloc_wrap.h
main.o: kstring.h malloc_wrap.h utils.h
malloc_wrap.o: malloc_wrap.h
maxk.o: bwa.h bntseq.h bwt.h bwamem.h kseq.h malloc_wrap.h
pemerge.o: ksw.h kseq.h malloc_wrap.h kstring.h bwa.h bntseq.h bwt.h utils.h
rle.o: rle.h
rope.o: rle.h rope.h
utils.o: utils.h ksort.h malloc_wrap.h kseq.h

.PHONY: test debug
test:
	./bwa mem -o out -v4 ../gaivi_test/HG38/GCF_000001405.39_GRCh38.p13_genomic.fna ../SRR043348_80K.fastq

debug:
	make clean && make -j10 debug=1 && CUDA_VISIBLE_DEVICES=0 cuda-gdb --args bwa mem -v4 -o out ../gaivi_test/HG38/GCF_000001405.39_GRCh38.p13_genomic.fna ../gaivi_test/datasets/SRR043348.fastq

profile:
	nsys profile --stat=true ./bwa mem -v4  ./GCF_000008865.2_ASM886v2_genomic.fna ./SRR10896389.fastq
