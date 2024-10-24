NVCC = nvcc

BUILDDIR  = /mnt/llm/workspace/yuyue2/my/build

MPIINCDIR = /mnt/llm/workspace/yuyue2/.local/mpicc/include/
MPILNKDIR = /mnt/llm/workspace/yuyue2/.local/mpicc/lib/

# A100
NCCL80INCDIR = /mnt/llm/workspace/yuyue2/.local/nccl_80/include/
NCCL80LNKDIR = /mnt/llm/workspace/yuyue2/.local/nccl_80/lib/

# RTX 4090
NCCL89INCDIR = /mnt/llm/workspace/yuyue2/.local/nccl_89/include/
NCCL89LNKDIR = /mnt/llm/workspace/yuyue2/.local/nccl_89/lib/

FLAGS   = -std=c++17 -O3 --generate-line-info
LDFLAGS = -lmpi -lnccl

all: a100 4090

a100: ${BUILDDIR}/real-343/allreduce

${BUILDDIR}/real-343/allreduce: allreduce.cu
	${NVCC} -I${MPIINCDIR} -L${MPILNKDIR} -I${NCCL80DIR} -L${NCCL80LNKDIR} ${FLAGS} ${LDFLAGS} -gencode=arch=compute_80,code=sm_80 -o $@ $<

4090: ${BUILDDIR}/gpu/allreduce

${BUILDDIR}/gpu/allreduce: allreduce.cu
	${NVCC} -I${MPIINCDIR} -L${MPILNKDIR} -I${NCCL89DIR} -L${NCCL89LNKDIR} ${FLAGS} ${LDFLAGS} -gencode=arch=compute_89,code=sm_89 -o $@ $<

.PHONY: clean

clean:
	rm ${BUILDDIR}/*/allreduce