# Simple HIP debug build Makefile
# Usage examples:
#   make matmul GPU_ARCH=gfx90a
#   make matmulblas HIPBLAS=1
#   make matmulblas ROCBLAS=1 GPU_ARCH=gfx1100
#   make -j
#   make clean
#   make debug-matmul ARGS=1024

ROCM_PATH ?= /opt/rocm
HIPCC     ?= hipcc

# Try to autodetect one supported offload arch; override with `GPU_ARCH=...` if needed
# Try to autodetect one supported offload arch; override with `GPU_ARCH=...` if needed
# Prefer amdgpu-arch if available, else parse rocminfo, else default.
GPU_ARCH ?= $(shell amdgpu-arch 2>/dev/null | head -n1)
ifeq ($(GPU_ARCH),)
	GPU_ARCH := $(shell rocminfo 2>/dev/null | sed -n 's/.*\(gfx[0-9a-z][0-9a-z]*\).*/\1/p' | head -n1)
endif
ifeq ($(GPU_ARCH),)
	GPU_ARCH := gfx90a
endif

# Common compile and link flags for debug builds (host + device)
HIPCCFLAGS  += -g -O0 -fno-omit-frame-pointer --offload-arch=$(GPU_ARCH)
LDFLAGS   += -Wl,-rpath,$(ROCM_PATH)/lib -L$(ROCM_PATH)/lib

# Link libraries
LDLIBS    += -lamdhip64

# Select BLAS backend for matmulblas: set HIPBLAS=1 or ROCBLAS=1
ifdef HIPBLAS
  LDLIBS += -lhipblas
endif
ifdef ROCBLAS
  LDLIBS += -lrocblas
endif

BINARIES := matmul matmulblas her2

all: $(BINARIES)

matmul: matmul.cpp
	$(HIPCC) $(HIPCCFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

matmulblas: matmulblas.cpp
	$(HIPCC) $(HIPCCFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

her2: her2.cpp
	$(HIPCC) $(HIPCCFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

.PHONY: clean print-arch debug-matmul debug-matmulblas debug-her2

clean:
	rm -f $(BINARIES)

print-arch:
	@echo Using GPU offload arch: $(GPU_ARCH)

# Convenience debug targets using rocgdb
debug-matmul: matmul
	rocgdb --args ./matmul $(ARGS)

debug-matmulblas: matmulblas
	rocgdb --args ./matmulblas $(ARGS)

debug-her2: her2
	rocgdb --args ./her2 $(ARGS)
