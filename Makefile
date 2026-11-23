CC ?= gcc
ARCH := $(shell uname -m)

CFLAGS ?= -O3 -ffp-contract=off -fopenmp -Wall -Wextra -Werror -I. -Icsrc
PIC_CFLAGS ?= $(CFLAGS) -fPIC
EXE_CFLAGS ?= $(CFLAGS)
AVX2_FLAGS = -mavx2 -mfma -DTRACE_ENABLE_AVX2

ifeq ($(ARCH),x86_64)
    DISPATCH_OBJS = csrc/paged_dispatch.o csrc/paged_gemm_scalar.o csrc/paged_gemm_avx2.o
    DISPATCH_DEFS = -DTRACE_ENABLE_AVX2
else
    DISPATCH_OBJS = csrc/paged_dispatch.o csrc/paged_gemm_scalar.o
    DISPATCH_DEFS =
endif

TARGET = libpaged_gemm.so
LIBTERNARY = trace/ops/libternary.so
RUNNER_LIB = libtrace_infer.so
RUNNER_BIN = trace_infer
TOKENIZER_LIB = libtrace_tokenizer.so
LM_LIB = libtrace_lm.so
LM_BIN = trace_lm

INFER_SRCS = csrc/trace_loader.c csrc/trace_causal.c
INFER_OBJS = csrc/trace_loader.o csrc/trace_causal.o
HEADERS = csrc/trace_infer.h csrc/trace_math.h csrc/paged_unpack.h csrc/paged_dispatch.h
TOKENIZER_SRC = csrc/lm/trace_tokenizer.c
TOKENIZER_HDR = csrc/lm/trace_tokenizer.h
LM_SRCS = csrc/lm/trace_lm_loader.c csrc/lm/trace_lm_kv.c csrc/lm/trace_lm_infer.c
LM_OBJS = csrc/lm/trace_lm_loader.o csrc/lm/trace_lm_kv.o csrc/lm/trace_lm_infer.o
LM_HDRS = csrc/lm/trace_lm.h csrc/lm/trace_lm_kv.h $(HEADERS)

PYTHON ?= $(shell if [ -f venv/bin/python ]; then echo venv/bin/python; else echo python3; fi)

.PHONY: all clean test lint check

all: $(TARGET) $(LIBTERNARY) $(RUNNER_LIB) $(RUNNER_BIN) $(TOKENIZER_LIB) $(LM_LIB) $(LM_BIN)

csrc/paged_dispatch.o: csrc/paged_dispatch.c csrc/paged_dispatch.h
	$(CC) $(PIC_CFLAGS) $(DISPATCH_DEFS) -c $< -o $@

csrc/paged_gemm_scalar.o: csrc/paged_gemm_scalar.c csrc/paged_dispatch.h
	$(CC) $(PIC_CFLAGS) -c $< -o $@

csrc/paged_gemm_avx2.o: csrc/paged_gemm_avx2.c csrc/paged_dispatch.h csrc/paged_unpack.h
	$(CC) $(PIC_CFLAGS) $(AVX2_FLAGS) -c $< -o $@

csrc/trace_loader.o: csrc/trace_loader.c csrc/trace_infer.h
	$(CC) $(PIC_CFLAGS) -c $< -o $@

csrc/trace_causal.o: csrc/trace_causal.c $(HEADERS)
	$(CC) $(PIC_CFLAGS) -c $< -o $@

csrc/lm/trace_lm_loader.o: csrc/lm/trace_lm_loader.c csrc/lm/trace_lm.h
	$(CC) $(PIC_CFLAGS) -c $< -o $@

csrc/lm/trace_lm_kv.o: csrc/lm/trace_lm_kv.c csrc/lm/trace_lm_kv.h
	$(CC) $(PIC_CFLAGS) -c $< -o $@

csrc/lm/trace_lm_infer.o: csrc/lm/trace_lm_infer.c $(LM_HDRS)
	$(CC) $(PIC_CFLAGS) -c $< -o $@

$(TARGET): $(DISPATCH_OBJS)
	$(CC) $(PIC_CFLAGS) -shared -o $@ $(DISPATCH_OBJS)

$(LIBTERNARY): $(TARGET)
	@mkdir -p trace/ops
	cp $< $@

$(RUNNER_LIB): $(INFER_OBJS) $(DISPATCH_OBJS)
	$(CC) $(PIC_CFLAGS) -shared -o $@ $(INFER_OBJS) $(DISPATCH_OBJS) -lm

$(RUNNER_BIN): $(INFER_SRCS) $(DISPATCH_OBJS) $(HEADERS)
	$(CC) $(EXE_CFLAGS) -DTRACE_INFER_MAIN -o $@ $(INFER_SRCS) $(DISPATCH_OBJS) -lm

$(TOKENIZER_LIB): $(TOKENIZER_SRC) $(TOKENIZER_HDR)
	$(CC) $(PIC_CFLAGS) -shared -o $@ $(TOKENIZER_SRC)

$(LM_LIB): $(LM_OBJS) $(DISPATCH_OBJS)
	$(CC) $(PIC_CFLAGS) -shared -o $@ $(LM_OBJS) $(DISPATCH_OBJS) -lm

$(LM_BIN): csrc/lm/trace_lm_cli.c $(LM_LIB) $(TOKENIZER_LIB) $(LM_HDRS) $(TOKENIZER_HDR)
	$(CC) $(EXE_CFLAGS) -o $@ csrc/lm/trace_lm_cli.c -L. -ltrace_lm -ltrace_tokenizer -lm -Wl,-rpath,'$$ORIGIN'

test: all
	PYTHONPATH=. $(PYTHON) tests/test_smoke.py
	PYTHONPATH=. $(PYTHON) tests/test_avx2_kernel.py
	PYTHONPATH=. $(PYTHON) tests/test_trace_infer.py
	PYTHONPATH=. $(PYTHON) tests/test_trace_lm_infer.py
	PYTHONPATH=. $(PYTHON) tests/test_parity_fase3.py

lint:
	@if [ -f venv/bin/ruff ]; then venv/bin/ruff check .; else echo "ruff no instalado en venv, omitiendo linting"; fi

check: lint test

clean:
	rm -f $(TARGET) $(LIBTERNARY) $(RUNNER_LIB) $(RUNNER_BIN) $(TOKENIZER_LIB) $(LM_LIB) $(LM_BIN)
	rm -f csrc/*.o csrc/lm/*.o
