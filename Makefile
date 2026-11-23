CC ?= gcc
CFLAGS ?= -O3 -mavx2 -mfma -ffp-contract=off -fopenmp -Wall -Wextra -Werror -shared -fPIC -I. -Icsrc
EXE_CFLAGS ?= -O3 -mavx2 -mfma -ffp-contract=off -fopenmp -Wall -Wextra -Werror -I. -Icsrc
TARGET = libpaged_gemm.so
LIBTERNARY = trace/ops/libternary.so
RUNNER_LIB = libtrace_infer.so
RUNNER_BIN = trace_infer
TOKENIZER_LIB = libtrace_tokenizer.so
LM_LIB = libtrace_lm.so
LM_BIN = trace_lm

GEMM_SRC = csrc/paged_gemm.c
INFER_SRCS = csrc/trace_loader.c csrc/trace_causal.c $(GEMM_SRC)
HEADERS = csrc/trace_infer.h csrc/trace_math.h csrc/paged_unpack.h
TOKENIZER_SRC = csrc/lm/trace_tokenizer.c
TOKENIZER_HDR = csrc/lm/trace_tokenizer.h
LM_SRCS = csrc/lm/trace_lm_loader.c csrc/lm/trace_lm_kv.c csrc/lm/trace_lm_infer.c $(GEMM_SRC)
LM_HDRS = csrc/lm/trace_lm.h csrc/lm/trace_lm_kv.h $(HEADERS)

PYTHON ?= $(shell if [ -f venv/bin/python ]; then echo venv/bin/python; else echo python3; fi)

.PHONY: all clean test lint check

all: $(TARGET) $(LIBTERNARY) $(RUNNER_LIB) $(RUNNER_BIN) $(TOKENIZER_LIB) $(LM_LIB) $(LM_BIN)

$(TARGET): $(GEMM_SRC) $(HEADERS)
	$(CC) $(CFLAGS) $< -o $@

$(LIBTERNARY): $(TARGET)
	cp $< $@

$(RUNNER_LIB): $(INFER_SRCS) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $(INFER_SRCS) -lm

$(RUNNER_BIN): $(INFER_SRCS) $(HEADERS)
	$(CC) $(EXE_CFLAGS) -DTRACE_INFER_MAIN -o $@ $(INFER_SRCS) -lm

$(TOKENIZER_LIB): $(TOKENIZER_SRC) $(TOKENIZER_HDR)
	$(CC) $(CFLAGS) -o $@ $(TOKENIZER_SRC)

$(LM_LIB): $(LM_SRCS) $(LM_HDRS)
	$(CC) $(CFLAGS) -o $@ $(LM_SRCS) -lm

$(LM_BIN): csrc/lm/trace_lm_cli.c $(LM_LIB) $(TOKENIZER_LIB) $(LM_HDRS) $(TOKENIZER_HDR)
	$(CC) $(EXE_CFLAGS) -o $@ csrc/lm/trace_lm_cli.c -L. -ltrace_lm -ltrace_tokenizer -lm -Wl,-rpath,'$$ORIGIN'

test: all
	PYTHONPATH=. $(PYTHON) tests/test_smoke.py
	PYTHONPATH=. $(PYTHON) tests/test_avx2_kernel.py
	PYTHONPATH=. $(PYTHON) tests/test_trace_infer.py
	PYTHONPATH=. $(PYTHON) tests/test_trace_lm_infer.py

lint:
	@if [ -f venv/bin/ruff ]; then venv/bin/ruff check .; else echo "ruff no instalado en venv, omitiendo linting"; fi

check: lint test

clean:
	rm -f $(TARGET) $(LIBTERNARY) $(RUNNER_LIB) $(RUNNER_BIN) $(TOKENIZER_LIB) $(LM_LIB) $(LM_BIN)
