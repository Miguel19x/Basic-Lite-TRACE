CC ?= gcc
CFLAGS ?= -O3 -mavx2 -mfma -ffp-contract=off -fopenmp -Wall -Wextra -Werror -shared -fPIC
EXE_CFLAGS ?= -O3 -mavx2 -mfma -ffp-contract=off -fopenmp -Wall -Wextra -Werror
TARGET = libpaged_gemm.so
LIBTERNARY = trace/ops/libternary.so
RUNNER_LIB = libtrace_infer.so
RUNNER_BIN = trace_infer
TOKENIZER_LIB = libtrace_tokenizer.so

GEMM_SRC = csrc/paged_gemm.c
INFER_SRCS = csrc/trace_loader.c csrc/trace_causal.c $(GEMM_SRC)
HEADERS = csrc/trace_infer.h csrc/trace_math.h
TOKENIZER_SRC = csrc/lm/trace_tokenizer.c
TOKENIZER_HDR = csrc/lm/trace_tokenizer.h
PYTHON ?= $(shell if [ -f venv/bin/python ]; then echo venv/bin/python; else echo python3; fi)

.PHONY: all clean test lint check

all: $(TARGET) $(LIBTERNARY) $(RUNNER_LIB) $(RUNNER_BIN) $(TOKENIZER_LIB)

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

test: all
	PYTHONPATH=. $(PYTHON) tests/test_smoke.py
	PYTHONPATH=. $(PYTHON) tests/test_avx2_kernel.py
	PYTHONPATH=. $(PYTHON) tests/test_trace_infer.py

lint:
	@if [ -f venv/bin/ruff ]; then venv/bin/ruff check .; else echo "ruff no instalado en venv, omitiendo linting"; fi

check: lint test

clean:
	rm -f $(TARGET) $(LIBTERNARY) $(RUNNER_LIB) $(RUNNER_BIN) $(TOKENIZER_LIB)
