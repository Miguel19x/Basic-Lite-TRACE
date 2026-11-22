CC ?= gcc
CFLAGS ?= -O3 -mavx2 -mfma -ffp-contract=off -fopenmp -Wall -Wextra -shared -fPIC
EXE_CFLAGS ?= -O3 -mavx2 -mfma -ffp-contract=off -fopenmp -Wall -Wextra
TARGET = libpaged_gemm.so
LIBTERNARY = trace/ops/libternary.so
RUNNER_LIB = libtrace_infer.so
RUNNER_BIN = trace_infer
SRC = paged_gemm_kernel.c
RUNNER_SRC = trace_infer.c
PYTHON ?= $(shell if [ -f venv/bin/python ]; then echo venv/bin/python; else echo python3; fi)

.PHONY: all clean test

all: $(TARGET) $(LIBTERNARY) $(RUNNER_LIB) $(RUNNER_BIN)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $< -o $@

$(LIBTERNARY): $(TARGET)
	cp $< $@

$(RUNNER_LIB): $(RUNNER_SRC) $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(RUNNER_BIN): $(RUNNER_SRC) $(SRC)
	$(CC) $(EXE_CFLAGS) -DTRACE_INFER_MAIN -o $@ $^ -lm

test: all
	PYTHONPATH=. $(PYTHON) tests/test_smoke.py
	PYTHONPATH=. $(PYTHON) tests/test_avx2_kernel.py
	PYTHONPATH=. $(PYTHON) tests/test_trace_infer.py

clean:
	rm -f $(TARGET) $(LIBTERNARY) $(RUNNER_LIB) $(RUNNER_BIN)
