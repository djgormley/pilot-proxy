PYTHON ?= python3
NVCC ?= nvcc
GNURADIO_PYTHON ?= /usr/bin/python3
GNURADIO_ENV ?= PYTHONNOUSERSITE=1
CUDA_CACHE_DIR ?= $(HOME)/.cache/pilot_proxy
CUDA_LIB := cuda/libfstatistic.so
CACHED_CUDA_LIB := $(CUDA_CACHE_DIR)/libfstatistic.so
PYTHON_TEST_ENV := MPLBACKEND=Agg PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PILOT_PROXY_USE_TEX=0 PYTHONPATH=src

.PHONY: build-kernel test-kernel test-python test-gpu test lint archive-integration-check generate-atsc audit-atsc quantize detect evaluate release-check commit-check release-clean freeze-check clean-cache clean docs docs-specs docs-specs-check

build-kernel:
	$(MAKE) -C cuda
	mkdir -p $(CUDA_CACHE_DIR)
	cp $(CUDA_LIB) $(CACHED_CUDA_LIB)
	@echo "Staged $(CACHED_CUDA_LIB)"

test-kernel:
	$(MAKE) -C cuda test

test-python:
	$(PYTHON) scripts/check_test_environment.py
	$(PYTHON_TEST_ENV) $(PYTHON) -m pytest tests -q

test: test-kernel test-python

test-gpu: test-kernel
	$(PYTHON_TEST_ENV) NVCC="$(NVCC)" $(PYTHON) -m pytest \
	    tests/kernel tests/core/test_reference_pfb_gpu.py --require-cuda -q -rs

lint:
	$(PYTHON) -m ruff check .

archive-integration-check:
	$(PYTHON) scripts/check_test_environment.py --integration
	$(PYTHON) scripts/check_current_layout_vocabulary.py
	$(PYTHON) scripts/check_current_profile_contract.py
	$(PYTHON) scripts/check_current_measurement_vocabulary.py
	$(PYTHON) scripts/check_current_configuration_identity.py
	$(PYTHON) scripts/check_current_schema_identity.py
	$(PYTHON_TEST_ENV) $(PYTHON) -m pytest tests/archive -q -rs

docs-specs:
	$(PYTHON) tools/generate_doc_specs.py

docs-specs-check:
	$(PYTHON) tools/generate_doc_specs.py --check

docs: docs-specs
	@command -v latexmk >/dev/null || { \
	    echo "latexmk not found -- see 'Build documentation' in README.md"; exit 1; }
	cd docs && for t in *.tex; do \
	    latexmk -pdf -interaction=nonstopmode -halt-on-error \
	        -auxdir=auxil -outdir=out $$t || exit 1; \
	done
	@echo "PDFs in docs/out/"

release-check:
	$(PYTHON) scripts/check_test_environment.py
	$(MAKE) lint
	$(MAKE) docs-specs-check
	bash scripts/check_no_legacy_guard_terms.sh
	$(PYTHON) scripts/check_current_product_vocabulary.py
	$(PYTHON) scripts/check_current_layout_vocabulary.py
	$(PYTHON) scripts/check_current_profile_contract.py
	$(PYTHON) scripts/check_current_measurement_vocabulary.py
	$(PYTHON) scripts/check_current_configuration_identity.py
	$(PYTHON) scripts/check_current_schema_identity.py
	$(PYTHON_TEST_ENV) $(PYTHON) tools/emit_fxfft_tables.py --check
	$(PYTHON_TEST_ENV) $(PYTHON) -m compileall -q src tests
	$(PYTHON_TEST_ENV) $(PYTHON) -m pytest tests -q
	$(MAKE) -C cuda clean test_c_header test_ref
	PYTHONPATH=src $(PYTHON) -m pilot_proxy.cli check-profile \
	    --receiver-profile configs/receiver_profiles/reference_800mhz_pfb.json
	PYTHONPATH=src $(PYTHON) -m pilot_proxy.cli check-layout \
	    --receiver-profile configs/receiver_profiles/reference_800mhz_pfb.json
	PYTHONPATH=src $(PYTHON) -m pilot_proxy.cli check-layout \
	    --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json \
	    --stream-map configs/stream_maps/chime_feed_pol_example.json
	tmpdir="$$(mktemp -d)"; \
	trap 'rm -rf "$$tmpdir"' EXIT; \
	PYTHONPATH=src $(PYTHON) -m pilot_proxy.cli export-runtime-weight-bundle \
	    --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json \
	    --weight-coordinate-system post_spectral_sense_normalization \
	    --physical-channel-range 14:36 \
	    --output-dir "$$tmpdir"; \
	PYTHONPATH=src $(PYTHON) -m pilot_proxy.cli validate-runtime-weight-bundle \
	    --bundle-dir "$$tmpdir"
	PYTHONPATH=src $(PYTHON) -m pilot_proxy.cli list-channels
	$(MAKE) -C cuda clean

commit-check:
	bash scripts/check_repo_clean_for_commit.sh

release-clean:
	rm -rf .pytest_cache .ruff_cache .idea build dist generated inspection \
	    __pycache__ src/pilot_proxy.egg-info docs/auxil docs/out docs/generated
	find src tests scripts tools -type d -name __pycache__ -prune \
	    -exec rm -rf {} +
	$(MAKE) -C cuda clean

freeze-check:
	$(MAKE) release-check
	$(MAKE) release-clean
	$(MAKE) commit-check

generate-atsc:
	$(GNURADIO_ENV) PYTHONPATH=src $(GNURADIO_PYTHON) -m pilot_proxy.testbench.generate_atsc_signal

audit-atsc:
	PYTHONPATH=src $(PYTHON) -m pilot_proxy.testbench.audit_atsc_signal \
	    --input-iq generated/atsc/atsc_8vsb_complex64.cfile \
	    --fail-on-quality

quantize:
	PYTHONPATH=src $(PYTHON) -m pilot_proxy.testbench.quantize \
	    --input-iq generated/atsc/atsc_8vsb_complex64.cfile \
	    --physical-channel 14 \
	    --frame-size-samples 16384

evaluate:
	PYTHONPATH=src $(PYTHON) -m pilot_proxy.testbench.evaluate_snr \
	    --input-iq generated/atsc/atsc_8vsb_complex64.cfile \
	    --physical-channel 14 \
	    --frame-size-samples 16384 \
	    --requested-data-shelf-snr-db -26 \
	    --noise-trials 10 \
	    --gnuradio-python $(GNURADIO_PYTHON)

detect:
	PYTHONPATH=src $(PYTHON) -m pilot_proxy.detect \
	    --input-detector-matrix generated/detector_input/detector_matrix_i4.npy

clean-cache:
	rm -f $(CACHED_CUDA_LIB)

clean:
	$(MAKE) -C cuda clean
	rm -rf generated
