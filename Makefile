.PHONY: all test nits docs graphs rm-graphs doctest coverage insta-test fixnits nightly

RUST_SRC=$(shell find . -type f -wholename '*/src/*.rs' -or -name 'Cargo.toml')
TESTS=$(shell find tests/ -type f -name '*.egg' -not -name '*repro-*')

WWW=${PWD}/target/www

all: test nits docs

# Build egglog and benchmark every tests/*.egg file with hyperfine, writing an
# HTML dashboard to nightly/output/ (matching `report=` in nightly-conf).
# Run nightly on nightly.cs.washington.edu. Dependencies install into a venv so
# this works on PEP 668 externally-managed systems; eval_live must be importable
# by the script's interpreter, so run it with the venv's python.
nightly:
	python3 -m venv nightly/.venv
	nightly/.venv/bin/pip install -q -r scripts/requirements.txt
	nightly/.venv/bin/python scripts/nightly_bench.py

test: doctest
	cargo insta test --test-runner nextest --release --workspace  --unreferenced reject

coverage:
	cargo llvm-cov nextest --release --workspace --lcov --output-path lcov.info
	# Note: doctests are not included in coverage reports

doctest:
	cargo test --doc --release --workspace


nits:
	@rustup component add clippy
	cargo clippy --tests -- -D warnings
	@rustup component add rustfmt
	cargo fmt --check
	cargo doc --workspace

fixnits:
	@rustup component add rustfmt
	cargo fmt
	@rustup component add rustfmt
	cargo clippy --fix --tests --workspace --allow-dirty

docs:
	mkdir -p ${WWW}/
	cargo doc --no-deps --all-features --workspace
	touch target/doc/.nojekyll # prevent github from trying to run jekyll
	cp www/index.html ${WWW}/index.html
	cp -r target/doc ${WWW}/docs

graphs: $(patsubst %.egg,%.svg,$(filter-out $(wildcard tests/repro-*.egg),$(wildcard tests/*.egg)))

json: $(patsubst %.egg,%.json,$(filter-out $(wildcard tests/repro-*.egg),$(wildcard tests/*.egg)))

%.svg: %.egg
	cargo run --release -- --to-dot --to-svg  $^

%.json: %.egg
	cargo run --release -- --to-json $^

rm-graphs:
	rm -f tests/*.dot tests/*.svg
