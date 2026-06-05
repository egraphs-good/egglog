.PHONY: all test nits docs graphs rm-graphs doctest coverage insta-test fixnits

RUST_SRC=$(shell find . -type f -wholename '*/src/*.rs' -or -name 'Cargo.toml')
TESTS=$(shell find tests/ -type f -name '*.egg' -not -name '*repro-*')

WWW=${PWD}/target/www

# Keep release-mode test builds checking debug assertions without changing the
# normal release profile used by CodSpeed benchmarks and production binaries.
TEST_PROFILE_ENV=CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=true

all: test nits docs

test: doctest
	$(TEST_PROFILE_ENV) cargo insta test --test-runner nextest --release --workspace  --unreferenced reject

coverage:
	$(TEST_PROFILE_ENV) cargo llvm-cov nextest --release --workspace --lcov --output-path lcov.info
	# Note: doctests are not included in coverage reports

doctest:
	$(TEST_PROFILE_ENV) cargo test --doc --release --workspace


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
