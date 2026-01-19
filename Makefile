.PHONY: all test nits docs graphs rm-graphs doctest coverage test

RUST_SRC=$(shell find . -type f -wholename '*/src/*.rs' -or -name 'Cargo.toml')
TESTS=$(shell find tests/ -type f -name '*.egg' -not -name '*repro-*')

WWW=${PWD}/target/www

all: test nits docs

test: doctest
	cargo nextest run --release --workspace

coverage: doctest
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
	cargo clippy --fix --workspace --allow-dirty

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
