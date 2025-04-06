.PHONY: all web test nits docs serve graphs rm-graphs

RUST_SRC=$(shell find . -type f -wholename '*/src/*.rs' -or -name 'Cargo.toml')
TESTS=$(shell find tests/ -type f -name '*.egg' -not -name '*repro-*')

WWW=${PWD}/target/www/

WEB_SRC=$(wildcard web-demo/static/*)

WASM=web_demo.js web_demo_bg.wasm
DIST_WASM=$(addprefix ${WWW}, ${WASM})

all: test nits web docs

test:
	cargo nextest run --release
	# nextest doesn't run doctests, so do it here
	cargo test --doc --release

nits:
	@rustup component add clippy
	cargo clippy --tests -- -D warnings
	@rustup component add rustfmt
	cargo fmt --check

docs:
	mkdir -p ${WWW}
	cargo doc --no-deps --all-features
	touch target/doc/.nojekyll # prevent github from trying to run jekyll
	cp -r target/doc ${WWW}/docs

web: docs ${DIST_WASM} ${WEB_SRC} ${WWW}/examples.json
	mkdir -p ${WWW}
	cp ${WEB_SRC} ${WWW}
	find target -name .gitignore -delete  # ignored files are wonky to deploy

serve:
	cargo watch --shell "make web && python3 -m http.server 8080 -d ${WWW}"

${WWW}/examples.json: web-demo/examples.py ${TESTS}
	$^ > $@

${DIST_WASM}: ${RUST_SRC}
	wasm-pack build web-demo --target no-modules --no-typescript --out-dir ${WWW}
	rm -f ${WWW}/{.gitignore,package.json}

graphs: $(patsubst %.egg,%.svg,$(filter-out $(wildcard tests/repro-*.egg),$(wildcard tests/*.egg)))

json: $(patsubst %.egg,%.json,$(filter-out $(wildcard tests/repro-*.egg),$(wildcard tests/*.egg)))

%.svg: %.egg
	cargo run --release -- --to-dot --to-svg  $^

%.json: %.egg
	cargo run --release -- --to-json $^

rm-graphs:
	rm -f tests/*.dot tests/*.svg

# TODO: remove before merging
collect-todos:
	@ cargo nextest run -r --no-fail-fast -- --skip math_microbenchmark 2>&1 >/dev/null \
	| grep "not yet implemented:" \
	| sort | uniq -c | sort -n
	@ cargo nextest run -r --no-fail-fast -- --skip math_microbenchmark 2>&1 >/dev/null \
	| grep "     Summary" \

# TODO: remove before merging
collect-wins:
	@ cargo nextest run -r --no-fail-fast -- --skip math_microbenchmark 2>&1 >/dev/null \
	| grep "PASS " \
	| grep ":files" \
	| grep -v "fail-typecheck" \
	| grep -v "resugar" \
	| sed 's/^.*egglog::files //' \
	| sort | uniq -c | sort -n
	@ cargo nextest run -r --no-fail-fast -- --skip math_microbenchmark 2>&1 >/dev/null \
	| grep "     Summary" \

# TODO: remove before merging
collect-fails:
	@ cargo nextest run -r --no-fail-fast -- --skip math_microbenchmark 2>&1 >/dev/null \
	| grep "test panicked:" \
	| grep -v "not yet implemented:" \
	| sort | uniq -c | sort -n
	@ cargo nextest run -r --no-fail-fast -- --skip math_microbenchmark 2>&1 >/dev/null \
	| grep "     Summary" \
