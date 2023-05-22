.PHONY: all web test serve graphs rm-graphs

RUST_SRC=$(shell find -type f -wholename '*/src/*.rs' -or -name 'Cargo.toml')
TESTS=$(shell find tests/ -type f -name '*.egg' -not -name '*repro-*')

WWW=${PWD}/target/www/

WEB_SRC=$(wildcard web-demo/static/*)

WASM=web_demo.js web_demo_bg.wasm
DIST_WASM=$(addprefix ${WWW}, ${WASM})

all: test web

test:
	cargo test --release -- -Zunstable-options --report-time
	@rustup component add clippy
	cargo clippy --tests -- -D warnings
	@rustup component add rustfmt
	cargo fmt --check

web: ${DIST_WASM} ${WEB_SRC} ${WWW}/examples.json
	mkdir -p ${WWW}
	cp ${WEB_SRC} ${WWW}

serve:
	cargo watch --shell "make web && python3 -m http.server 8080 -d ${WWW}"

${WWW}/examples.json: web-demo/examples.py ${TESTS}
	$^ > $@

${DIST_WASM}: ${RUST_SRC}
	wasm-pack build web-demo --target no-modules --no-typescript --out-dir ${WWW}
	rm -f ${WWW}/{.gitignore,package.json}

graphs: $(patsubst %.egg,%.svg,$(filter-out tests//fail-typecheck/%, ${TESTS}))

%.svg: %.egg
	cargo run -- --save-dot --save-svg  $^

rm-graphs:
	rm -f tests/*.dot tests/*.svg
