.PHONY: all web test nits docs serve

RUST_SRC=$(shell find -type f -wholename '*/src/*.rs' -or -name 'Cargo.toml')
TESTS=$(shell find tests/ -type f -name '*.egg' -not -name '*repro-*')

WWW=${PWD}/target/www/

WEB_SRC=$(wildcard web-demo/static/*)

WASM=web_demo.js web_demo_bg.wasm
DIST_WASM=$(addprefix ${WWW}, ${WASM})

all: test nits web docs

test:
	cargo test --release -- -Zunstable-options --report-time

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


