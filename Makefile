.PHONY: all web test serve

RUST_SRC=$(shell find -type f -wholename '*/src/*.rs' -or -name 'Cargo.toml')
TESTS=$(shell find tests/ -type f -name '*.egg')

WWW=${PWD}/target/www/

WEB_SRC=web-demo/index.html web-demo/worker.js

WASM=web_demo.js web_demo_bg.wasm
DIST_WASM=$(addprefix ${WWW}, ${WASM})

all: test web

test:
	cargo test
	cargo clippy --tests -- -D warnings
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


