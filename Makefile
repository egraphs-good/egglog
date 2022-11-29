.PHONY: all web test serve

RUST_SRC=$(shell find -type f -wholename '*/src/*.rs' -or -name 'Cargo.toml')
TESTS=$(shell find tests/ -type f -name '*.egg')

WWW=${PWD}/target/www/

WEB_SRC=$(wildcard web-demo/static/*)

WASM=web_demo.js web_demo_bg.wasm
DIST_WASM=$(addprefix ${WWW}, ${WASM})

all: test web

test:
	@rustup component add llvm-tools-preview
	cargo install grcov
	RUSTFLAGS='-Cinstrument-coverage' LLVM_PROFILE_FILE='target/debug/instrument-coverage/%p-%m.profraw' cargo test
	@rustup component add clippy
	cargo clippy --tests -- -D warnings
	@rustup component add rustfmt
	cargo fmt --check

coverage-html:
	grcov ./target/debug/instrument-coverage/ -s src/ --binary-path ./target/debug/ -t html --branch  -o ./target/debug/coverage/
	echo 'Wrote coverage report to ./target/debug/coverage/index.html'

coverage-lcov:
	grcov ./target/debug/instrument-coverage/ -s src/ --binary-path ./target/debug/ -t lcov --branch  -o ./target/debug/lcov.info

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


