WWW=${PWD}/target/www/

all: test nits docs

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
