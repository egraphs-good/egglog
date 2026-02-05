# egglog: The Next-Generation Equality Saturation Engine

<a href="https://egraphs-good.github.io/egglog/">
    <img alt="Web Demo" src="https://img.shields.io/badge/-web demo-blue"></a>
<a href="https://egraphs-good.github.io/egglog/docs/egglog">
    <img alt="Main Branch Documentation" src="https://img.shields.io/badge/docs-main-blue"></a>
<a href="https://codspeed.io/egraphs-good/egglog">
    <img src="https://img.shields.io/endpoint?url=https://codspeed.io/badge.json" alt="CodSpeed Badge"/></a>
<a href="https://egraphs.zulipchat.com/#narrow/stream/375765-egglog">
    <img src="https://img.shields.io/badge/zulip-join%20chat-blue" alt="Zulip Chat"/></a>

This is the repo for the core of the `egglog` engine, which combines the power of equality saturation and Datalog.

For getting started, try out the [egglog tutorial](https://egraphs-good.github.io/egglog-tutorial)!

You can also [run egglog in your web browser](https://egraphs-good.github.io/egglog/) or check out [the documentation](https://egraphs-good.github.io/egglog/docs/egglog).

For a "battery-included" experience, we recommend [egglog-experimental](https://github.com/egraphs-good/egglog-experimental). It provides more features through additional `egglog` plugins.

If you want to cite `egglog`, please use [this citation](./CITATION.bib).

---

The following instructions are for using/developing the core directly.

## Prerequisites & compilation

Install [Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html).

```
git clone git@github.com:egraphs-good/egglog.git
cargo install --path=egglog
```

## Usage

The core can be used in REPL mode with:

```
cargo run --release
```

The standard mode processes an input file:

```
cargo run --release [-f fact-directory] [--to-dot] [--to-svg] [-j --threads <THREADS>] <files.egg>
```

* The `--to-dot` command will save a graphviz dot file at the end of the program, replacing the `.egg` extension with `.dot`.
* The `--to-svg`, which requires [Graphviz to be installed](https://graphviz.org/download/), will save a graphviz svg file at the end of the program, replacing the `.egg` extension with `.svg`.
* Set `RUST_LOG=INFO` to see more logging messages, as we use [env-logger](https://docs.rs/env_logger/latest/env_logger/) defaulting to `warn`.
* The `-j` option specifies the number of threads to use for parallel execution. The default value is `1`, which runs everything in a single thread. Passing `0` will use the maximum inferred parallelism available on the current system.

One can also use `egglog` as a Rust library by adding the following to your `Cargo.toml`:

```
[dependencies]
egglog = "1.0.0"
```

See also the [Python binding](https://github.com/egraphs-good/egglog-python) for using `egglog` in Python.

Egglog can also be compiled to WebAssembly, see [./wasm-example](./wasm-example) for more information.

## Development

To view documentation in a browser, run `cargo doc --open`.

To get started locally, install Rust, [cargo-insta](https://insta.rs/docs/cli/), and [cargo-nextest](https://nexte.st/docs/installation/pre-built-binaries/).

Run `make test` to run the core `egglog` tests.

If the snapshots are out of date, run `make update-snapshots` to update them.

## Community extensions

The community has maintained egglog extensions for IDEs. However, they are outdated at the time of writing.

* [@hatoo](https://github.com/hatoo) maintains an [egglog-language extension](https://marketplace.visualstudio.com/items?itemName=hatookov.egglog-language) in VS Code (just search for "egglog" in VS Code). (Outdated)
* [@segeljakt](https://github.com/segeljakt) maintains a [Neovim plugin](https://github.com/segeljakt/tree-sitter-egg) for egglog using tree-sitter. (Outdated)

## Benchmarks

All PRs use [`codspeed`](https://codspeed.io/) to evaluate the performance of a
change against a suite of micro-benchmarks. You should see a "performance
report" from codspeed a few minutes after posting a PR for review. Generally
speaking, PRs should only improve performance or leave it unchanged, though
exceptions are possible when warranted.

To debug performance issues, we recommend looking at the codspeed profiles, or
running your own using
[samply](https://github.com/mstange/samply),
[flamegraph-rs](https://github.com/flamegraph-rs/flamegraph),
[cargo-instruments](https://github.com/cmyr/cargo-instruments) (on MacOS) or
`perf` (on Linux). All codspeed benchmarks correspond to named `.egg` files,
usually in the `tests/` directory. For example, to debug an issue with
`extract-vec-bench`, you can run the following commands:

```bash
# install samply
cargo install --locked samply
# build a profile build which includes debug symbols
cargo build --profile profiling
# run the egglog file and profile
samply record ./target/profiling/egglog tests/extract-vec-bench.egg
# [optional] run the egglog file without logging or printing messages, which can help reduce the stdout
# when you are profiling extracting a large expression
env RUST_LOG=error samply record ./target/profiling/egglog --no-messages tests/extract-vec-bench.egg
```

## Parallelism

egglog has support to run programs in parallel
via the `-j` flag. This support is relatively new and most users just run
egglog single-threaded; the codspeed benchmarks only evaluate single-threaded
performance. However, please take care not to pessimize parallel performance
where possible (e.g. by adding coarse-grained locks).

We use rayon's global thread pool for parallelism, and the number of threads used
is set to `1` by default when egglog's CLI is run. If you use egglog as a library,
you can control the level of parallelism by setting rayon's `num_threads`.

### Codspeed specifics

We run all of our "examples" [as benchmarks in codspeed](https://codspeed.io/egraphs-good/egglog).
These are in CI for every commit in main and for all PRs. It will run the
examples with extra instrumentation added so that it can capture a single trace
of the CPU interactions
([src](https://docs.codspeed.io/features/understanding-the-metrics/)):

> CodSpeed instruments your benchmarks to measure the performance of your code.
> A benchmark will be run only once and the CPU behavior will be simulated.
> This ensures that the measurement is as accurate as possible, taking into
> account not only the instructions executed but also the cache and memory
> access patterns. The simulation gives us an equivalent of the CPU cycles that
> includes cache and memory access.

Since many of the shorter running benchmarks have unstable timings due to non
deterministic performance ([like in the memory
allocator](https://github.com/oxc-project/backlog/issues/89)), we
["ignore"](https://docs.codspeed.io/features/ignoring-benchmarks/) them in
codspeed. That way, we still capture their performance, but their timings don't
show up in our reports by default.

We use 50ms as our cutoff currently, any benchmarks shorter than that are
ignored. This number was selected to try to ignore any benchmarks with have
changes > 1% when they haven't been modified. Note that all the ignoring is
done manually, so if you add another example that's short, an admin on the
codspeed project will need to manually ignore it.

