# egglog

<a href="https://egraphs-good.github.io/egglog/">
    <img alt="Web Demo" src="https://img.shields.io/badge/-web demo-blue"></a>
<a href="https://egraphs-good.github.io/egglog/docs/egglog">
    <img alt="Main Branch Documentation" src="https://img.shields.io/badge/docs-main-blue"></a>
<a href="https://codspeed.io/egraphs-good/egglog">
    <img src="https://img.shields.io/endpoint?url=https://codspeed.io/badge.json" alt="CodSpeed Badge"/></a>

This is the repo for the `egglog` tool accompanying the paper
  "Better Together: Unifying Datalog and Equality Saturation"
  ([ACM DL](https://dl.acm.org/doi/10.1145/3591239), [arXiv](https://arxiv.org/abs/2304.04332)).

If you use this work, please use [this citation](./CITATION.bib).

See also the Python binding, which provides a bit more documentation:
https://egglog-python.readthedocs.io/

## Chat

There is a Zulip chat about egglog here:
https://egraphs.zulipchat.com/#narrow/stream/375765-egglog

## Prerequisites & compilation

```
apt-get install make cargo
cargo install cargo-nextest
make all
```


## Usage

```
cargo run [-f fact-path] [--to-json] [--to-dot] [--to-svg] <files.egg>
```

or just

```
cargo run
```

for the REPL.

* The `--to-dot` command will save a graphviz dot file at the end of the program, replacing the `.egg` extension with `.dot`.
* The `--to-svg`, which requires [Graphviz to be installed](https://graphviz.org/download/), will save a graphviz svg file at the end of the program, replacing the `.egg` extension with `.svg`.


## Community extensions

* [@hatoo](https://github.com/hatoo) maintains an [egglog-language extension](https://marketplace.visualstudio.com/items?itemName=hatookov.egglog-language) in VS Code (just search for "egglog" in VS Code).
* [@segeljakt](https://github.com/segeljakt) maintains a [Neovim plugin](https://github.com/segeljakt/tree-sitter-egg) for egglog using tree-sitter.

## Development

To run the tests use `make test`.

## Benchmarks

We run all of our "examples" [as benchmarks in codspeed](https://codspeed.io/egraphs-good/egglog). These are in CI
for every commit in main and for all PRs. It will run the examples with extra instrumentation added so that it can
capture a single trace of the CPU interactions ([src](https://docs.codspeed.io/features/understanding-the-metrics/)):

> CodSpeed instruments your benchmarks to measure the performance of your code. A benchmark will be run only once and the CPU behavior will be simulated. This ensures that the measurement is as accurate as possible, taking into account not only the instructions executed but also the cache and memory access patterns. The simulation gives us an equivalent of the CPU cycles that includes cache and memory access.

Since many of the shorter running benchmarks have unstable timings due to non deterministic performance ([like in the memory allocator](https://github.com/oxc-project/backlog/issues/89)),
we ["ignore"](https://docs.codspeed.io/features/ignoring-benchmarks/) them in codspeed. That way, we still
capture their performance, but their timings don't show up in our reports by default.

We use 50ms as our cutoff currently, any benchmarks shorter than that are ignored. This number was selected to try to ignore
any benchmarks with have changes > 1% when they haven't been modified. Note that all the ignoring is done manually,
so if you add another example that's short, an admin on the codspeed project will need to manually ignore it.

# Documentation

To view documentation, run `cargo doc --open`.



TODO migrate the following documentation to cargo doc:
### Sort: i64

Signed 64-bit integers supporting these primitives:

```
+ - * / %           ; arithmetic
& | ^ << >> not-i64 ; bit-wise operations
< > <= >=           ; comparisons
min max log2
to-f64
to-string
```

### Sort: f64

64-bit floating point numbers supporting these primitives:

```
+ - * / %           ; arithmetic
< > <= >=           ; comparisons
min max neg
to-i64
to-string
```

### Sort: map

A map from a key type to a value type supporting these primitives:

```
empty
insert
get
not-contains
contains
set-union
set-diff
set-intersect
map-remove
```

### Sort: rational

Rational numbers (fractions) with 64-bit precision for numerator and denominator with these primitives:

```
+ - * /         ; arithmetic
min max neg abs floor ceil round
rational        ; construct from a numerator and denominator
numer denom     ; get numerator and denominator
pow log sqrt
< > <= >=       ; comparisons
```

These primitives are only defined when the result itself is a pure rational.

### Sort: string

Use double quotes to get a quote: `"Foo "" Bar"` is `Foo " Bar`.
No primitives defined.
