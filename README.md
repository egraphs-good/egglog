# egglog

<a href="https://egraphs-good.github.io/egglog/">
    <img alt="Web Demo" src="https://img.shields.io/badge/-web demo-blue"></a>
<a href="https://egraphs-good.github.io/egglog/docs/egglog">
    <img alt="Main Branch Documentation" src="https://img.shields.io/badge/docs-main-blue"></a>

This is repo for the `egglog` tool accompanying the paper
  "Better Together: Unifying Datalog and Equality Saturation"
  ([ACM DL](https://dl.acm.org/doi/10.1145/3591239), [arXiv](https://arxiv.org/abs/2304.04332)).

If you use this work, please use [this citation](./CITATION.bib).

See also the Python binding, which provides a bit more documentation:
https://egglog-python.readthedocs.io/en/latest/

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
cargo run [-f fact-path] [-naive] [--to-json] [--to-dot] [--to-svg] <files.egg>
```

or just

```
cargo run
```

for the REPL.

* The `--to-dot` command will save a graphviz dot file at the end of the program, replacing the `.egg` extension with `.dot`.
* The `--to-svg`, which requires [Graphviz to be installed](https://graphviz.org/download/), will save a graphviz svg file at the end of the program, replacing the `.egg` extension with `.svg`.


## VS Code plugin

There is a VS Code extension in the vscode folder. Install using 'Install from VSIX...' in the three-dot menu of the extensions tab and pick `vscode/eggsmol-1.0.0/eggsmol-1.0.0.vsix`.

### Enhancing the VS code extension

If you want to hack on the VS Code extension, install nodejs, and make your changes in the files in the `vscode/egglog-1.0.0` folder.

Then run

```
code vscode/eggsmol-1.0.0
```

and use F5 to run the extension in a new window. When satisfied, then install VSCE if you do not already have it:

```
npm install -g @vscode/vsce
```

Run `vsce package` in the `vscode/eggsmol-1.0.0` folder to reconstruct the .vsix file and install it manually.

## Development

To run the tests use `make test`.

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
