# eggplant


`eggplant` is the High-Level Rust API repo for the `egglog` tool accompanying the paper
  "Better Together: Unifying Datalog and Equality Saturation"
  ([ACM DL](https://dl.acm.org/doi/10.1145/3591239), [arXiv](https://arxiv.org/abs/2304.04332)).

It is hard to do Graph operation directly on EGraph becasue EGraph is a highly compressed data structure.

Basing on that fact, `eggplant` provides out-of box Graph API that allow you to do revisions on EGraph intuitively.

There is also a Proc-Macro lib for users to quickly define a suite of DSL.

# How to define a DSL?


# How to do pattern rewrite?


It work as a middle-layer buffer for EGraph. Besides, there is a bunch of trait to help developers quickly do customization for EGraph, 


# Documentation

To view documentation, run `cargo doc --open`.

