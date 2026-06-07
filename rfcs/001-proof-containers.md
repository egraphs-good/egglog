# RFC 001: Container Proof Support

Status: Draft

## Summary

Add proof-mode support for fixed-shape Rust container sorts when container values appear inside eq-sort constructors. This first slice covers `Pair` and the non-`unstable-fn` pieces of `Either`: constructors, simple projections, proof views, and rebuild rules.

At the proof-object boundary, supported fixed-shape containers behave like ordinary constructors. Generated view and rebuild rules are internal machinery; extracted equality proofs for rebuilt container values lower to ordinary `Congr`/`Trans`/`Sym` proof terms over the container constructor symbol.

Direct non-eq container globals remain unsupported. Tests should wrap container values in eq-sort constructors when they need top-level names.

## Motivation

Proof mode treats constructors as the term language it can reason about. Rust-level containers such as `Pair` and `Either` are primitive/container values, so they need explicit proof metadata before the proof encoder can:

- recognize container constructor primitives as proof terms;
- generate view tables for canonicalized container rows;
- generate rebuild rules that produce congruence proofs;
- recognize projection primitives such as `pair-first` and `either-unwrap-left`.

The target shape is:

```scheme
(sort IntPair (Pair i64 i64))
(datatype Box (Box IntPair))

(let boxed (Box (pair 1 2)))
(prove (= boxed (Box (pair 1 2))))
```

and:

```scheme
(sort MathOrInt (Either Math i64))
(datatype Box (Box MathOrInt))

(let left (Box (either-left (A))))
(prove (= left (Box (either-left (A)))))
```

## Design

### Presort Support

Presorts advertise whether declarations using that presort can appear in proof mode:

```rust
pub trait Presort {
    fn supports_proof_encoding() -> bool {
        false
    }
}
```

`TypeInfo` records this during presort registration, so `file_supports_proofs` asks type information instead of hard-coding specific presort names in proof encoding helpers.

### Container Metadata

Instantiated sorts describe their proof-level constructors and projections:

```rust
pub struct ContainerProofSpec {
    pub constructors: Vec<ContainerProofConstructorSpec>,
}

pub struct ContainerProofConstructorSpec {
    pub name: &'static str,
    pub input_sorts: Vec<ArcSort>,
    pub projections: Vec<ContainerProofProjectionSpec>,
}

pub struct ContainerProofProjectionSpec {
    pub primitive: &'static str,
    pub field: usize,
}
```

`PairSort` has one constructor, `pair`, with projections `pair-first` and `pair-second`.

`EitherSort` has two constructors, `either-left` and `either-right`, with projections `either-unwrap-left` and `either-unwrap-right`.

The proof encoder asks the container sort for metadata, validates primitive input/output sorts against the selected constructor/projection, and emits the same term/view/rebuild machinery for any supported container.

### Primitive Validators

Supported container primitives use `add_primitive_with_validator!` so proof checking can reconstruct `TermDag` terms:

- `pair`, `pair-first`, `pair-second`
- `either-left`, `either-right`, `either-unwrap-left`, `either-unwrap-right`

This slice covers `Either` constructors and unwrap projections. `either-match` remains runtime-only because it depends on function containers.

### Proof Encoding

For each constructor in `container_proof_spec`, proof instrumentation generates:

- a view table keyed by that constructor's input sorts plus the container output;
- proof-carrying rows when proofs are enabled;
- congruence rules for rows with the same canonical children and different outputs;
- rebuild rules that canonicalize proof-canonicalizable children and the container output.

The same code handles one-constructor containers such as `Pair` and multi-constructor containers such as `Either`.

When a supported fixed-shape container constructor participates in a proof, the proof encoder keeps enough constructor structure to produce ordinary proof-format congruence:

```scheme
(Congr
  (= (Box (pair (A) 2)) (Box (pair (B) 2)))
  ...
  (Congr
    (= (pair (A) 2) (pair (B) 2))
    ...
    (Sym (= (A) (B)) ...)
    0)
  0)
```

For `Either`, this applies only within the same constructor head: `either-left` with `either-left`, or `either-right` with `either-right`. There is no constructor congruence proof between `either-left` and `either-right`.

Projection and unwrap facts may still be explained by the user rule or primitive validator that introduced them. The explicit congruence requirement is about equality of constructor-shaped container values, especially rebuild/canonicalization after a child equality.

### Proof Normal Form

Generic primitive calls still need proof-normal-form lifting when they contain constructor or function-call arguments. That keeps arbitrary primitive reasoning out of the proof term language.

Supported container constructor primitives are different: their `container_proof_spec` metadata says they are proof-level constructors. In proof mode, normal form preserves their constructor-shaped arguments enough for later instrumentation to build `Congr` proofs over `pair`, `either-left`, or `either-right`.

This exception is narrow:

- only primitives recognized as container constructors by the output sort's `container_proof_spec` qualify;
- projection primitives such as `pair-first` and `either-unwrap-left` do not become constructors;
- arbitrary primitives, including global primitive/container values, continue using the existing proof-normal-form restrictions.

The implementation treats proof-supported container constructor primitives closer to user datatypes, while still using the container view/rebuild tables for lookup and canonicalization.

### Unsupported Globals

Direct non-eq container globals remain unsupported:

```scheme
(sort IntPair (Pair i64 i64))
(let p (pair 1 2)) ; still unsupported with --proofs
```

Direct globals are out of scope for this RFC.

## Non-Goals

- Support containers beyond `Pair` and the constructor/projection pieces of `Either`.
- Support direct global primitive/container values in proof mode.
- Add a public user-land API for container proof specs.

## Rationale And Alternatives

### User-Land Datatypes Only

Users can model pairs and eithers as ordinary datatypes today, and those proofs already use constructor congruence. That is useful as a comparison target, but it does not help existing Rust-level container primitives such as `pair`, `pair-first`, `either-left`, and `either-unwrap-left`. This RFC keeps the proof object close to the user-land datatype shape while supporting the native container values.

### Hard-Code Pair And Either In Proof Encoding

The smallest implementation could special-case `Pair`, `pair`, `pair-first`, `Either`, and the either primitive names directly in proof encoding. The selected design puts those names in the sort implementations instead. Proof encoding consumes generic constructor/projection metadata, so another fixed-shape container can reuse the same proof-core path.

### Opaque Container Proof Rules

Another option is to explain rebuilt container equalities with a container-specific proof rule or Rust callback. For fixed-shape containers, ordinary constructor congruence is enough: a child proof can be lifted through `pair`, `either-left`, or `either-right`. Using the existing `Congr` proof shape also makes container-backed snapshots comparable to user-land datatype snapshots.

### Direct Container Globals In The First Slice

Direct globals for non-eq container values would be useful, but proof-mode global desugaring still does not support non-eq primitive/container let bindings. This RFC keeps those globals out of scope and uses eq-sort constructors as the supported top-level proof boundary.

### Broader Container Families

`Maybe` can likely follow the same fixed-shape constructor/projection pattern, but it is not needed for the first reviewable slice. `Map`, `Set`, and other reordered or normalized containers need a different proof story because positional `Congr` does not explain ordering, lookup, multiplicity, or semantic normalization.

### Custom Union-Find Backend

A proof-aware union-find backend may eventually simplify generated proof-mode machinery, but it is not required for Pair/Either proof support. The current design works through the existing view, rebuild, and proof-normal-form paths.

## Drawbacks And Risks

- This adds a small proof metadata API to container sorts, so future non-fixed-shape containers may need a different extension point.
- `Either` is only partially supported: constructors and unwrap projections are covered, but `either-match` is not.
- Proof mode remains inconsistent for direct non-eq globals until global primitive/container values get their own design.

## Acceptance Criteria

- Proof mode accepts `Pair` and supported `Either` sort declarations.
- Pair and Either constructor primitives can appear inside eq-sort constructors and produce proof snapshots.
- Pair projections and Either unwrap projections can be used in rules whose results are proved.
- Rebuilding after a child equality produces explicit `Congr` over `pair`, `either-left`, or `either-right`, not an opaque container proof.
- User-land datatype mirror tests produce comparable proof shapes for construction, projection, and rebuild behavior.
- Direct non-eq container globals remain unsupported in proof mode.

Current test coverage:

- `tests/proofs/pair-proof.egg`
- `tests/proofs/either-proof.egg`
- `tests/proofs/pair-userland-proof.egg`
- `tests/proofs/either-userland-proof.egg`
- `tests/snapshots/files__proof_unsupported_files.snap`

## Open Questions

- Is field-index projection metadata sufficient for future containers, or do richer eliminators need a different shape?
- What metadata shape would be needed to support `either-match` proofs?
- For future reordered containers such as `Map` and `Set`, what proof object should explain normalization, ordering, key lookup, and multiplicity? Positional `Congr` is sufficient for `Pair` and same-variant `Either`, but not for semantic container equality.
