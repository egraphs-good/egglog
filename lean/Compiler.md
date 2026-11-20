# Writing a verified compiler pass

The compiler API transforms an inductive Lean AST and returns a proof that the output and input have the same denotation for every assignment to their free variables. The Boolean example defines a register-based IR. The strength reduction example defines natural-number arithmetic with a variable-count shift. [Boolean IR](Egglog/Examples/Boolean.lean) · [Strength reduction](Egglog/Examples/StrengthReduction.lean)

## Define a language in Lean

Import `Egglog.DSL` to define an optimizer in a Lean module. For a new language, give the register constructor and each fixed-arity operator its denotation once.

```lean
egg_language Expr using Ops : Nat with_vars reg where
  | maximum x y => max x y
  | next x => x + 1
  | copy x => x
  | yes => 1
  | inBounds i n => if i < n then 1 else 0
  otherwise 0
```

This generates the inductive `Expr`, its interpreter, e-graph pattern constructors, the codec and its semantic agreement theorem. Use `Expr.language` directly or pass it a model extended with tables. The runnable bounds-check pass uses `Expr.language (F.model (R.model Ops.model))`. For an established compiler IR, keep its own interpreter and connect it as below. [Complete module](Egglog/Examples/BoundsCheck.lean)

## Connect the IR to its semantics

Define an interpreter for the existing AST, for example `Expr.eval (regs : Nat → Bool) : Expr → Bool`. An `egg_signature` declaration gives each e-graph operator a pattern constructor, a fixed arity and a meaning. A `costing` annotation sets its local extraction cost. The default cost is one.

```lean
egg_signature Ops : Bool where
  | yes => true
  | no => false
  | neg x => !x
  | conj a b => a && b
  | disj a b => a || b
  | mux c a b => if c then a else b
  otherwise false
```

The fallback value interprets malformed operator applications in the total semantic model. For a regular inductive IR whose constructors and signature operators share names, `egg_codec encode, decode : Expr using Ops` generates both directions and the decoder's inverse proof. It reads the recursive fields from the inductive type and uses the signature's operators, so the AST and signature can declare constructors in different orders.

In the Boolean IR, `neg`, `conj`, `disj` and `mux` are generated this way. Only the payload encodings need clauses. A register index becomes a free variable, and the two Boolean literals select different nullary operators.

```lean
namespace Expr

egg_codec encode, decode : Expr using Ops where
  | reg i <=> .var i
  | lit true <=> Ops.yes
  | lit false <=> Ops.no

end Expr
```

The automatic form supports a single, non-indexed inductive with explicit fields and direct recursion, including specialized types such as `Control Nat`. An override may mix recursive children and payloads, reorder operands, or choose a different operator. The explicit `egg_codec … where` form also accepts a complete list of bidirectional clauses with the recursive children marked in brackets.

Connect these components in an `Egglog.Compiler.Language`. Its `encode_sound` field is the language author's one-time proof that interpreting the encoded tree agrees with the AST interpreter. Structural induction suffices for both examples. The generated decoder checks operator arities and constructs the inverse proof on each successful decode. A custom decoder can implement `Compiler.Decoder` directly or use `Decoder.checked` to check its reconstruction.

Each `Language` supplies one semantic carrier and one source type. A compiler can extend its constructor model with named relations and partial functions. The same certified e-graph holds the AST, relation rows and function values throughout the schedule. Both `Egglog.DSL` and the umbrella import `Egglog` expose the typed session API.

## Prove and schedule rewrites

A single `egg_rule` expression constructs the executable patterns and checks their semantic equality. Inside its proof, each named metavariable denotes an arbitrary value of the semantic carrier.

```lean
def sameBranch : VerifiedRule Ops.model :=
  egg_rule (c, x) : Ops.mux c x x ==> x by exact ite_self x
```

A `when` clause introduces an equality premise and can bind a new pattern variable. The strength reduction rule matches a multiplier's e-class against a power of two. It can discover the exponent after an earlier rule proves that a different expression for the multiplier is equal to that power.

```lean
def mulPow2 : GuardedRule Ops.model :=
  egg_rule (x, y, n) : Ops.times x y ==> Ops.shl x n when (h : y = Ops.pow2 n) by
    change y = 2 ^ n at h
    change x * y = x <<< n
    rw [h, Nat.shiftLeft_eq]
```

### Analysis tables and their proofs

The bounds-check compiler records `atMost i k` in a relation. It also keeps one value `k` for each key `n` in the partial function `lower`. Reading that cell gives a rule the proposition `k < n`. Two writes to the same key can carry different, valid lower bounds. Their maximum is the stronger bound. [Runnable bounds-check compiler](Egglog/Examples/BoundsCheck.lean)

```lean
egg_relations R : Nat where
  | atMost i k => i ≤ k

egg_functions F : Nat over Ops.model where
  | lower n ↦ v => v < n
    merge old new => Ops.maximum old new
    preserving ho hn by exact Nat.max_lt.mpr ⟨ho, hn⟩
```

Use `F.model (R.model Ops.model)` in the language definition. The function declaration takes its keys and values from the language's semantic carrier. Each write proves the declared invariant. The merge is a Lean function on value terms and may construct an e-node. Its preservation proof covers every valuation of the free registers. It also applies when a later e-class union makes two previously distinct keys equal. The propositions erase in native code. The table still holds the current value, and a rule lookup checks the matched entry.

A rule can assert a relation with `yield`, write a function with `set`, or justify a union. Within a `do { ... }` head each action has its own proof and runs in order for the same substitution. The example finds a term `next (maximum x y)` and writes both `x` and `y` as lower bounds. A consumer joins the check itself, the upper-bound relation and the current lower-bound value.

```lean
def checkFromLower : RelRule ir.model :=
  egg_rule (i, k, n) : when (upper : R.atMost i k) (lower : F.lower n ↦ k)
    rewrite inBounds i n ==> yes by
    change truth (i < n) = 1
    exact if_pos (Nat.lt_of_le_of_lt (show i ≤ k from upper) (show k < n from lower))
```

The witness `k` comes from the tables. A missing function entry produces no match. A read supplies its invariant about the value at match time, which remains true even if a later write chooses a stronger bound. `rewrite lhs ==> rhs` matches an existing `lhs`. A head `union lhs = rhs` can introduce both terms. Rules sharing a stage search one snapshot and their heads update the graph for the next round. A host compiler can also put an independently proved fact into `ir.session` and read back a value with its invariant. [Typed host example](RelationalSessionTests.lean) · [Function collision and update tests](RelationalFunctionTests.lean)

### Running rulesets

Rulesets are Lean arrays. A schedule runs all of its stages on one e-graph. In the Boolean pass, `control` checks for equivalent branches and `algebra` can establish their equivalence.

```lean
def phases : Certified.Schedule ir.model := egg_schedule {
  run control,
  saturate 4 { run algebra, run control }
}
```

| Stage | Execution |
| --- | --- |
| `run rules` | Run one round of the given unconditional rules. |
| `guarded rules` | Run one round of rules with equality premises. All premises share a substitution. |
| `relational rules` | Run one round of positive rules over constructors, named relations and partial functions. |
| `analyze values` | Propagate an exact value through e-classes using verified constructor transfers, merge equal results and rebuild congruences. |
| `repeat n { ... }` | Execute the enclosed sequence up to `n` times, stopping early at a fixed point. |
| `saturate n { ... }` | Saturate with an explicit iteration limit. |
| `saturate { ... }` | Continue until a fixed point. Expanding rules can diverge. |
| `until (x) (lhs = rhs) ... { ... }` | Skip the sequence when all equalities, relation atoms and function lookups have a simultaneous match. Repeating this stage rechecks the facts before each iteration. |

A `Certified.ValueAnalysis m β` supplies a partial transfer function and a meaning for each analysis value. For every successful transfer the caller proves that interpreting the operator on its children gives the meaning of the result. The engine can then merge classes with equal values. ParaBit uses 64-bit truth tables as `β` and functions from Boolean assignments to single bits as their meaning. [Working analysis](Egglog/Examples/ParaBitDecision.lean)

A custom match scheduler can choose which substitutions fire and carry state between rounds. Wrap proved rewrites with `Certified.NamedRule.plain`, or pair names with guarded rules. The `with_scheduler rules with policy for limit` stage calls `policy.choose` with the rule name, current graph statistics and candidate substitutions. Selected indices fire once and the remaining substitutions are available in the next round. The default `canStop` waits for the pending selections to empty before stopping on an unchanged graph. The scheduler regression postpones a match on the first round and applies it on the second. [Example](CompilerTests.lean)

## Use the result in a compiler or theorem

`ir.optimize phases input cost` returns a dependent result with an `output` AST and an erased `sound` proof. Omit `cost` for unit operator costs or pass `Ops.cost` for declared costs. If the decoder rejects an extracted tree, the total pass returns the input. Use `ir.optimize?` to observe this case as `none`.

```lean
def optimize (input : Expr) : Expr := (ir.optimize phases input).output

theorem optimize_correct (input : Expr) (regs : Nat → Bool) :
    (optimize input).eval regs = input.eval regs := by
  unfold optimize
  exact (ir.optimize phases input).sound regs
```

Use `ir.proveEq?` to seed both programs in one graph and obtain an optional proof of their equivalence. For a closed query with a bounded schedule, `egg_prove` asks Lean's kernel to reduce the query. Its result is a theorem over every register assignment, including inputs unknown when compiling.

```lean
theorem foldedReg : ir.Equivalent (.conj (.reg 7) (.lit true)) (.reg 7) := by
  egg_prove ir using (egg_schedule { run #[andTrue] })
```

The optimizer correctness theorem applies to the entire pass and requires no per-input kernel reduction. Proof erasure leaves the e-graph search and extraction as the runtime computation. Both examples, rejected rewrites, incorrect decoders, multi-pattern guards and schedule stop conditions are exercised by the compiler test suite. [Compiler tests](CompilerTests.lean)

For a module with several output roots, add each input through `ir.session.add`, run the schedule once, and create `session.extractor.batch`. Each `batch.extract? handle` returns an updated batch and an optional source-language result with its equivalence proof. Passing the returned batch to the next extraction shares the cost computation and the recovered subtrees across roots. The Llama case uses this approach on 225 projections. [Typed session API](Egglog/Session.lean) · [Luminal and eggcc case studies](bench/Advanced.md)

## Reproduce the compiler comparison

From the repository root, build the two native executables and run both workloads.

```sh
cargo build --release --locked --example lean_comparison
(cd lean && lake build compiler-bench)
python3 lean/bench/compiler.py --size 1024 --runs 7 --warmup 2 --enforce
python3 lean/bench/compiler.py --size 4096 16384 65536 --runs 5 --warmup 1 --csv large-compiler.csv
```

Each sample starts from an in-memory IR with free registers. The timer includes insertion into the e-graph, all scheduled rule applications, minimum-cost extraction and decoding back into the native IR. Both runners compare the result with the same expected tree after stopping the clock. The Rust runner uses the crate's `Write::add` constructor API through a registered primitive. It assigns the `Reg` constructor zero cost because egglog charges one for its primitive register index. The resulting register expression costs one in both languages. The harness checks host load before each workload, interleaves fresh single-threaded processes and applies the 1.30× threshold separately to each median. [Comparison harness](bench/compiler.py) · [Lean runner](CompilerBench.lean) · [Rust runner](../examples/lean_comparison/compiler.rs)

Size counts separately rewritable leaves of a balanced tree. At 65,536 leaves the Boolean input contains 720,895 constructor occurrences and extracts to 131,071. Strength reduction takes 327,678 constructor occurrences to 262,143. The native pass timer includes Lean's release of its private graph. Rust releases its graph after the result check. Each CSV row also gives whole-process peak RSS in bytes, elapsed time and CPU time including parsing and cleanup. If the pass timer exceeds twice the entire child's CPU time plus 250 ms, the harness records the interrupted pair and repeats both engines. Status and trial fields distinguish retries and warmups. [Large compiler samples](bench/results/large-compiler.csv)
