
A class is productive here when it contains a **finite ground constructor term**.
For example, `x = f(x)` has no such representative; adding `A()` to `x` supplies
one. This example uses the JSON input API so it also works without egglog:

```
use egraph_comparison::{Database, Certificate, certificate, verify};
let cyclic: Database = serde_json::from_value(serde_json::json!({
    "version": 1,
    "classes": {"x": {"sort": "E"}},
    "functions": {
        "A": {"kind": "constructor", "inputs": [], "output": "E"},
        "f": {"kind": "constructor", "inputs": ["E"], "output": "E"}
    },
    "rows": [{"function": "f", "inputs": ["x"], "output": "x"}]
}))?;
let mut grounded = cyclic.clone();
grounded.rows.push(egraph_comparison::Row {
    function: "A".into(), inputs: vec![], output: "x".into(), subsumed: false,
});
let witness = certificate(&grounded, &cyclic)?.unwrap();
assert!(matches!(witness, Certificate::MissingTerm { .. }));
assert!(verify(&witness, &grounded, &cyclic)?);
assert!(certificate(&cyclic, &cyclic)?.is_none());
# Ok::<(), Box<dyn std::error::Error>>(())
```
