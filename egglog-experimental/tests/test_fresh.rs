use egglog_experimental::*;

#[test]
fn test_fresh_desugaring() {
    // A simple test to check if desugaring works
    let program = r#"
(datatype MySort (MySortConstructor))

;; Test with a simple query pattern
(relation test_rel (MySort))
(MySortConstructor)
(rule ((MySortConstructor))
      ((test_rel (unstable-fresh! MySort))))

(run 1)

;; Check that the relation was populated
(check (test_rel ?x))
    "#;

    let mut egraph = new_experimental_egraph();
    let result = egraph.parse_and_run_program(None, program);
    match result {
        Ok(_) => (),
        Err(e) => {
            panic!("test_fresh_desugaring failed: {:?}", e);
        }
    }
}

#[test]
fn test_fresh_basic() {
    let program = r#"
(datatype Math
    (Let Math Math)
    (Var String))
    
(rule ((Let a b))
      ((Let (unstable-fresh! Math) b)))

(Let (Var "x") (Var "y"))
(run 1)

;; Check that a fresh value was created for the first argument
;; The rule replaces only the first arg, so ?a should be different from (Var "x")
(check (Let ?a (Var "y")) (!= ?a (Var "x")))
    "#;

    let mut egraph = new_experimental_egraph();
    let result = egraph.parse_and_run_program(None, program);
    match result {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Error: {:?}", e);
            panic!("test_fresh_basic failed: {:?}", e);
        }
    }
}

#[test]
fn test_fresh_unique() {
    let program = r#"
(datatype Math
    (Let Math Math)
    (Var String))
    
;; Rule with multiple fresh! calls
(rule ((Let a b))
      ((Let (unstable-fresh! Math) (unstable-fresh! Math))))

(Let (Var "x") (Var "y"))
(run 1)

;; Check that fresh values were created
(check (Let ?a ?b))
    "#;

    let mut egraph = new_experimental_egraph();
    let result = egraph.parse_and_run_program(None, program);
    match result {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Error: {:?}", e);
            panic!("test_fresh_unique failed: {:?}", e);
        }
    }
}

#[test]
fn test_fresh_multiple_sorts() {
    let program = r#"
(datatype Expr
    (Add Expr Expr)
    (Num i64))
    
(datatype TypedExpr
    (TNum i64 String)
    (TAdd TypedExpr TypedExpr String))

;; Rule that uses fresh! with different sorts in the query
(rule ((= e (Add (Num x) (Num y)))
       (TNum z ty))
      ((TAdd (unstable-fresh! TypedExpr) (unstable-fresh! TypedExpr) ty)))

;; Set up initial facts
(let expr1 (Add (Num 1) (Num 2)))
(TNum 42 "int")
(run 1)

;; Check that fresh values were created
(check (TAdd ?a ?b ?ty))
    "#;

    let mut egraph = new_experimental_egraph();
    let result = egraph.parse_and_run_program(None, program);
    match result {
        Ok(_) => (),
        Err(e) => {
            panic!("test_fresh_multiple_sorts failed: {:?}", e);
        }
    }
}

#[test]
fn test_fresh_with_cost() {
    // Test the :cost option - fresh with high cost should lose to lower cost alternative
    let program = r#"
(datatype MySort 
    (MySortConstructor)
    (LowCostAlt))  ;; default cost is 1

(MySortConstructor)

;; Create a fresh value with high cost and union with low-cost alternative
(rule ((MySortConstructor))
      ((union (unstable-fresh! MySort :cost 100) (LowCostAlt))))

(run 1)

;; Extract should give us the low-cost alternative
(extract (LowCostAlt))
    "#;

    let mut egraph = new_experimental_egraph();
    let result = egraph.parse_and_run_program(None, program);
    match result {
        Ok(outputs) => {
            let output = outputs
                .iter()
                .map(|o| o.to_string())
                .collect::<Vec<_>>()
                .join("\n");
            assert!(
                output.contains("LowCostAlt"),
                "Expected extraction to choose LowCostAlt due to lower cost, got: {}",
                output
            );
        }
        Err(e) => {
            panic!("test_fresh_with_cost failed: {:?}", e);
        }
    }
}

#[test]
fn test_fresh_with_unextractable() {
    // Test the :unextractable flag - unextractable fresh should not be extracted
    let program = r#"
(datatype MySort 
    (MySortConstructor)
    (ExtractableAlt))

(MySortConstructor)

;; Create an unextractable fresh value and union with extractable alternative
(rule ((MySortConstructor))
      ((union (unstable-fresh! MySort :unextractable) (ExtractableAlt))))

(run 1)

;; Extract should give us the extractable alternative
(extract (ExtractableAlt))
    "#;

    let mut egraph = new_experimental_egraph();
    let result = egraph.parse_and_run_program(None, program);
    match result {
        Ok(outputs) => {
            let output = outputs
                .iter()
                .map(|o| o.to_string())
                .collect::<Vec<_>>()
                .join("\n");
            assert!(
                output.contains("ExtractableAlt"),
                "Expected extraction to choose ExtractableAlt since fresh is unextractable, got: {}",
                output
            );
        }
        Err(e) => {
            panic!("test_fresh_with_unextractable failed: {:?}", e);
        }
    }
}

#[test]
fn test_fresh_extractable_by_default() {
    // Test that fresh without :unextractable flag is extractable
    // When fresh has lower cost, it should be extracted
    //
    // The query has no variables, so the fresh constructor only takes an i64 index.
    // Total cost = constructor cost (1) + i64 index arg (1) = 2
    // So we use cost 3 for the alternative (exactly 1 more than fresh)
    let program = r#"
(datatype MySort 
    (MySortConstructor)
    (AltConstructor :cost 3))

(MySortConstructor)

;; Create an extractable fresh value (default) and union with higher-cost alternative
(rule ((MySortConstructor))
      ((union (unstable-fresh! MySort :cost 1) (AltConstructor))))

(run 1)

;; Extract - should NOT choose the high-cost alternative
(extract (AltConstructor))
    "#;

    let mut egraph = new_experimental_egraph();
    let result = egraph.parse_and_run_program(None, program);
    match result {
        Ok(outputs) => {
            let output = outputs
                .iter()
                .map(|o| o.to_string())
                .collect::<Vec<_>>()
                .join("\n");
            // Should NOT contain the high-cost alternative
            assert!(
                !output.contains("AltConstructor"),
                "Expected extraction to choose fresh value (lower cost), got: {}",
                output
            );
        }
        Err(e) => {
            panic!("test_fresh_extractable_by_default failed: {:?}", e);
        }
    }
}

#[test]
fn test_fresh_default_cost_is_one() {
    // Test that fresh without :cost has default cost of 1
    //
    // The query has no variables, so the fresh constructor only takes an i64 index.
    // Total cost = constructor cost (1) + i64 index arg (1) = 2
    // So we use cost 3 for the alternative (exactly 1 more than fresh)
    let program = r#"
(datatype MySort 
    (MySortConstructor)
    (HighCostAlt :cost 3))

(MySortConstructor)

;; Create a fresh value with default cost (1) and union with higher-cost alt
(rule ((MySortConstructor))
      ((union (unstable-fresh! MySort) (HighCostAlt))))

(run 1)

;; Extract - should NOT choose the high-cost alternative
(extract (HighCostAlt))
    "#;

    let mut egraph = new_experimental_egraph();
    let result = egraph.parse_and_run_program(None, program);
    match result {
        Ok(outputs) => {
            let output = outputs
                .iter()
                .map(|o| o.to_string())
                .collect::<Vec<_>>()
                .join("\n");
            // Should NOT contain the high-cost alternative
            assert!(
                !output.contains("HighCostAlt"),
                "Expected extraction to choose fresh value (default cost 1) over cost-3 alternative, got: {}",
                output
            );
        }
        Err(e) => {
            panic!("test_fresh_default_cost_is_one failed: {:?}", e);
        }
    }
}
