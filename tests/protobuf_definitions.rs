use egglog::{
    builtin::definitions::{reconcile_declarations, reconcile_rulesets, retain_rulesets},
    proto as pb,
};

#[test]
fn structural_sort_aliases_merge_and_survive_alias_only_relocation() {
    let mut destination = pb::Program::default();
    for (symbol, reversed) in [("A", false), ("Z", true), ("A", true)] {
        let scalar = pb::Sort {
            kind: Some(pb::sort::Kind::Family(pb::HostSort {
                name: "i64".into(),
                args: vec![],
            })),
            ..Default::default()
        };
        let applied = pb::Sort {
            kind: Some(pb::sort::Kind::Family(pb::HostSort {
                name: "Vec".into(),
                args: vec![u32::from(reversed)],
            })),
            egglog_aliases: vec![pb::EgglogTypeBinding {
                symbol: symbol.into(),
            }],
            ..Default::default()
        };
        let source = pb::Program {
            ir_version: 1,
            sorts: if reversed {
                vec![applied, scalar]
            } else {
                vec![scalar, applied]
            },
            ..Default::default()
        };
        reconcile_declarations(&source, &mut destination).unwrap();
        assert_eq!(
            destination.sorts.len(),
            2,
            "alias-only sort and child survive"
        );
    }
    assert_eq!(
        destination.sorts[1]
            .egglog_aliases
            .iter()
            .map(|a| a.symbol.as_str())
            .collect::<Vec<_>>(),
        ["A", "Z"]
    );
    let mut plain = destination.sorts.clone();
    plain[1].egglog_aliases.clear();
    assert_eq!(
        egglog::builtin::import_sort(&plain, 1, &mut destination.sorts).unwrap(),
        1
    );
    assert_eq!(destination.sorts.len(), 2);
    retain_rulesets(&mut destination, &[]).unwrap();
    assert_eq!(destination.sorts.len(), 2);
    assert_eq!(destination.sorts[1].egglog_aliases.len(), 2);
}

#[test]
fn compiler_sort_aliases_share_types_without_registering_native_instances() {
    let mut graph = egglog::EGraph::default();
    graph
        .parse_and_run_program(None, "(sort A (Vec i64))")
        .unwrap();
    graph
        .bind_source_sort_alias("Z", "A", egglog::span!())
        .unwrap();
    assert_eq!(graph.type_info().get_sort_by_name("Z").unwrap().name(), "A");
    assert_eq!(
        graph
            .type_info()
            .get_arcsorts_by(|sort| sort.name() == "A")
            .len(),
        1
    );
    let command = graph
        .parse_program(None, "(function same (Z) A :merge new)")
        .unwrap()
        .remove(0);
    graph.resolve_command_preserving_globals(command).unwrap();
    graph.parse_and_run_program(None, "(push)").unwrap();
    graph
        .bind_source_sort_alias("Scoped", "A", egglog::span!())
        .unwrap();
    graph.parse_and_run_program(None, "(pop)").unwrap();
    assert!(graph.type_info().get_sort_by_name("Scoped").is_none());
    graph
        .bind_source_sort_alias("Scoped", "A", egglog::span!())
        .unwrap();
    assert!(
        graph
            .bind_source_sort_alias("Z", "A", egglog::span!())
            .is_err()
    );
    assert!(
        graph
            .bind_source_sort_alias("same", "A", egglog::span!())
            .is_err()
    );
    assert!(
        graph
            .bind_source_sort_alias("Missing", "unknown", egglog::span!())
            .is_err()
    );
    graph.parse_and_run_program(None, "(sort Nominal)").unwrap();
    assert!(
        graph
            .bind_source_sort_alias("Other", "Nominal", egglog::span!())
            .is_err()
    );
    assert!(graph.parse_and_run_program(None, "(sort Z)").is_err());
    assert!(
        graph
            .parse_and_run_program(None, "(function Z () i64 :merge new)")
            .is_err()
    );
    graph
        .parse_and_run_program(None, "(sort B (Vec i64))")
        .unwrap();
    assert_eq!(graph.type_info().get_sort_by_name("B").unwrap().name(), "B");
    assert_eq!(
        graph
            .type_info()
            .get_arcsorts_by(|sort| matches!(sort.name(), "A" | "B"))
            .len(),
        2
    );
}

// Ordered graph fixtures deliberately share the same rule body/index: only
// ruleset occurrence topology distinguishes the negative cases.
fn occurrence_fixture(children: &[&[u32]]) -> pb::Program {
    let mut program = pb::Program {
        ir_version: 1,
        rules: vec![pb::RuleDecl {
            kind: Some(pb::rule_decl::Kind::Rule(pb::Rule::default())),
            eval_mode: pb::RuleEvalMode::Seminaive.into(),
            ..Default::default()
        }],
        rulesets: (0..2)
            .map(|_| pb::Ruleset {
                kind: Some(pb::ruleset::Kind::Rules(pb::RuleList { rules: vec![0] })),
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    };
    for children in children {
        program.rulesets.push(pb::Ruleset {
            name: Some("parent".into()),
            kind: Some(pb::ruleset::Kind::Combined(pb::CombinedRuleset {
                rulesets: children
                    .iter()
                    .map(|i| pb::RulesetRef {
                        kind: Some(pb::ruleset_ref::Kind::Index(*i)),
                    })
                    .collect(),
            })),
            ..Default::default()
        });
    }
    program
}

#[test]
fn ruleset_duplicate_presentations_preserve_global_topology() {
    for children in [
        vec![&[0, 1][..], &[1, 0][..]],
        vec![&[0, 0][..], &[0, 1][..]],
    ] {
        let mut destination = pb::Program::default();
        let previous = destination.clone();
        assert!(reconcile_rulesets(&occurrence_fixture(&children), &mut destination).is_err());
        assert_eq!(destination, previous);
    }
    for children in [vec![&[0, 1][..], &[0, 1][..]], vec![&[0][..], &[1][..]]] {
        let mut destination = pb::Program::default();
        let map = reconcile_rulesets(&occurrence_fixture(&children), &mut destination).unwrap();
        assert_eq!(map.rulesets[2], map.rulesets[3]);
        assert_eq!(destination.rules.len(), 1);
        assert_eq!(map.rulesets[0] == map.rulesets[1], children[0].len() == 1);
        let previous = destination.clone();
        reconcile_rulesets(&occurrence_fixture(&children), &mut destination).unwrap();
        assert_eq!(destination, previous);
    }
    let mut distinct_rules = occurrence_fixture(&[&[0], &[1]]);
    distinct_rules.rules.push(distinct_rules.rules[0].clone());
    let Some(pb::ruleset::Kind::Rules(b)) = &mut distinct_rules.rulesets[1].kind else {
        unreachable!()
    };
    b.rules[0] = 1;
    assert!(reconcile_rulesets(&distinct_rules, &mut pb::Program::default()).is_err());
}

#[test]
fn ruleset_four_distinct_leaves_match_duplicate_presentations_in_both_orders() {
    for reverse in [false, true] {
        let mut source = occurrence_fixture(&[]);
        source.rulesets.extend(source.rulesets.clone()); // distinct A, B, C, D
        for children in [[0, 1], [2, 3]] {
            source.rulesets.push(pb::Ruleset {
                name: Some("parent".into()),
                kind: Some(pb::ruleset::Kind::Combined(pb::CombinedRuleset {
                    rulesets: children
                        .into_iter()
                        .map(|i| pb::RulesetRef {
                            kind: Some(pb::ruleset_ref::Kind::Index(i)),
                        })
                        .collect(),
                })),
                ..Default::default()
            });
        }
        if reverse {
            source.rulesets.swap(4, 5);
        }
        let mut destination = pb::Program::default();
        let map = reconcile_rulesets(&source, &mut destination).unwrap();
        assert_eq!(map.rulesets[0], map.rulesets[2]);
        assert_eq!(map.rulesets[1], map.rulesets[3]);
        assert_ne!(map.rulesets[0], map.rulesets[1]);
        assert_eq!(map.rulesets[4], map.rulesets[5]);
        assert_eq!(destination.rulesets.len(), 3);
        assert_eq!(destination.rules.len(), 1);
        let previous = destination.clone();
        source.rulesets.swap(4, 5);
        let resent = reconcile_rulesets(&source, &mut destination).unwrap();
        assert_eq!(resent.rulesets, map.rulesets);
        assert_eq!(destination, previous);
    }
}

#[test]
fn ruleset_global_rule_sharing_rejects_split_and_merge_in_both_orders() {
    let mut shared = occurrence_fixture(&[&[0], &[1]]);
    shared.rulesets[3].name = Some("other".into());
    let mut distinct = shared.clone();
    distinct.rules.push(distinct.rules[0].clone());
    let Some(pb::ruleset::Kind::Rules(b)) = &mut distinct.rulesets[1].kind else {
        unreachable!()
    };
    b.rules[0] = 1;
    for (installed, conflicting) in [(&shared, &distinct), (&distinct, &shared)] {
        for reverse in [false, true] {
            let mut destination = pb::Program::default();
            let map = reconcile_rulesets(installed, &mut destination).unwrap();
            assert_ne!(map.rulesets[0], map.rulesets[1]); // never a child-sharing mismatch
            retain_rulesets(&mut destination, &[map.rulesets[2], map.rulesets[3]]).unwrap();
            let previous = destination.clone();
            let mut control = installed.clone();
            let mut invalid = conflicting.clone();
            if reverse {
                control.rulesets.swap(2, 3);
                invalid.rulesets.swap(2, 3);
            }
            reconcile_rulesets(&control, &mut destination).unwrap();
            assert_eq!(destination, previous);
            let error = reconcile_rulesets(&invalid, &mut destination).unwrap_err();
            assert_eq!(error, "conflicting rule sharing");
            assert_eq!(destination, previous);
        }
    }
}

#[test]
fn ruleset_installed_roots_match_simultaneously_without_merging_or_splitting() {
    let mut original = occurrence_fixture(&[&[0], &[0]]);
    original.rulesets[3].name = Some("other".into());
    let mut destination = pb::Program::default();
    reconcile_rulesets(&original, &mut destination).unwrap();
    let previous = destination.clone();
    let mut split = original.clone();
    let Some(pb::ruleset::Kind::Combined(c)) = &mut split.rulesets[3].kind else {
        unreachable!()
    };
    c.rulesets[0].kind = Some(pb::ruleset_ref::Kind::Index(1));
    assert!(reconcile_rulesets(&split, &mut destination).is_err());
    assert_eq!(destination, previous);

    let mut destination = pb::Program::default();
    reconcile_rulesets(&split, &mut destination).unwrap();
    let previous = destination.clone();
    assert!(reconcile_rulesets(&original, &mut destination).is_err());
    assert_eq!(destination, previous);
    // Reversing the two submitted roots must not change either rejection.
    original.rulesets.swap(2, 3);
    assert!(reconcile_rulesets(&original, &mut destination).is_err());
    assert_eq!(destination, previous);
    let mut destination = pb::Program::default();
    reconcile_rulesets(&original, &mut destination).unwrap();
    let previous = destination.clone();
    split.rulesets.swap(2, 3);
    assert!(reconcile_rulesets(&split, &mut destination).is_err());
    assert_eq!(destination, previous);
}

#[test]
fn ruleset_retained_name_reference_and_matched_local_child_are_one_occurrence() {
    let mut original = occurrence_fixture(&[&[0], &[0]]);
    original.rulesets[0].name = Some("child".into());
    original.rulesets[3].name = Some("other".into());
    let mut destination = pb::Program::default();
    let map = reconcile_rulesets(&original, &mut destination).unwrap();
    retain_rulesets(
        &mut destination,
        &[map.rulesets[0], map.rulesets[2], map.rulesets[3]],
    )
    .unwrap();
    let mut resupply = original;
    resupply.rulesets[0].name = None;
    let Some(pb::ruleset::Kind::Combined(other)) = &mut resupply.rulesets[3].kind else {
        unreachable!()
    };
    other.rulesets[0].kind = Some(pb::ruleset_ref::Kind::Name("child".into()));
    let map = reconcile_rulesets(&resupply, &mut destination).unwrap();
    assert_eq!(
        destination.rulesets[map.rulesets[0]].name.as_deref(),
        Some("child")
    );
}

#[test]
fn ruleset_names_cycles_options_and_retention_obey_occurrence_contract() {
    let source = occurrence_fixture(&[&[0]]);
    let mut destination = pb::Program::default();
    let map = reconcile_rulesets(&source, &mut destination).unwrap();
    retain_rulesets(&mut destination, &[map.rulesets[2]]).unwrap();
    assert_eq!(destination.rulesets.len(), 2); // unused anonymous B is transient
    assert_eq!(destination.rules.len(), 1);
    let retained_child = 0;

    let mut named = source.clone();
    named.rulesets[0].name = Some("child".into());
    let map = reconcile_rulesets(&named, &mut destination).unwrap();
    assert_eq!(map.rulesets[0], retained_child);
    retain_rulesets(&mut destination, &[map.rulesets[0], map.rulesets[2]]).unwrap();
    let previous = destination.clone();
    let mut conflict = named.clone();
    conflict.rulesets[0].name = Some("renamed".into());
    assert!(reconcile_rulesets(&conflict, &mut destination).is_err());
    assert_eq!(destination, previous);
    for option in 0..3 {
        let mut conflict = named.clone();
        match option {
            0 => conflict.rules[0].eval_mode = pb::RuleEvalMode::Naive.into(),
            1 => conflict.rules[0].no_decomp = true,
            _ => conflict.rules[0].include_subsumed = true,
        }
        assert!(reconcile_rulesets(&conflict, &mut destination).is_err());
        assert_eq!(destination, previous);
    }
    let map = reconcile_rulesets(&source, &mut destination).unwrap();
    assert_eq!(
        destination.rulesets[map.rulesets[0]].name.as_deref(),
        Some("child")
    );
    retain_rulesets(&mut destination, &[map.rulesets[0], map.rulesets[2]]).unwrap();
    assert_eq!(destination, previous);

    // Explicit retained-name references need no copied children or rules.
    let mut by_name = pb::Program {
        ir_version: 1,
        rulesets: vec![pb::Ruleset {
            name: Some("new".into()),
            kind: Some(pb::ruleset::Kind::Combined(pb::CombinedRuleset {
                rulesets: vec![pb::RulesetRef {
                    kind: Some(pb::ruleset_ref::Kind::Name("child".into())),
                }],
            })),
            ..Default::default()
        }],
        ..Default::default()
    };
    reconcile_rulesets(&by_name, &mut destination).unwrap();
    let previous = destination.clone();
    for name in ["missing", "new"] {
        let Some(pb::ruleset::Kind::Combined(c)) = &mut by_name.rulesets[0].kind else {
            unreachable!()
        };
        c.rulesets[0].kind = Some(pb::ruleset_ref::Kind::Name(name.into()));
        assert!(reconcile_rulesets(&by_name, &mut destination).is_err());
        assert_eq!(destination, previous);
    }
    let mut anonymous = occurrence_fixture(&[]);
    anonymous.rulesets.truncate(1);
    let first = reconcile_rulesets(&anonymous, &mut destination).unwrap();
    let second = reconcile_rulesets(&anonymous, &mut destination).unwrap();
    assert_ne!(first.rulesets, second.rulesets);
    assert_ne!(first.rules, second.rules);

    let mut two_names = occurrence_fixture(&[&[0], &[1]]);
    two_names.rulesets[0].name = Some("a".into());
    two_names.rulesets[1].name = Some("b".into());
    assert!(reconcile_rulesets(&two_names, &mut pb::Program::default()).is_err());

    let mut unanchored = source;
    unanchored.rulesets.truncate(1);
    unanchored.rulesets[0].name = Some("equal-but-new".into());
    let new = reconcile_rulesets(&unanchored, &mut destination).unwrap();
    assert_ne!(new.rulesets[0], retained_child);
    assert_ne!(new.rules[0], 0);
}

#[test]
fn ruleset_rule_bodies_compare_order_and_union_identity_after_relocation() {
    let mut source = occurrence_fixture(&[&[0]]);
    source.sorts.push(pb::Sort {
        kind: Some(pb::sort::Kind::Eq("Value".into())),
        ..Default::default()
    });
    source.nodes = (0..2)
        .map(|_| pb::Node {
            sort_id: 0,
            kind: Some(pb::node::Kind::Union(pb::Union::default())),
            ..Default::default()
        })
        .collect();
    source.files.push(pb::SourceFile {
        name: "rule.egg".into(),
        contents: Some("rule".into()),
    });
    source.rules[0].span = Some(pb::Span {
        file: 0,
        start: 0,
        end: 4,
    });
    let Some(pb::rule_decl::Kind::Rule(rule)) = &mut source.rules[0].kind else {
        unreachable!()
    };
    rule.head = vec![
        pb::Action {
            kind: Some(pb::action::Kind::Term(0)),
            ..Default::default()
        },
        pb::Action {
            kind: Some(pb::action::Kind::Delete(pb::Call {
                func: "F".into(),
                args: vec![0],
            })),
            ..Default::default()
        },
    ];
    let mut destination = pb::Program::default();
    let map = reconcile_rulesets(&source, &mut destination).unwrap();
    retain_rulesets(&mut destination, &[map.rulesets[2]]).unwrap();
    let previous = destination.clone();
    for change in 0..4 {
        let mut changed = source.clone();
        let Some(pb::rule_decl::Kind::Rule(rule)) = &mut changed.rules[0].kind else {
            unreachable!()
        };
        match change {
            0 => rule.head.reverse(),
            1 => {
                let Some(pb::action::Kind::Delete(call)) = &mut rule.head[1].kind else {
                    unreachable!()
                };
                call.args[0] = 1;
            }
            2 => {
                let Some(pb::action::Kind::Delete(call)) = &mut rule.head[1].kind else {
                    unreachable!()
                };
                call.func = "G".into();
            }
            _ => {
                changed.rules[0].kind = Some(pb::rule_decl::Kind::Rewrite(pb::Rewrite {
                    lhs: 0,
                    rhs: 0,
                    ..Default::default()
                }))
            }
        }
        assert!(reconcile_rulesets(&changed, &mut destination).is_err());
        assert_eq!(destination, previous);
    }
    // Ordinary syntax sharing may differ, but the single Union in both ordered
    // actions must stay one identity. Diagnostics and arena indices may change.
    let Some(pb::rule_decl::Kind::Rule(rule)) = &mut source.rules[0].kind else {
        unreachable!()
    };
    rule.head[0].kind = Some(pb::action::Kind::Term(1));
    let Some(pb::action::Kind::Delete(call)) = &mut rule.head[1].kind else {
        unreachable!()
    };
    call.args[0] = 1;
    source.files.insert(
        0,
        pb::SourceFile {
            name: "unused".into(),
            contents: None,
        },
    );
    source.rules[0].span.as_mut().unwrap().file = 1;
    source.rules[0].name = Some("new diagnostic".into());
    let map = reconcile_rulesets(&source, &mut destination).unwrap();
    retain_rulesets(&mut destination, &[map.rulesets[2]]).unwrap();
    assert_eq!(destination, previous);
    let mut reconstructed = pb::Program::default();
    reconcile_rulesets(&destination, &mut reconstructed).unwrap();
    assert_eq!(reconstructed, destination);
    reconcile_rulesets(&reconstructed, &mut destination).unwrap();
    assert_eq!(destination, previous);
}

#[test]
fn ruleset_forward_names_default_name_and_invalid_edges_are_distinct() {
    let mut forward = occurrence_fixture(&[&[0]]);
    forward.rulesets[0].name = Some(String::new());
    let Some(pb::ruleset::Kind::Combined(parent)) = &mut forward.rulesets[2].kind else {
        unreachable!()
    };
    parent.rulesets[0].kind = Some(pb::ruleset_ref::Kind::Name(String::new()));
    forward.rulesets.rotate_right(1); // parent refers forward to the default leaf
    let mut destination = pb::Program::default();
    let map = reconcile_rulesets(&forward, &mut destination).unwrap();
    assert_ne!(map.rulesets[1], map.rulesets[2]); // empty name is not absence
    let previous = destination.clone();
    for kind in [
        pb::ruleset_ref::Kind::Name("missing".into()),
        pb::ruleset_ref::Kind::Name("parent".into()),
        pb::ruleset_ref::Kind::Index(99),
    ] {
        let mut invalid = forward.clone();
        let Some(pb::ruleset::Kind::Combined(parent)) = &mut invalid.rulesets[0].kind else {
            unreachable!()
        };
        parent.rulesets[0].kind = Some(kind);
        assert!(reconcile_rulesets(&invalid, &mut destination).is_err());
        assert_eq!(destination, previous);
    }
}

#[test]
fn ruleset_and_existing_default_closures_relocate_and_retain_in_one_program() {
    let mut declarations = box_definition();
    declarations.files.push(pb::SourceFile {
        name: "defaults.egg".into(),
        contents: Some("default merge".into()),
    });
    let default_span = Some(pb::Span {
        file: 0,
        start: 0,
        end: 7,
    });
    for declaration in &mut declarations.declarations {
        declaration.span = default_span;
    }
    for sort in &mut declarations.sorts {
        sort.span = default_span;
    }
    declarations.nodes[0].span = default_span;
    declarations.nodes[2].span = Some(pb::Span {
        file: 0,
        start: 8,
        end: 13,
    });
    let mut destination = pb::Program::default();
    reconcile_declarations(&declarations, &mut destination).unwrap();
    let defaults = destination.clone();

    let mut source = occurrence_fixture(&[&[0, 1]]);
    source.files = vec![
        pb::SourceFile {
            name: "discarded.egg".into(),
            contents: Some("unused".into()),
        },
        pb::SourceFile {
            name: "rule.egg".into(),
            contents: Some("rule".into()),
        },
    ];
    let rule_span = Some(pb::Span {
        file: 1,
        start: 0,
        end: 4,
    });
    source.sorts = vec![
        pb::Sort {
            kind: Some(pb::sort::Kind::Eq("Unused".into())),
            ..Default::default()
        },
        pb::Sort {
            kind: Some(pb::sort::Kind::Eq("Box".into())),
            ..Default::default()
        },
        pb::Sort {
            kind: Some(pb::sort::Kind::Family(pb::HostSort {
                name: "i64".into(),
                args: vec![],
            })),
            ..Default::default()
        },
    ];
    source.nodes = vec![
        pb::Node {
            sort_id: 2,
            kind: Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                value: Some(pb::primitive_value::Value::I64(9)),
            })),
            span: Some(pb::Span {
                file: 0,
                start: 0,
                end: 6,
            }),
        },
        pb::Node {
            sort_id: 2,
            kind: Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                value: Some(pb::primitive_value::Value::I64(4)),
            })),
            span: rule_span,
        },
        pb::Node {
            sort_id: 1,
            kind: Some(pb::node::Kind::Call(pb::Call {
                func: "Box".into(),
                args: vec![1],
            })),
            span: rule_span,
        },
    ];
    source.rules[0].span = rule_span;
    source.rules[0].kind = Some(pb::rule_decl::Kind::Rule(pb::Rule {
        query: vec![],
        head: vec![pb::Action {
            kind: Some(pb::action::Kind::Term(2)),
            span: rule_span,
        }],
    }));
    for ruleset in &mut source.rulesets {
        ruleset.span = rule_span;
    }
    source.rules.push(pb::RuleDecl {
        eval_mode: pb::RuleEvalMode::Seminaive.into(),
        kind: Some(pb::rule_decl::Kind::Rule(pb::Rule {
            query: vec![],
            head: vec![pb::Action {
                kind: Some(pb::action::Kind::Term(0)),
                ..Default::default()
            }],
        })),
        ..Default::default()
    });
    source.rulesets.push(pb::Ruleset {
        kind: Some(pb::ruleset::Kind::Rules(pb::RuleList { rules: vec![1] })),
        ..Default::default()
    });
    let map = reconcile_rulesets(&source, &mut destination).unwrap();
    retain_rulesets(&mut destination, &[map.rulesets[2]]).unwrap();
    assert_eq!(destination.declarations, defaults.declarations);
    assert_eq!(&destination.nodes[..defaults.nodes.len()], defaults.nodes);
    assert_eq!(destination.sorts, defaults.sorts);
    assert_eq!(
        destination.files,
        [defaults.files[0].clone(), source.files[1].clone()]
    );
    assert_eq!(destination.nodes.len(), defaults.nodes.len() + 2);
    assert_eq!(destination.rules.len(), 1);
    assert_eq!(destination.rulesets.len(), 3); // distinct leaves, one shared rule
    let installed = destination.clone();

    // Resupply declarations/defaults and shared rules together, all three
    // auxiliary arenas reversed. No parallel/default-only retention path.
    let mut relocated = installed.clone();
    let last_node = relocated.nodes.len() as u32 - 1;
    relocated.sorts.reverse();
    relocated.nodes.reverse();
    relocated.files.reverse();
    for sort in &mut relocated.sorts {
        if let Some(span) = &mut sort.span {
            span.file = 1 - span.file;
        }
    }
    for node in &mut relocated.nodes {
        node.sort_id = 1 - node.sort_id;
        if let Some(span) = &mut node.span {
            span.file = 1 - span.file;
        }
        if let Some(pb::node::Kind::Call(call)) = &mut node.kind {
            for arg in &mut call.args {
                *arg = last_node - *arg;
            }
        }
    }
    for declaration in &mut relocated.declarations {
        if let Some(span) = &mut declaration.span {
            span.file = 1 - span.file;
        }
        match declaration.kind.as_mut().unwrap() {
            pb::declaration::Kind::Constructor(c) => {
                c.inputs[0].sort = 1 - c.inputs[0].sort;
                c.output = 1 - c.output;
            }
            pb::declaration::Kind::Function(f) => {
                f.output = 1 - f.output;
                f.merge = f.merge.map(|n| last_node - n);
            }
            pb::declaration::Kind::EqSort(_) => (),
            _ => unreachable!(),
        }
        if let Some(bindings) = &mut declaration.bindings {
            let python = &mut bindings.python.as_mut().unwrap().views[0];
            let Some(pb::binding_owner::Kind::Sort(sort)) =
                &mut python.owner.as_mut().unwrap().kind
            else {
                unreachable!()
            };
            *sort = 1 - *sort;
            python.params[0].default_expr = python.params[0].default_expr.map(|n| last_node - n);
            let rust = &mut bindings.rust.as_mut().unwrap().views[0];
            let Some(pb::binding_owner::Kind::Sort(sort)) = &mut rust.owner.as_mut().unwrap().kind
            else {
                unreachable!()
            };
            *sort = 1 - *sort;
        }
    }
    for rule in &mut relocated.rules {
        if let Some(span) = &mut rule.span {
            span.file = 1 - span.file;
        }
        let Some(pb::rule_decl::Kind::Rule(rule)) = &mut rule.kind else {
            unreachable!()
        };
        for action in &mut rule.head {
            if let Some(span) = &mut action.span {
                span.file = 1 - span.file;
            }
            let Some(pb::action::Kind::Term(node)) = &mut action.kind else {
                unreachable!()
            };
            *node = last_node - *node;
        }
    }
    for ruleset in &mut relocated.rulesets {
        if let Some(span) = &mut ruleset.span {
            span.file = 1 - span.file;
        }
    }
    assert_eq!(
        reconcile_declarations(&relocated, &mut destination).unwrap(),
        [false, false, false]
    );
    let map = reconcile_rulesets(&relocated, &mut destination).unwrap();
    retain_rulesets(&mut destination, &[map.rulesets[2]]).unwrap();
    assert_eq!(destination, installed);
    let mut reconstructed = pb::Program::default();
    reconcile_declarations(&relocated, &mut reconstructed).unwrap();
    let map = reconcile_rulesets(&relocated, &mut reconstructed).unwrap();
    retain_rulesets(&mut reconstructed, &[map.rulesets[2]]).unwrap();
    reconcile_declarations(&reconstructed, &mut destination).unwrap();
    let map = reconcile_rulesets(&reconstructed, &mut destination).unwrap();
    retain_rulesets(&mut destination, &[map.rulesets[2]]).unwrap();
    assert_eq!(destination, installed);
}

#[test]
fn scalar_views_relocate_and_freeze_exact_tail_presence() {
    let mut graph = egglog::EGraph::default();
    for key in [
        "string.concat",
        "string.replace",
        "bool.or",
        "bool.not",
        "i64.to_string",
    ] {
        let mut source = pb::Program::default();
        graph
            .type_info()
            .export_builtin_definition(&format!("egglog.core.{key}"), &mut source)
            .unwrap();
        let mut destination = pb::Program {
            ir_version: 1,
            sorts: vec![pb::Sort {
                kind: Some(pb::sort::Kind::Eq("Unrelated".into())),
                ..Default::default()
            }],
            ..Default::default()
        };
        reconcile_declarations(&source, &mut destination).unwrap();
        let installed = destination.clone();
        for _ in 0..3 {
            reconcile_declarations(&source, &mut destination).unwrap();
            reconcile_declarations(&installed, &mut destination).unwrap();
            assert_eq!(destination, installed);
        }
        let mut absent = source.clone();
        for d in &mut absent.declarations {
            d.bindings = None;
            if let Some(pb::declaration::Kind::HostSortFamily(f)) = &mut d.kind {
                f.bindings = None;
            }
        }
        reconcile_declarations(&absent, &mut destination).unwrap();
        assert_eq!(destination, installed);
        if key == "string.concat" {
            // Some(1) and None have the same mapped-slot shape but different
            // surface arities. First supply freezes each language's presence.
            for python in [false, true] {
                let mut changed = source.clone();
                let bindings = changed
                    .declarations
                    .iter_mut()
                    .find_map(|d| d.bindings.as_mut())
                    .unwrap();
                if python {
                    bindings.python.as_mut().unwrap().views[0].tail_repetitions = Some(1);
                } else {
                    bindings.rust.as_mut().unwrap().views[0].tail_repetitions = Some(1);
                }
                reconcile_declarations(&changed, &mut pb::Program::default()).unwrap();
                assert!(reconcile_declarations(&changed, &mut destination).is_err());
                assert_eq!(destination, installed);
            }
        }
    }
}

#[test]
fn fixed_tail_views_accept_zero_and_swapped_receiver_slots() {
    let mut graph = egglog::EGraph::default();
    let mut source = pb::Program::default();
    graph
        .type_info()
        .export_builtin_definition("egglog.core.string.concat", &mut source)
        .unwrap();
    for count in [0, 1, 2, 5] {
        let mut changed = source.clone();
        let bindings = changed
            .declarations
            .iter_mut()
            .find_map(|d| d.bindings.as_mut())
            .unwrap();
        let py = bindings.python.as_mut().unwrap();
        py.views.truncate(1);
        py.views[0].tail_repetitions = Some(count);
        py.views[0].params = (0..count)
            .map(|i| pb::PythonParameter {
                core_input: Some(i),
                name: format!("arg{i}"),
                default_expr: None,
            })
            .collect();
        let rs = bindings.rust.as_mut().unwrap();
        rs.views.truncate(1);
        rs.views[0].tail_repetitions = Some(count);
        rs.views[0].params = (0..count)
            .map(|i| pb::RustParameter {
                core_input: Some(i),
                name: format!("arg{i}"),
                borrowed: false,
            })
            .collect();
        let mut installed = pb::Program::default();
        reconcile_declarations(&changed, &mut installed).unwrap();
        let bindings = installed
            .declarations
            .iter()
            .find_map(|d| d.bindings.as_ref())
            .unwrap();
        assert_eq!(
            bindings.python.as_ref().unwrap().views[0].tail_repetitions,
            Some(count)
        );
        assert_eq!(
            bindings.rust.as_ref().unwrap().views[0].tail_repetitions,
            Some(count)
        );
    }
    let b = source
        .declarations
        .iter_mut()
        .find_map(|d| d.bindings.as_mut())
        .unwrap();
    let py = &mut b.python.as_mut().unwrap().views[1];
    py.receiver = Some(1);
    py.params[0].core_input = Some(0);
    let rs = &mut b.rust.as_mut().unwrap().views[1];
    rs.receiver.as_mut().unwrap().core_input = Some(1);
    rs.params[0].core_input = Some(0);
    reconcile_declarations(&source, &mut pb::Program::default()).unwrap();
}

#[test]
fn fixed_tail_python_forms_reject_first_supply_and_absent_block_enrichment() {
    use pb::PythonCallKind as K;
    let mut graph = egglog::EGraph::default();
    let mut source = pb::Program::default();
    graph
        .type_info()
        .export_builtin_definition("egglog.core.bool.or", &mut source)
        .unwrap();
    let index = source
        .declarations
        .iter()
        .position(|d| matches!(&d.kind, Some(pb::declaration::Kind::HostPrimitive(p)) if p.name == "egglog.core.bool.or"))
        .unwrap();
    let owner = source.declarations[index]
        .bindings
        .as_ref()
        .unwrap()
        .python
        .as_ref()
        .unwrap()
        .views[0]
        .owner;
    let mut absent = source.clone();
    absent.declarations[index].bindings.as_mut().unwrap().python = None;
    let mut accepted = vec![];
    for kind in [
        K::Constant,
        K::Property,
        K::Initializer,
        K::ClassMethod,
        K::ClassVariable,
    ] {
        let mut changed = source.clone();
        changed.declarations[index]
            .bindings
            .as_mut()
            .unwrap()
            .python = Some(pb::PythonBindings {
            views: vec![pb::PythonCallable {
                kind: kind.into(),
                path: match kind {
                    K::Constant => vec!["test".into(), "C".into()],
                    K::Initializer => vec![],
                    _ => vec!["value".into()],
                },
                owner: if kind == K::Constant { None } else { owner },
                receiver: (kind == K::Property).then_some(0),
                tail_repetitions: Some(u32::from(kind == K::Property)),
                ..Default::default()
            }],
        });
        for enrich in [false, true] {
            let mut destination = pb::Program::default();
            if enrich {
                reconcile_declarations(&absent, &mut destination).unwrap();
            }
            let before = destination.clone();
            let rejected = reconcile_declarations(&changed, &mut destination).is_err();
            if !rejected || destination != before {
                accepted.push((kind, enrich));
            }
        }
        // A preexisting language block already rejects a changed view on
        // conflict; this alone cannot test first-supply form validation.
        let mut frozen = pb::Program::default();
        reconcile_declarations(&source, &mut frozen).unwrap();
        let before = frozen.clone();
        assert!(reconcile_declarations(&changed, &mut frozen).is_err());
        assert_eq!(frozen, before);
    }
    assert!(
        accepted.is_empty(),
        "unsupported (kind, enrichment) accepted: {accepted:?}"
    );
}

#[test]
fn fixed_tail_views_derive_prefix_and_pattern_types_and_keep_fixed_defaults() {
    let mut graph = egglog::EGraph::default();
    let mut source = pb::Program::default();
    graph
        .type_info()
        .export_builtin_definition("egglog.core.bool.or", &mut source)
        .unwrap();
    graph
        .type_info()
        .export_builtin_definition("egglog.core.i64.add", &mut source)
        .unwrap();
    let integer = source
        .sorts
        .iter()
        .position(|s| {
            matches!(&s.kind,
        Some(pb::sort::Kind::Family(f)) if f.name == "i64")
        })
        .unwrap() as u32;
    let boolean = source
        .sorts
        .iter()
        .position(|s| {
            matches!(&s.kind,
        Some(pb::sort::Kind::Family(f)) if f.name == "bool")
        })
        .unwrap() as u32;
    source.nodes.push(pb::Node {
        sort_id: boolean,
        kind: Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
            value: Some(pb::primitive_value::Value::Bool(false)),
        })),
        ..Default::default()
    });
    let index = source
        .declarations
        .iter()
        .position(|d| {
            matches!(&d.kind,
        Some(pb::declaration::Kind::HostPrimitive(p)) if p.name == "egglog.core.bool.or")
        })
        .unwrap();
    let declaration = &mut source.declarations[index];
    let Some(pb::declaration::Kind::HostPrimitive(p)) = &mut declaration.kind else {
        unreachable!()
    };
    let Some(pb::host_primitive::Typing::Signature(s)) = &mut p.typing else {
        unreachable!()
    };
    // Shape-only mutation of the exported record: bool, (i64, bool)* -> bool.
    // R=2 makes slots [bool, i64, bool, i64, bool], with receiver slot 3.
    s.inputs = vec![pb::Arg {
        name: "fixed".into(),
        sort: boolean,
    }];
    s.varargs = vec![
        pb::Arg {
            name: "key".into(),
            sort: integer,
        },
        pb::Arg {
            name: "value".into(),
            sort: boolean,
        },
    ];
    let b = declaration.bindings.as_mut().unwrap();
    let py = &mut b.python.as_mut().unwrap().views[0];
    py.owner.as_mut().unwrap().kind = Some(pb::binding_owner::Kind::Sort(integer));
    py.receiver = Some(3);
    py.params = [1, 2, 4, 0]
        .into_iter()
        .map(|i| pb::PythonParameter {
            core_input: Some(i),
            name: format!("arg{i}"),
            default_expr: (i == 0).then_some(0),
        })
        .collect();
    py.mutates = Some(0);
    let rs = &mut b.rust.as_mut().unwrap().views[0];
    rs.owner = py.owner;
    rs.receiver.as_mut().unwrap().core_input = Some(3);
    rs.params = [0, 1, 2, 4]
        .into_iter()
        .map(|i| pb::RustParameter {
            core_input: Some(i),
            name: format!("arg{i}"),
            borrowed: false,
        })
        .collect();
    rs.trait_impl = None;
    let mut destination = pb::Program::default();
    reconcile_declarations(&source, &mut destination).unwrap();
    let installed = destination.clone();
    reconcile_declarations(&source, &mut destination).unwrap();
    assert_eq!(destination, installed);
    for invalid in 0..4 {
        let mut changed = source.clone();
        let b = changed.declarations[index].bindings.as_mut().unwrap();
        match invalid {
            0 => {
                b.python.as_mut().unwrap().views[0]
                    .owner
                    .as_mut()
                    .unwrap()
                    .kind = Some(pb::binding_owner::Kind::Sort(boolean))
            }
            1 => {
                b.rust.as_mut().unwrap().views[0]
                    .owner
                    .as_mut()
                    .unwrap()
                    .kind = Some(pb::binding_owner::Kind::Sort(boolean))
            }
            2 => b.python.as_mut().unwrap().views[0].mutates = Some(2),
            _ => b.python.as_mut().unwrap().views[0].params[2].default_expr = Some(0),
        }
        assert!(
            reconcile_declarations(&changed, &mut pb::Program::default()).is_err(),
            "case={invalid}"
        );
    }
}

#[test]
fn fixed_tail_views_reject_invalid_counts_slots_types_and_defaults_transactionally() {
    let mut graph = egglog::EGraph::default();
    let mut source = pb::Program::default();
    graph
        .type_info()
        .export_builtin_definition("egglog.core.bool.or", &mut source)
        .unwrap();
    // A real second scalar family supplies the wrong-sort owner control.
    graph
        .type_info()
        .export_builtin_definition("egglog.core.i64.add", &mut source)
        .unwrap();
    let integer = source
        .sorts
        .iter()
        .position(|s| {
            matches!(&s.kind,
        Some(pb::sort::Kind::Family(f)) if f.name == "i64")
        })
        .unwrap() as u32;
    let boolean = source
        .sorts
        .iter()
        .position(|s| {
            matches!(&s.kind,
        Some(pb::sort::Kind::Family(f)) if f.name == "bool")
        })
        .unwrap() as u32;
    source.nodes.push(pb::Node {
        sort_id: boolean,
        kind: Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
            value: Some(pb::primitive_value::Value::Bool(false)),
        })),
        ..Default::default()
    });
    for python in [false, true] {
        for invalid in 0..10 {
            let mut changed = source.clone();
            let d = changed
                .declarations
                .iter_mut()
                .find(|d| {
                    matches!(&d.kind,
                Some(pb::declaration::Kind::HostPrimitive(p)) if p.name == "egglog.core.bool.or")
                })
                .unwrap();
            let b = d.bindings.as_mut().unwrap();
            if python {
                let v = &mut b.python.as_mut().unwrap().views[0];
                match invalid {
                    0 => v.tail_repetitions = None,
                    1 => v.tail_repetitions = Some(0),
                    2 => v.tail_repetitions = Some(u32::MAX),
                    3 => v.params[0].core_input = Some(0),
                    4 => v.params.clear(),
                    5 => v.params[0].core_input = None,
                    6 => {
                        v.owner.as_mut().unwrap().kind =
                            Some(pb::binding_owner::Kind::Sort(integer))
                    }
                    7 => v.params[0].default_expr = Some(0),
                    8 => v.mutates = Some(0),
                    _ => v.receiver = Some(2),
                }
            } else {
                let v = &mut b.rust.as_mut().unwrap().views[0];
                match invalid {
                    0 => v.tail_repetitions = None,
                    1 => v.tail_repetitions = Some(0),
                    2 => v.tail_repetitions = Some(u32::MAX),
                    3 => v.params[0].core_input = Some(0),
                    4 => v.params.clear(),
                    5 => v.params[0].core_input = None,
                    6 => {
                        v.owner.as_mut().unwrap().kind =
                            Some(pb::binding_owner::Kind::Sort(integer))
                    }
                    7 => v.receiver.as_mut().unwrap().core_input = None,
                    8 => v.params.push(v.params[0].clone()),
                    _ => v.receiver.as_mut().unwrap().core_input = Some(2),
                }
            }
            let mut destination = source.clone();
            let before = destination.clone();
            assert!(
                reconcile_declarations(&changed, &mut destination).is_err(),
                "python={python}, case={invalid}"
            );
            assert_eq!(destination, before);
            let mut empty = pb::Program::default();
            if invalid == 0 {
                // Without the count, receiver0 consumes the first homogeneous
                // tail operand and parameter1 maps its remainder. This is a
                // valid first supply, but changes the frozen binding above.
                reconcile_declarations(&changed, &mut empty).unwrap();
                let installed = empty.clone();
                assert!(
                    reconcile_declarations(&changed, &mut empty)
                        .unwrap()
                        .iter()
                        .all(|changed| !changed)
                );
                assert_eq!(empty, installed);
                assert!(reconcile_declarations(&source, &mut empty).is_err());
                assert_eq!(empty, installed);
            } else {
                assert!(
                    reconcile_declarations(&changed, &mut empty).is_err(),
                    "first supply: python={python}, case={invalid}"
                );
                assert_eq!(empty, pb::Program::default());
            }
        }
    }
    for invalid in 0..3 {
        let mut changed = source.clone();
        let d = changed
            .declarations
            .iter_mut()
            .find(|d| {
                matches!(&d.kind,
            Some(pb::declaration::Kind::HostPrimitive(p)) if p.name == "egglog.core.bool.or")
            })
            .unwrap();
        let Some(pb::declaration::Kind::HostPrimitive(p)) = &mut d.kind else {
            unreachable!()
        };
        if invalid == 2 {
            p.typing = Some(pb::host_primitive::Typing::Application(
                pb::FunctionApplication {},
            ));
        } else {
            let Some(pb::host_primitive::Typing::Signature(s)) = &mut p.typing else {
                unreachable!()
            };
            if invalid == 0 {
                // Keep two fixed slots to isolate the empty-tail guard.
                s.inputs = vec![s.varargs[0].clone(); 2];
                s.varargs.clear();
            } else {
                s.varargs.push(s.varargs[0].clone());
                let b = d.bindings.as_mut().unwrap();
                b.python.as_mut().unwrap().views[0].tail_repetitions = Some(u32::MAX);
                b.rust.as_mut().unwrap().views[0].tail_repetitions = Some(u32::MAX);
            }
        }
        assert!(
            reconcile_declarations(&changed, &mut pb::Program::default()).is_err(),
            "signature case={invalid}"
        );
    }
}

#[test]
fn container_language_views_relocate_enrich_and_freeze_without_growth() {
    let mut graph = egglog::EGraph::default();
    for key in [
        "pair.make",
        "pair.first",
        "pair.second",
        "vec.empty",
        "vec.of",
        "vec.get",
        "map.empty",
        "map.of",
        "map.get",
        "map.insert",
    ] {
        let mut source = pb::Program::default();
        graph
            .type_info()
            .export_builtin_definition(&format!("egglog.core.{key}"), &mut source)
            .unwrap();
        let mut absent = source.clone();
        for d in &mut absent.declarations {
            if let Some(pb::declaration::Kind::HostSortFamily(f)) = &mut d.kind {
                f.bindings = None;
            }
            if let Some(b) = &mut d.bindings {
                b.python = None;
                b.rust = None;
            }
        }
        let mut destination = pb::Program::default();
        assert_eq!(
            reconcile_declarations(&absent, &mut destination).unwrap(),
            vec![true; source.declarations.len()]
        );
        for python in [true, false] {
            let mut language = source.clone();
            for d in &mut language.declarations {
                if let Some(pb::declaration::Kind::HostSortFamily(f)) = &mut d.kind {
                    let b = f.bindings.as_mut().expect("family language views");
                    if python {
                        b.rust = None;
                    } else {
                        b.python = None;
                    }
                }
                if let Some(b) = &mut d.bindings {
                    assert!(b.python.is_some() && b.rust.is_some());
                    if python {
                        b.rust = None;
                    } else {
                        b.python = None;
                    }
                }
            }
            assert_eq!(
                reconcile_declarations(&language, &mut destination).unwrap(),
                vec![false; source.declarations.len()]
            );
        }
        let installed = destination.clone();
        let mut permuted = source.clone();
        let remap = |i: u32| source.sorts.len() as u32 - 1 - i;
        permuted.sorts.reverse();
        for sort in &mut permuted.sorts {
            if let Some(pb::sort::Kind::Family(f)) = &mut sort.kind {
                for arg in &mut f.args {
                    *arg = remap(*arg);
                }
            }
        }
        for d in &mut permuted.declarations {
            if let Some(pb::declaration::Kind::HostPrimitive(p)) = &mut d.kind {
                let Some(pb::host_primitive::Typing::Signature(s)) = &mut p.typing else {
                    unreachable!()
                };
                for arg in s.inputs.iter_mut().chain(&mut s.varargs) {
                    arg.sort = remap(arg.sort);
                    arg.name = format!("diagnostic_{}", arg.name);
                }
                s.output = s.output.map(remap);
            }
            if let Some(b) = &mut d.bindings {
                for owner in b
                    .python
                    .as_mut()
                    .unwrap()
                    .views
                    .iter_mut()
                    .map(|v| v.owner.as_mut().unwrap())
                    .chain(
                        b.rust
                            .as_mut()
                            .unwrap()
                            .views
                            .iter_mut()
                            .map(|v| v.owner.as_mut().unwrap()),
                    )
                {
                    let Some(pb::binding_owner::Kind::Sort(i)) = &mut owner.kind else {
                        unreachable!()
                    };
                    *i = remap(*i);
                }
            }
        }
        permuted.declarations.reverse();
        for _ in 0..3 {
            assert_eq!(
                reconcile_declarations(&permuted, &mut destination).unwrap(),
                vec![false; source.declarations.len()]
            );
            reconcile_declarations(&absent, &mut destination).unwrap();
            assert_eq!(
                destination, installed,
                "metadata relocation/resupply for {key}"
            );
        }
        for conflict in 0..4 {
            let mut changed = source.clone();
            for d in &mut changed.declarations {
                if conflict < 2 {
                    if let Some(pb::declaration::Kind::HostSortFamily(f)) = &mut d.kind {
                        let b = f.bindings.as_mut().unwrap();
                        let language = if conflict == 0 {
                            &mut b.python
                        } else {
                            &mut b.rust
                        };
                        let language = language.as_mut().unwrap();
                        if let Some(parameter) = language.type_params.first_mut() {
                            *parameter = "Renamed".into();
                        } else {
                            language.path[0] = "renamed".into();
                        }
                    }
                } else if let Some(b) = &mut d.bindings {
                    if conflict == 2 {
                        let v = &mut b.python.as_mut().unwrap().views[0];
                        if v.path.is_empty() {
                            v.params[0].name = "renamed".into();
                        } else {
                            v.path[0] = "renamed".into();
                        }
                    } else {
                        b.rust.as_mut().unwrap().views[0].path[0] = "renamed".into();
                    }
                }
            }
            assert!(
                reconcile_declarations(&changed, &mut destination).is_err(),
                "{key}, conflict {conflict}"
            );
            assert_eq!(destination, installed);
        }
    }
}

#[test]
fn pair_language_views_reject_invalid_owners_and_receivers_transactionally() {
    let mut graph = egglog::EGraph::default();
    for key in ["make", "first", "second"] {
        let mut source = pb::Program::default();
        graph
            .type_info()
            .export_builtin_definition(&format!("egglog.core.pair.{key}"), &mut source)
            .unwrap();
        let variable = source
            .sorts
            .iter()
            .position(|s| s.kind == Some(pb::sort::Kind::Var(0)))
            .unwrap() as u32;
        for invalid in 0..6 {
            let mut changed = source.clone();
            let unbound = changed.sorts.len() as u32;
            changed.sorts.push(pb::Sort {
                kind: Some(pb::sort::Kind::Var(2)),
                ..Default::default()
            });
            let b = changed
                .declarations
                .iter_mut()
                .find_map(|d| d.bindings.as_mut())
                .unwrap();
            match invalid {
                0 => {
                    b.python.as_mut().unwrap().views[0]
                        .owner
                        .as_mut()
                        .unwrap()
                        .kind = Some(pb::binding_owner::Kind::Sort(variable))
                }
                1 => {
                    b.rust.as_mut().unwrap().views[0]
                        .owner
                        .as_mut()
                        .unwrap()
                        .kind = Some(pb::binding_owner::Kind::Sort(unbound))
                }
                2 => b.python.as_mut().unwrap().views[0].receiver = Some(1),
                3 => {
                    b.rust.as_mut().unwrap().views[0].receiver = Some(pb::RustReceiver {
                        core_input: Some(1),
                        borrowed: true,
                    })
                }
                4 => {
                    b.python.as_mut().unwrap().views[0]
                        .owner
                        .as_mut()
                        .unwrap()
                        .kind = Some(pb::binding_owner::Kind::Sort(u32::MAX))
                }
                _ => {
                    b.rust.as_mut().unwrap().views[0]
                        .owner
                        .as_mut()
                        .unwrap()
                        .kind = Some(pb::binding_owner::Kind::Sort(u32::MAX))
                }
            }
            let mut destination = graph.type_info().builtin_catalog().unwrap().definitions;
            let before = destination.clone();
            assert!(
                reconcile_declarations(&changed, &mut destination).is_err(),
                "{key}, invalid view {invalid}"
            );
            assert_eq!(destination, before);
            let mut empty = pb::Program::default();
            assert!(
                reconcile_declarations(&changed, &mut empty).is_err(),
                "{key}, invalid first supply {invalid}"
            );
            assert_eq!(empty, pb::Program::default());
        }
    }
}

#[test]
fn declaration_namespaces_coexist_with_defaults_and_frozen_resupply() {
    let mut source = pair_default_definition();
    source.sorts[1].kind = Some(pb::sort::Kind::Eq("Pair".into()));
    let Some(pb::declaration::Kind::EqSort(eq)) = &mut source.declarations[0].kind else {
        unreachable!()
    };
    eq.name = "Pair".into();
    let Some(pb::declaration::Kind::Constructor(c)) = &mut source.declarations[1].kind else {
        unreachable!()
    };
    c.name = "Pair".into();
    let Some(pb::declaration::Kind::HostSortFamily(f)) = &mut source.declarations[3].kind else {
        unreachable!()
    };
    f.bindings = Some(pb::SortBindings {
        python: Some(pb::TypeBinding {
            path: vec!["test".into(), "HostPair".into()],
            type_params: vec!["A".into(), "B".into()],
        }),
        ..Default::default()
    });
    source.nodes.push(pb::Node {
        sort_id: 1,
        kind: Some(pb::node::Kind::Call(pb::Call {
            func: "Pair".into(),
            args: vec![2],
        })),
        ..Default::default()
    });
    source.declarations.push(pb::Declaration {
        kind: Some(pb::declaration::Kind::Function(pb::Function {
            name: "use_pair".into(),
            inputs: vec![pb::Arg {
                sort: 1,
                name: "value".into(),
            }],
            output: 0,
            ..Default::default()
        })),
        bindings: Some(pb::CallableBindings {
            python: Some(pb::PythonBindings {
                views: vec![pb::PythonCallable {
                    kind: pb::PythonCallKind::Function.into(),
                    path: vec!["test".into(), "use_pair".into()],
                    params: vec![pb::PythonParameter {
                        core_input: Some(0),
                        name: "value".into(),
                        default_expr: Some(3),
                    }],
                    ..Default::default()
                }],
            }),
            ..Default::default()
        }),
        ..Default::default()
    });
    for reverse in [false, true] {
        for incremental in [false, true] {
            let mut incoming = source.clone();
            if reverse {
                incoming.declarations.reverse();
            }
            let mut destination = pb::Program::default();
            if incremental {
                // Install the two sort namespaces in both arrival orders before
                // supplying callable/default closures. Neither shadows the other.
                let sorts = pb::Program {
                    ir_version: 1,
                    declarations: incoming
                        .declarations
                        .iter()
                        .filter(|d| {
                            matches!(
                                d.kind,
                                Some(
                                    pb::declaration::Kind::EqSort(_)
                                        | pb::declaration::Kind::HostSortFamily(_)
                                )
                            )
                        })
                        .cloned()
                        .collect(),
                    ..Default::default()
                };
                for d in sorts.declarations {
                    reconcile_declarations(
                        &pb::Program {
                            ir_version: 1,
                            declarations: vec![d],
                            ..Default::default()
                        },
                        &mut destination,
                    )
                    .unwrap();
                }
            }
            reconcile_declarations(&incoming, &mut destination).unwrap();
            assert_eq!(destination.declarations.len(), 5);
            assert!(
                destination
                    .sorts
                    .iter()
                    .any(|s| s.kind == Some(pb::sort::Kind::Eq("Pair".into())))
            );
            assert!(destination.sorts.iter().any(|s| matches!(&s.kind,
                Some(pb::sort::Kind::Family(f)) if f.name == "Pair" && f.args.len() == 2)));
            let frozen = destination.clone();
            incoming.declarations.reverse();
            assert_eq!(
                reconcile_declarations(&incoming, &mut destination).unwrap(),
                [false; 5]
            );
            assert_eq!(
                destination, frozen,
                "compatible resupply retains exact closures"
            );
            for bad in 0..5 {
                let mut conflict = source.clone();
                match bad {
                    0 => {
                        let Some(pb::declaration::Kind::EqSort(eq)) =
                            &mut conflict.declarations[0].kind
                        else {
                            unreachable!()
                        };
                        eq.bindings.as_mut().unwrap().python.as_mut().unwrap().path[1] =
                            "Changed".into();
                    }
                    1 | 2 => {
                        let Some(pb::declaration::Kind::HostSortFamily(f)) =
                            &mut conflict.declarations[3].kind
                        else {
                            unreachable!()
                        };
                        if bad == 1 {
                            f.arity = 1;
                        } else {
                            f.bindings.as_mut().unwrap().python.as_mut().unwrap().path[1] =
                                "Changed".into();
                        }
                    }
                    3 => {
                        let Some(pb::declaration::Kind::Constructor(c)) =
                            &mut conflict.declarations[1].kind
                        else {
                            unreachable!()
                        };
                        c.unextractable = true;
                    }
                    _ => {
                        conflict.declarations[1]
                            .bindings
                            .as_mut()
                            .unwrap()
                            .python
                            .as_mut()
                            .unwrap()
                            .views[0]
                            .params[0]
                            .name = "Changed".into()
                    }
                }
                assert!(
                    reconcile_declarations(&conflict, &mut destination).is_err(),
                    "conflict {bad}"
                );
                assert_eq!(destination, frozen, "conflict {bad} must be transactional");
            }
        }
    }
    // Neither same-named sort declaration can satisfy a default callee lookup.
    let mut no_callable = source.clone();
    no_callable.declarations.remove(1);
    let mut destination = pb::Program::default();
    assert_eq!(
        reconcile_declarations(&no_callable, &mut destination).unwrap_err(),
        "unknown default callee"
    );
    assert_eq!(destination, pb::Program::default());

    // A nominal declaration cannot satisfy or mask the family's arity check.
    let Some(pb::sort::Kind::Family(f)) = &mut source.sorts[3].kind else {
        unreachable!()
    };
    f.args.pop();
    let mut destination = pb::Program::default();
    assert!(reconcile_declarations(&source, &mut destination).is_err());
    assert_eq!(destination, pb::Program::default());
}

#[test]
fn documentation_presence_is_retained_but_not_semantic_identity() {
    for first in [None, Some(""), Some(" docs\n")] {
        let mut source = box_definition();
        for d in &mut source.declarations {
            d.doc = first.map(str::to_owned);
        }
        let mut destination = pb::Program::default();
        reconcile_declarations(&source, &mut destination).unwrap();
        let installed = destination.clone();
        assert!(
            destination
                .declarations
                .iter()
                .all(|d| d.doc.as_deref() == first)
        );
        for later in [None, Some(""), Some("other")] {
            let mut resupply = source.clone();
            for d in &mut resupply.declarations {
                d.doc = later.map(str::to_owned);
            }
            let mut independent = destination.clone();
            reconcile_declarations(&resupply, &mut independent).unwrap();
            assert_eq!(
                independent, installed,
                "resupply retains first-record diagnostics"
            );
            assert_eq!(destination, installed);
            let Some(pb::declaration::Kind::Constructor(c)) = &mut resupply.declarations[1].kind
            else {
                unreachable!()
            };
            c.unextractable = true;
            assert!(reconcile_declarations(&resupply, &mut independent).is_err());
            assert_eq!(independent, installed);
            let Some(pb::declaration::Kind::Constructor(c)) = &mut resupply.declarations[1].kind
            else {
                unreachable!()
            };
            c.unextractable = false;
            resupply.declarations[1]
                .bindings
                .as_mut()
                .unwrap()
                .python
                .as_mut()
                .unwrap()
                .views[0]
                .params[0]
                .name = "conflicting".into();
            assert!(reconcile_declarations(&resupply, &mut independent).is_err());
            assert_eq!(independent, installed);
        }
    }
}

fn constant_definition() -> pb::Program {
    let mut p = box_definition();
    p.nodes.clear();
    let Some(pb::declaration::Kind::Constructor(c)) = &mut p.declarations[1].kind else {
        unreachable!()
    };
    c.inputs.clear();
    let Some(pb::declaration::Kind::Function(f)) = &mut p.declarations[2].kind else {
        unreachable!()
    };
    f.merge = None;
    p.declarations.push(pb::Declaration {
        kind: Some(pb::declaration::Kind::HostPrimitive(pb::HostPrimitive {
            name: "test.constant".into(),
            typing: Some(pb::host_primitive::Typing::Signature(
                pb::GenericSignature {
                    output: Some(0),
                    ..Default::default()
                },
            )),
        })),
        ..Default::default()
    });
    for (d, name) in p.declarations[1..]
        .iter_mut()
        .zip(["BOX", "CURRENT", "HOST"])
    {
        d.bindings = Some(pb::CallableBindings {
            python: Some(pb::PythonBindings {
                views: vec![pb::PythonCallable {
                    kind: pb::PythonCallKind::Constant.into(),
                    path: vec!["test".into(), name.into()],
                    ..Default::default()
                }],
            }),
            ..Default::default()
        });
    }
    p
}

#[test]
fn constant_views_install_resupply_and_freeze_without_evaluation() {
    let source = constant_definition();
    let mut absent = source.clone();
    for d in &mut absent.declarations {
        d.bindings = None;
    }
    let mut destination = pb::Program::default();
    reconcile_declarations(&absent, &mut destination).unwrap();
    assert_eq!(
        reconcile_declarations(&source, &mut destination).unwrap(),
        [false; 4]
    );
    let frozen = destination.clone();
    for _ in 0..3 {
        reconcile_declarations(&source, &mut destination).unwrap();
        reconcile_declarations(&absent, &mut destination).unwrap();
        assert_eq!(destination, frozen);
    }
    let mut function = source;
    function.declarations[2]
        .bindings
        .as_mut()
        .unwrap()
        .python
        .as_mut()
        .unwrap()
        .views[0]
        .kind = pb::PythonCallKind::Function.into();
    let mut independent = pb::Program::default();
    reconcile_declarations(&function, &mut independent).unwrap();
    assert_ne!(independent, frozen, "nullary FUNCTION is not CONSTANT");
    assert!(reconcile_declarations(&function, &mut destination).is_err());
    assert_eq!(
        destination, frozen,
        "first supplied Python metadata is frozen"
    );
}

#[test]
fn empty_constant_names_preserve_exact_metadata_and_freeze_transactionally() {
    for path in [vec!["".to_string()], vec!["example".into(), "".into()]] {
        let mut source = constant_definition();
        source.declarations[2]
            .bindings
            .as_mut()
            .unwrap()
            .python
            .as_mut()
            .unwrap()
            .views[0]
            .path = path;
        let mut absent = source.clone();
        absent.declarations[2].bindings = None;
        let mut destination = pb::Program::default();
        reconcile_declarations(&absent, &mut destination).unwrap();
        let mut independent = destination.clone();
        reconcile_declarations(&source, &mut destination).unwrap();
        assert_eq!(
            destination.declarations[2].bindings,
            source.declarations[2].bindings
        );
        let frozen = destination.clone();
        reconcile_declarations(&source, &mut destination).unwrap();
        reconcile_declarations(&absent, &mut destination).unwrap();
        assert_eq!(destination, frozen);

        let mut renamed = source.clone();
        renamed.declarations[2]
            .bindings
            .as_mut()
            .unwrap()
            .python
            .as_mut()
            .unwrap()
            .views[0]
            .path
            .last_mut()
            .unwrap()
            .push_str("named");
        assert!(reconcile_declarations(&renamed, &mut destination).is_err());
        assert_eq!(
            destination, frozen,
            "empty and nonempty names must not coalesce"
        );
        reconcile_declarations(&renamed, &mut independent).unwrap();
        let independent_frozen = independent.clone();
        assert!(reconcile_declarations(&source, &mut independent).is_err());
        assert_eq!(
            independent, independent_frozen,
            "clone freezes its own first supply"
        );

        let mut function = source.clone();
        function.declarations[2]
            .bindings
            .as_mut()
            .unwrap()
            .python
            .as_mut()
            .unwrap()
            .views[0]
            .kind = pb::PythonCallKind::Function.into();
        let mut fresh = pb::Program::default();
        assert!(reconcile_declarations(&function, &mut fresh).is_err());
        assert_eq!(
            fresh,
            pb::Program::default(),
            "FUNCTION does not admit an empty final name"
        );
        for bad in [
            vec![],
            vec!["", ""],
            vec!["", "value"],
            vec!["example", "", "value"],
            vec!["example", "", ""],
        ] {
            let mut invalid = source.clone();
            invalid.declarations[2]
                .bindings
                .as_mut()
                .unwrap()
                .python
                .as_mut()
                .unwrap()
                .views[0]
                .path = bad.into_iter().map(str::to_owned).collect();
            assert!(reconcile_declarations(&invalid, &mut fresh).is_err());
            assert_eq!(fresh, pb::Program::default());
            assert!(reconcile_declarations(&invalid, &mut destination).is_err());
            assert_eq!(destination, frozen);
        }
        let Some(pb::declaration::Kind::Function(f)) = &mut source.declarations[2].kind else {
            unreachable!()
        };
        f.name.clear();
        assert!(reconcile_declarations(&source, &mut fresh).is_err());
        assert_eq!(
            fresh,
            pb::Program::default(),
            "semantic names remain nonempty"
        );
    }
}

#[test]
fn constant_views_reject_malformed_forms_and_signatures_transactionally() {
    let source = constant_definition();
    let mut destination = pb::Program::default();
    reconcile_declarations(&source, &mut destination).unwrap();
    for bad in 0..15 {
        let mut candidate = source.clone();
        let view = &mut candidate.declarations[3]
            .bindings
            .as_mut()
            .unwrap()
            .python
            .as_mut()
            .unwrap()
            .views[0];
        match bad {
            0 => view.path.clear(),
            1 => view.path.insert(0, String::new()),
            2 => {
                view.owner = Some(pb::BindingOwner {
                    kind: Some(pb::binding_owner::Kind::Sort(0)),
                })
            }
            3 => view.receiver = Some(0),
            4 => view.params.push(pb::PythonParameter {
                core_input: Some(0),
                name: "x".into(),
                ..Default::default()
            }),
            5 => view.mutates = Some(0),
            6 => view.kind = 99,
            7..=10 => {
                let Some(pb::declaration::Kind::HostPrimitive(h)) =
                    &mut candidate.declarations[3].kind
                else {
                    unreachable!()
                };
                let Some(pb::host_primitive::Typing::Signature(s)) = &mut h.typing else {
                    unreachable!()
                };
                match bad {
                    7 => s.inputs.push(pb::Arg {
                        name: "x".into(),
                        sort: 0,
                    }),
                    8 => {
                        s.varargs = vec![pb::Arg {
                            name: "xs".into(),
                            sort: 0,
                        }]
                    }
                    9 => s.type_params.push("Undetermined".into()),
                    _ => s.output = Some(99),
                }
            }
            11 => {
                let Some(pb::declaration::Kind::Constructor(c)) =
                    &mut candidate.declarations[1].kind
                else {
                    unreachable!()
                };
                c.inputs.push(pb::Arg {
                    name: "x".into(),
                    sort: 0,
                });
            }
            12 => {
                let Some(pb::declaration::Kind::Function(f)) = &mut candidate.declarations[2].kind
                else {
                    unreachable!()
                };
                f.inputs.push(pb::Arg {
                    name: "x".into(),
                    sort: 0,
                });
            }
            13 => candidate.sorts[0].kind = Some(pb::sort::Kind::Var(0)),
            _ => {
                let Some(pb::declaration::Kind::HostPrimitive(h)) =
                    &mut candidate.declarations[3].kind
                else {
                    unreachable!()
                };
                h.typing = Some(pb::host_primitive::Typing::Application(
                    pb::FunctionApplication {},
                ));
            }
        }
        let before = destination.clone();
        assert!(
            reconcile_declarations(&candidate, &mut destination).is_err(),
            "bad case {bad}"
        );
        assert_eq!(destination, before, "bad case {bad}");
        let mut fresh = pb::Program::default();
        assert!(
            reconcile_declarations(&candidate, &mut fresh).is_err(),
            "fresh bad case {bad}"
        );
        assert_eq!(fresh, pb::Program::default());
    }
}

fn box_definition() -> pb::Program {
    pb::Program {
        ir_version: 1,
        sorts: vec![
            pb::Sort {
                kind: Some(pb::sort::Kind::Family(pb::HostSort {
                    name: "i64".into(),
                    args: vec![],
                })),
                ..Default::default()
            },
            pb::Sort {
                kind: Some(pb::sort::Kind::Eq("Box".into())),
                ..Default::default()
            },
        ],
        nodes: vec![
            pb::Node {
                sort_id: 0,
                kind: Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                    value: Some(pb::primitive_value::Value::I64(4)),
                })),
                ..Default::default()
            },
            pb::Node {
                sort_id: 0,
                kind: Some(pb::node::Kind::Call(pb::Call {
                    func: "current".into(),
                    args: vec![],
                })),
                ..Default::default()
            },
            pb::Node {
                sort_id: 0,
                kind: Some(pb::node::Kind::Var("new".into())),
                ..Default::default()
            },
        ],
        declarations: vec![
            pb::Declaration {
                kind: Some(pb::declaration::Kind::EqSort(pb::EqSort {
                    name: "Box".into(),
                    bindings: Some(pb::SortBindings {
                        python: Some(pb::TypeBinding {
                            path: vec!["test".into(), "Box".into()],
                            ..Default::default()
                        }),
                        ..Default::default()
                    }),
                })),
                ..Default::default()
            },
            pb::Declaration {
                kind: Some(pb::declaration::Kind::Constructor(pb::Constructor {
                    name: "Box".into(),
                    inputs: vec![pb::Arg {
                        sort: 0,
                        name: "value".into(),
                    }],
                    output: 1,
                    ..Default::default()
                })),
                bindings: Some(pb::CallableBindings {
                    python: Some(pb::PythonBindings {
                        views: vec![pb::PythonCallable {
                            kind: pb::PythonCallKind::Initializer.into(),
                            owner: Some(pb::BindingOwner {
                                kind: Some(pb::binding_owner::Kind::Sort(1)),
                            }),
                            params: vec![pb::PythonParameter {
                                core_input: Some(0),
                                name: "value".into(),
                                default_expr: Some(0),
                            }],
                            ..Default::default()
                        }],
                    }),
                    rust: Some(pb::RustBindings {
                        views: vec![pb::RustCallable {
                            path: vec!["new".into()],
                            owner: Some(pb::BindingOwner {
                                kind: Some(pb::binding_owner::Kind::Sort(1)),
                            }),
                            params: vec![pb::RustParameter {
                                core_input: Some(0),
                                name: "value".into(),
                                borrowed: false,
                            }],
                            ..Default::default()
                        }],
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            },
            pb::Declaration {
                kind: Some(pb::declaration::Kind::Function(pb::Function {
                    name: "current".into(),
                    inputs: vec![],
                    output: 0,
                    merge: Some(2),
                })),
                ..Default::default()
            },
        ],
        ..Default::default()
    }
}

#[test]
fn pair_defaults_validate_order_presence_and_structural_child_sorts() {
    for reverse in [false, true] {
        let mut source = pair_default_definition();
        if reverse {
            let Some(pb::sort::Kind::Family(f)) = &mut source.sorts[3].kind else {
                unreachable!()
            };
            f.args.reverse();
            let Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                value: Some(pb::primitive_value::Value::Pair(p)),
            })) = &mut source.nodes[2].kind
            else {
                unreachable!()
            };
            std::mem::swap(&mut p.first, &mut p.second);
        }
        // Incoming node and family child use different indices for equal sorts.
        source.sorts.push(source.sorts[0].clone());
        source.nodes[0].sort_id = 4;
        let mut destination = pb::Program::default();
        reconcile_declarations(&source, &mut destination).unwrap();
        let frozen = destination.clone();
        for _ in 0..3 {
            reconcile_declarations(&source, &mut destination).unwrap();
            assert_eq!(destination, frozen);
        }
        for bad in 0..8 {
            let mut invalid = source.clone();
            match bad {
                0..=4 => {
                    let Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                        value: Some(pb::primitive_value::Value::Pair(p)),
                    })) = &mut invalid.nodes[2].kind
                    else {
                        unreachable!()
                    };
                    match bad {
                        0 => p.first = None,
                        1 => p.second = None,
                        2 => p.first = Some(999),
                        3 => std::mem::swap(&mut p.first, &mut p.second),
                        _ => p.second = Some(2),
                    }
                }
                5 => invalid.nodes[0].kind = Some(pb::node::Kind::Var("unbound".into())),
                6 => {
                    let Some(pb::sort::Kind::Family(f)) = &mut invalid.sorts[3].kind else {
                        unreachable!()
                    };
                    f.args.pop();
                }
                _ => invalid.nodes[2].sort_id = 0,
            }
            let mut fresh = pb::Program::default();
            assert!(
                reconcile_declarations(&invalid, &mut fresh).is_err(),
                "bad {bad}"
            );
            assert_eq!(fresh, pb::Program::default());
            assert!(
                reconcile_declarations(&invalid, &mut destination).is_err(),
                "bad resupply {bad}"
            );
            assert_eq!(destination, frozen);
        }
    }
}

#[test]
fn map_defaults_validate_ordered_entries_and_closed_child_sorts() {
    let mut source = pair_default_definition();
    let Some(pb::sort::Kind::Family(f)) = &mut source.sorts[3].kind else {
        unreachable!()
    };
    f.name = "Map".into();
    let Some(pb::declaration::Kind::HostSortFamily(f)) =
        &mut source.declarations.last_mut().unwrap().kind
    else {
        unreachable!()
    };
    f.name = "Map".into();
    source.nodes[2].kind = Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
        value: Some(pb::primitive_value::Value::Map(pb::MapValue {
            entries: vec![
                pb::MapEntry {
                    key: Some(0),
                    value: Some(1),
                },
                pb::MapEntry {
                    key: Some(0),
                    value: Some(1),
                },
            ],
        })),
    }));
    let mut destination = pb::Program::default();
    reconcile_declarations(&source, &mut destination).unwrap();
    let frozen = destination.clone();
    reconcile_declarations(&source, &mut destination).unwrap();
    assert_eq!(destination, frozen);
    assert!(destination.nodes.iter().any(|n| matches!(&n.kind,
        Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue { value: Some(pb::primitive_value::Value::Map(m)) }))
        if m.entries.len() == 2 && m.entries[0] == m.entries[1] && m.entries[0].key == Some(0))));
    for bad in 0..10 {
        let mut invalid = source.clone();
        let Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
            value: Some(pb::primitive_value::Value::Map(m)),
        })) = &mut invalid.nodes[2].kind
        else {
            unreachable!()
        };
        match bad {
            0 => m.entries[0].key = None,
            1 => m.entries[0].value = None,
            2 => m.entries[0].key = Some(999),
            3 => m.entries[0].value = Some(999),
            4 => m.entries[0].key = Some(1),
            5 => m.entries[0].value = Some(0),
            6 => m.entries[0].value = Some(2),
            7 => invalid.nodes[0].kind = Some(pb::node::Kind::Var("free".into())),
            8 => invalid.nodes[2].sort_id = 0,
            _ => invalid.sorts[0].kind = Some(pb::sort::Kind::Var(0)),
        }
        assert!(
            reconcile_declarations(&invalid, &mut destination).is_err(),
            "bad {bad}"
        );
        assert_eq!(destination, frozen);
    }
    let Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
        value: Some(pb::primitive_value::Value::Map(m)),
    })) = &mut source.nodes[2].kind
    else {
        unreachable!()
    };
    m.entries.clear();
    reconcile_declarations(&source, &mut pb::Program::default()).unwrap();
}

fn pair_default_definition() -> pb::Program {
    let mut p = box_definition();
    p.sorts.extend([
        pb::Sort {
            kind: Some(pb::sort::Kind::Family(pb::HostSort {
                name: "String".into(),
                args: vec![],
            })),
            ..Default::default()
        },
        pb::Sort {
            kind: Some(pb::sort::Kind::Family(pb::HostSort {
                name: "Pair".into(),
                args: vec![0, 2],
            })),
            ..Default::default()
        },
    ]);
    p.nodes.truncate(1);
    p.nodes.extend([
        pb::Node {
            sort_id: 2,
            kind: Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                value: Some(pb::primitive_value::Value::String("x".into())),
            })),
            ..Default::default()
        },
        pb::Node {
            sort_id: 3,
            kind: Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                value: Some(pb::primitive_value::Value::Pair(pb::PairValue {
                    first: Some(0),
                    second: Some(1),
                })),
            })),
            ..Default::default()
        },
    ]);
    let Some(pb::declaration::Kind::Constructor(c)) = &mut p.declarations[1].kind else {
        unreachable!()
    };
    c.inputs[0].sort = 3;
    p.declarations[1]
        .bindings
        .as_mut()
        .unwrap()
        .python
        .as_mut()
        .unwrap()
        .views[0]
        .params[0]
        .default_expr = Some(2);
    let Some(pb::declaration::Kind::Function(f)) = &mut p.declarations[2].kind else {
        unreachable!()
    };
    f.merge = None;
    p.declarations.push(pb::Declaration {
        kind: Some(pb::declaration::Kind::HostSortFamily(pb::HostSortFamily {
            name: "Pair".into(),
            arity: 2,
            bindings: None,
        })),
        ..Default::default()
    });
    p
}

#[test]
fn pair_defaults_preserve_duplicate_slots_nested_values_and_union_topology() {
    let mut duplicate = pair_default_definition();
    let Some(pb::sort::Kind::Family(f)) = &mut duplicate.sorts[3].kind else {
        unreachable!()
    };
    f.args = vec![0, 0];
    duplicate.nodes[2].kind = Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
        value: Some(pb::primitive_value::Value::Pair(pb::PairValue {
            first: Some(0),
            second: Some(0),
        })),
    }));
    let mut stored = pb::Program::default();
    reconcile_declarations(&duplicate, &mut stored).unwrap();
    assert!(stored.nodes.iter().any(|n| matches!(&n.kind,
        Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue { value: Some(pb::primitive_value::Value::Pair(p)) })) if p.first == Some(0) && p.second == Some(0))));

    let mut nested = pair_default_definition();
    nested.sorts.extend([
        pb::Sort {
            kind: Some(pb::sort::Kind::Family(pb::HostSort {
                name: "Vec".into(),
                args: vec![3],
            })),
            ..Default::default()
        },
        pb::Sort {
            kind: Some(pb::sort::Kind::Family(pb::HostSort {
                name: "Pair".into(),
                args: vec![3, 4],
            })),
            ..Default::default()
        },
    ]);
    nested.nodes.extend([
        pb::Node {
            sort_id: 4,
            kind: Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                value: Some(pb::primitive_value::Value::Vec(pb::ValueList {
                    items: vec![2, 2],
                })),
            })),
            ..Default::default()
        },
        pb::Node {
            sort_id: 5,
            kind: Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                value: Some(pb::primitive_value::Value::Pair(pb::PairValue {
                    first: Some(2),
                    second: Some(3),
                })),
            })),
            ..Default::default()
        },
    ]);
    let Some(pb::declaration::Kind::Constructor(c)) = &mut nested.declarations[1].kind else {
        unreachable!()
    };
    c.inputs[0].sort = 5;
    nested.declarations[1]
        .bindings
        .as_mut()
        .unwrap()
        .python
        .as_mut()
        .unwrap()
        .views[0]
        .params[0]
        .default_expr = Some(4);
    nested.declarations.push(pb::Declaration {
        kind: Some(pb::declaration::Kind::HostSortFamily(pb::HostSortFamily {
            name: "Vec".into(),
            arity: 1,
            bindings: None,
        })),
        ..Default::default()
    });
    let mut destination = pb::Program::default();
    reconcile_declarations(&nested, &mut destination).unwrap();
    let frozen = destination.clone();
    nested.sorts.reverse();
    nested.nodes.reverse();
    for s in &mut nested.sorts {
        if let Some(pb::sort::Kind::Family(f)) = &mut s.kind {
            for i in &mut f.args {
                *i = 5 - *i;
            }
        }
    }
    for n in &mut nested.nodes {
        n.sort_id = 5 - n.sort_id;
        match &mut n.kind {
            Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                value: Some(pb::primitive_value::Value::Pair(p)),
            })) => {
                p.first = p.first.map(|i| 4 - i);
                p.second = p.second.map(|i| 4 - i);
            }
            Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                value: Some(pb::primitive_value::Value::Vec(v)),
            })) => {
                for i in &mut v.items {
                    *i = 4 - *i;
                }
            }
            _ => (),
        }
    }
    let Some(pb::declaration::Kind::Constructor(c)) = &mut nested.declarations[1].kind else {
        unreachable!()
    };
    c.inputs[0].sort = 0;
    c.output = 4;
    let b = nested.declarations[1].bindings.as_mut().unwrap();
    b.python.as_mut().unwrap().views[0].params[0].default_expr = Some(0);
    b.python.as_mut().unwrap().views[0]
        .owner
        .as_mut()
        .unwrap()
        .kind = Some(pb::binding_owner::Kind::Sort(4));
    b.rust.as_mut().unwrap().views[0]
        .owner
        .as_mut()
        .unwrap()
        .kind = Some(pb::binding_owner::Kind::Sort(4));
    let Some(pb::declaration::Kind::Function(f)) = &mut nested.declarations[2].kind else {
        unreachable!()
    };
    f.output = 5;
    for _ in 0..3 {
        reconcile_declarations(&nested, &mut destination).unwrap();
        assert_eq!(destination, frozen);
    }

    let mut shared = pair_default_definition();
    let Some(pb::sort::Kind::Family(f)) = &mut shared.sorts[3].kind else {
        unreachable!()
    };
    f.args = vec![1, 1];
    shared.nodes = ["Leaf", "Other"]
        .into_iter()
        .map(|name| pb::Node {
            sort_id: 1,
            kind: Some(pb::node::Kind::Call(pb::Call {
                func: name.into(),
                args: vec![],
            })),
            ..Default::default()
        })
        .collect();
    for name in ["Leaf", "Other"] {
        shared.declarations.push(pb::Declaration {
            kind: Some(pb::declaration::Kind::Constructor(pb::Constructor {
                name: name.into(),
                output: 1,
                ..Default::default()
            })),
            ..Default::default()
        });
    }
    shared.nodes.push(pb::Node {
        sort_id: 1,
        kind: Some(pb::node::Kind::Union(pb::Union {
            members: vec![0, 1],
        })),
        ..Default::default()
    });
    shared.nodes.push(pb::Node {
        sort_id: 3,
        kind: Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
            value: Some(pb::primitive_value::Value::Pair(pb::PairValue {
                first: Some(2),
                second: Some(2),
            })),
        })),
        ..Default::default()
    });
    shared.declarations[1]
        .bindings
        .as_mut()
        .unwrap()
        .python
        .as_mut()
        .unwrap()
        .views[0]
        .params[0]
        .default_expr = Some(3);
    let mut independent = shared.clone();
    independent.nodes.push(shared.nodes[2].clone());
    let Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
        value: Some(pb::primitive_value::Value::Pair(p)),
    })) = &mut independent.nodes[3].kind
    else {
        unreachable!()
    };
    p.second = Some(4);
    for (first, later) in [(&shared, &independent), (&independent, &shared)] {
        let mut destination = pb::Program::default();
        reconcile_declarations(first, &mut destination).unwrap();
        let frozen = destination.clone();
        for _ in 0..3 {
            reconcile_declarations(first, &mut destination).unwrap();
            assert_eq!(destination, frozen);
        }
        assert!(reconcile_declarations(later, &mut destination).is_err());
        assert_eq!(
            destination, frozen,
            "distinct Union topology cannot resupply frozen defaults"
        );
    }
}

#[test]
fn declaration_closures_relocate_and_resupply_structurally() {
    let source = box_definition();
    let mut destination = pb::Program::default();
    assert_eq!(
        reconcile_declarations(&source, &mut destination).unwrap(),
        [true, true, true]
    );
    let installed = destination.clone();
    let mut permuted = source.clone();
    permuted.sorts.reverse();
    permuted.nodes.reverse();
    for n in &mut permuted.nodes {
        n.sort_id = 1 - n.sort_id;
    }
    let Some(pb::declaration::Kind::Constructor(c)) = &mut permuted.declarations[1].kind else {
        unreachable!()
    };
    c.inputs[0].sort = 1;
    c.inputs[0].name = "diagnostic alpha rename".into();
    c.output = 0;
    let b = permuted.declarations[1].bindings.as_mut().unwrap();
    b.python.as_mut().unwrap().views[0]
        .owner
        .as_mut()
        .unwrap()
        .kind = Some(pb::binding_owner::Kind::Sort(0));
    b.python.as_mut().unwrap().views[0].params[0].default_expr = Some(2);
    b.rust.as_mut().unwrap().views[0]
        .owner
        .as_mut()
        .unwrap()
        .kind = Some(pb::binding_owner::Kind::Sort(0));
    let Some(pb::declaration::Kind::Function(f)) = &mut permuted.declarations[2].kind else {
        unreachable!()
    };
    f.output = 1;
    f.merge = Some(0);
    permuted.declarations.reverse();
    assert_eq!(
        reconcile_declarations(&permuted, &mut destination).unwrap(),
        [false, false, false]
    );
    assert_eq!(
        destination, installed,
        "compatible resupply must not retain unreachable template copies"
    );
    let before = destination.clone();
    let Some(pb::declaration::Kind::Function(f)) = &mut permuted.declarations[0].kind else {
        unreachable!()
    };
    f.merge = None;
    assert!(reconcile_declarations(&permuted, &mut destination).is_err());
    assert_eq!(destination, before);
}

#[test]
fn default_validation_is_contextual_typed_and_inert() {
    for bad in 0..6 {
        let mut source = box_definition();
        let root = match bad {
            0 => 2, // new is bound in the merge only.
            1 => {
                source.nodes[0].sort_id = 1;
                0
            }
            2 => {
                source.nodes[0].kind = Some(pb::node::Kind::PrimitiveValue(pb::PrimitiveValue {
                    value: Some(pb::primitive_value::Value::Bool(true)),
                }));
                0
            }
            3 => {
                let Some(pb::node::Kind::Call(c)) = &mut source.nodes[1].kind else {
                    unreachable!()
                };
                c.args.push(0);
                1
            }
            4 => {
                source.nodes[1].sort_id = 1;
                1
            }
            _ => {
                let Some(pb::node::Kind::Call(c)) = &mut source.nodes[1].kind else {
                    unreachable!()
                };
                c.func = "unknown".into();
                1
            }
        };
        source.declarations[1]
            .bindings
            .as_mut()
            .unwrap()
            .python
            .as_mut()
            .unwrap()
            .views[0]
            .params[0]
            .default_expr = Some(root);
        let mut destination = pb::Program::default();
        assert!(
            reconcile_declarations(&source, &mut destination).is_err(),
            "bad default {bad}"
        );
        assert_eq!(destination, pb::Program::default());
    }
    let mut source = box_definition();
    source.declarations[1]
        .bindings
        .as_mut()
        .unwrap()
        .python
        .as_mut()
        .unwrap()
        .views[0]
        .params[0]
        .default_expr = Some(1);
    // No evaluator is available here: a missing current() row is still valid code.
    reconcile_declarations(&source, &mut pb::Program::default()).unwrap();
}

#[test]
fn all_default_roots_preserve_union_identity_but_not_ordinary_sharing() {
    let mut source = box_definition();
    source.nodes.push(pb::Node {
        sort_id: 0,
        kind: Some(pb::node::Kind::Union(pb::Union {
            members: vec![0, 0],
        })),
        ..Default::default()
    });
    let Some(pb::declaration::Kind::Constructor(c)) = &mut source.declarations[1].kind else {
        unreachable!()
    };
    c.inputs.push(c.inputs[0].clone());
    let b = source.declarations[1].bindings.as_mut().unwrap();
    b.rust = None;
    let view = &mut b.python.as_mut().unwrap().views[0];
    view.params[0].default_expr = Some(3);
    view.params.push(pb::PythonParameter {
        core_input: Some(1),
        name: "other".into(),
        default_expr: Some(3),
    });
    let mut destination = pb::Program::default();
    reconcile_declarations(&source, &mut destination).unwrap();
    source.nodes.push(source.nodes[0].clone());
    let Some(pb::node::Kind::Union(u)) = &mut source.nodes[3].kind else {
        unreachable!()
    };
    u.members[1] = 4;
    reconcile_declarations(&source, &mut destination).unwrap();
    source.nodes.push(source.nodes[3].clone());
    source.declarations[1]
        .bindings
        .as_mut()
        .unwrap()
        .python
        .as_mut()
        .unwrap()
        .views[0]
        .params[1]
        .default_expr = Some(5);
    let before = destination.clone();
    assert!(reconcile_declarations(&source, &mut destination).is_err());
    assert_eq!(destination, before);
}

#[test]
fn languages_freeze_independently_and_compatible_imports_do_not_grow() {
    let source = box_definition();
    let mut absent = source.clone();
    absent.declarations[1].bindings = None;
    let mut destination = pb::Program::default();
    reconcile_declarations(&absent, &mut destination).unwrap();
    let mut python = source.clone();
    python.declarations[1].bindings.as_mut().unwrap().rust = None;
    reconcile_declarations(&python, &mut destination).unwrap();
    let mut rust = source.clone();
    rust.declarations[1].bindings.as_mut().unwrap().python = None;
    reconcile_declarations(&rust, &mut destination).unwrap();
    let installed = destination.clone();
    absent.files.push(pb::SourceFile {
        name: "unused".into(),
        contents: None,
    });
    absent.sorts.push(pb::Sort {
        kind: Some(pb::sort::Kind::Eq("Unused".into())),
        ..Default::default()
    });
    for _ in 0..20 {
        reconcile_declarations(&source, &mut destination).unwrap();
        reconcile_declarations(&absent, &mut destination).unwrap();
        assert_eq!(destination, installed);
    }
    python.declarations[1].bindings.as_mut().unwrap().python = Some(pb::PythonBindings::default());
    assert!(reconcile_declarations(&python, &mut destination).is_err());
    assert_eq!(destination, installed);
    let mut first_empty = pb::Program::default();
    reconcile_declarations(&python, &mut first_empty).unwrap();
    assert!(reconcile_declarations(&source, &mut first_empty).is_err());
}

#[test]
fn malformed_surface_mappings_reject_transactionally() {
    for invalid in 0..8 {
        let mut source = box_definition();
        let b = source.declarations[1].bindings.as_mut().unwrap();
        let v = &mut b.python.as_mut().unwrap().views[0];
        match invalid {
            0 => v.params[0].core_input = None,
            1 => v.params.push(v.params[0].clone()),
            2 => v.owner.as_mut().unwrap().kind = Some(pb::binding_owner::Kind::Sort(0)),
            3 => v.receiver = Some(0),
            4 => v.params[0].default_expr = Some(999),
            5 => {
                b.rust.as_mut().unwrap().views[0].receiver = Some(pb::RustReceiver {
                    core_input: Some(0),
                    borrowed: false,
                })
            }
            6 => b.rust.as_mut().unwrap().views[0].borrowed_self = true,
            _ => {
                source.nodes[1].kind = Some(pb::node::Kind::Call(pb::Call {
                    func: "current".into(),
                    args: vec![1],
                }));
                v.params[0].default_expr = Some(1);
            }
        }
        let mut destination = pb::Program::default();
        assert!(
            reconcile_declarations(&source, &mut destination).is_err(),
            "invalid binding {invalid}"
        );
        assert_eq!(destination, pb::Program::default());
    }
}

#[test]
fn default_calls_determine_all_generic_parameters_including_empty_results() {
    for variadic in [false, true] {
        let mut source = box_definition();
        source.sorts.push(pb::Sort {
            kind: Some(pb::sort::Kind::Var(0)),
            ..Default::default()
        });
        source.declarations.push(pb::Declaration {
            kind: Some(pb::declaration::Kind::HostPrimitive(pb::HostPrimitive {
                name: "generic".into(),
                typing: Some(pb::host_primitive::Typing::Signature(
                    pb::GenericSignature {
                        type_params: vec!["T".into()],
                        inputs: vec![],
                        output: Some(0),
                        varargs: if variadic {
                            vec![pb::Arg {
                                sort: 2,
                                name: "tail".into(),
                            }]
                        } else {
                            vec![]
                        },
                        ..Default::default()
                    },
                )),
            })),
            ..Default::default()
        });
        source.nodes[1].kind = Some(pb::node::Kind::Call(pb::Call {
            func: "generic".into(),
            args: vec![],
        }));
        source.declarations[1]
            .bindings
            .as_mut()
            .unwrap()
            .python
            .as_mut()
            .unwrap()
            .views[0]
            .params[0]
            .default_expr = Some(1);
        assert!(
            reconcile_declarations(&source, &mut pb::Program::default()).is_err(),
            "undetermined T with variadic={variadic}"
        );
        if variadic {
            let Some(pb::node::Kind::Call(c)) = &mut source.nodes[1].kind else {
                unreachable!()
            };
            c.args.push(0);
            reconcile_declarations(&source, &mut pb::Program::default()).unwrap();
        }
    }
    use egglog::sort::Presort;
    let mut source = box_definition();
    egglog::builtin::import_definition(
        &egglog::sort::VecSort::builtin_definitions()[0],
        &mut source,
    )
    .unwrap();
    let concrete = source.sorts.len() as u32;
    source.sorts.push(pb::Sort {
        kind: Some(pb::sort::Kind::Family(pb::HostSort {
            name: "Vec".into(),
            args: vec![0],
        })),
        ..Default::default()
    });
    let root = source.nodes.len() as u32;
    source.nodes.push(pb::Node {
        sort_id: concrete,
        kind: Some(pb::node::Kind::Call(pb::Call {
            func: "egglog.core.vec.empty".into(),
            args: vec![],
        })),
        ..Default::default()
    });
    let Some(pb::declaration::Kind::Constructor(c)) = &mut source.declarations[1].kind else {
        unreachable!()
    };
    c.inputs[0].sort = concrete;
    source.declarations[1]
        .bindings
        .as_mut()
        .unwrap()
        .python
        .as_mut()
        .unwrap()
        .views[0]
        .params[0]
        .default_expr = Some(root);
    reconcile_declarations(&source, &mut pb::Program::default()).unwrap();
    let Some(pb::sort::Kind::Family(f)) = &mut source.sorts[concrete as usize].kind else {
        unreachable!()
    };
    f.args.clear();
    assert!(reconcile_declarations(&source, &mut pb::Program::default()).is_err());
}
