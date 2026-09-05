use crate::{core_relations::BaseValuePrinter, *};
use egraph_comparison::{Class, Database, Error as ComparisonError, FunctionKind, Row};
use std::any::TypeId;

impl EGraph {
    /// Export a complete database for `egraph-comparison`, rebuilding first.
    ///
    /// Includes every visible constructor/function/relation, including empty
    /// tables and subsumed rows. Global bindings and hidden implementation tables
    /// are excluded. Costs and extraction preferences are not database contents.
    /// The initial exporter supports equality sorts and built-in scalar sorts.
    /// Unsupported containers/custom base types and term/proof encoding return
    /// an error, never a silently incomplete comparison database.
    #[cfg_attr(docsrs, doc(cfg(feature = "comparison")))]
    pub fn serialize_for_comparison(&mut self) -> Result<Database, ComparisonError> {
        if self.proof_state.original_typechecking.is_some() {
            return Err(ComparisonError(
                "comparison export does not yet project term/proof encoding to user tables".into(),
            ));
        }
        self.backend.flush_updates_no_rebuild();
        self.backend
            .rebuild_now()
            .map_err(|e| ComparisonError(e.to_string()))?;
        let mut database = Database::default();
        for (name, function) in &self.functions {
            if function.is_hidden() || function.is_let_binding() {
                continue;
            }
            let signature = function.func_type();
            // Relations use constructor tables internally, with a generated
            // non-unionable output sort. They do not build e-graph terms.
            let kind = if signature.subtype == FunctionSubtype::Constructor
                && !self.comparison_relation_sort(&signature.output)
            {
                FunctionKind::Constructor
            } else {
                FunctionKind::Function
            };
            database.functions.insert(
                name.clone(),
                egraph_comparison::Function {
                    kind,
                    inputs: signature
                        .input
                        .iter()
                        .map(|s| self.comparison_sort_name(s))
                        .collect(),
                    output: self.comparison_sort_name(&signature.output),
                },
            );
            let mut error = None;
            self.backend.for_each_while(function.backend_id, |row| {
                let result = (|| {
                    let (output, inputs) = row.vals.split_last().unwrap();
                    let inputs = inputs
                        .iter()
                        .zip(&signature.input)
                        .map(|(&value, sort)| self.comparison_value(&mut database, sort, value))
                        .collect::<Result<Vec<_>, _>>()?;
                    let output =
                        self.comparison_value(&mut database, &signature.output, *output)?;
                    database.rows.push(Row {
                        function: name.clone(),
                        inputs,
                        output,
                        subsumed: row.subsumed,
                    });
                    Ok::<_, ComparisonError>(())
                })();
                match result {
                    Ok(()) => true,
                    Err(e) => {
                        error = Some(e);
                        false
                    }
                }
            });
            if let Some(error) = error {
                return Err(error);
            }
        }
        database.validate()?;
        Ok(database)
    }

    fn comparison_value(
        &self,
        database: &mut Database,
        sort: &ArcSort,
        value: Value,
    ) -> Result<String, ComparisonError> {
        let id = self.value_to_class_id(sort, value).to_string();
        if database.classes.contains_key(&id) {
            return Ok(id);
        }
        let literal = if sort.is_eq_sort() {
            None
        } else {
            let ty = sort.value_type();
            let supported = [
                TypeId::of::<i64>(),
                TypeId::of::<bool>(),
                TypeId::of::<()>(),
                TypeId::of::<S>(),
                TypeId::of::<F>(),
                TypeId::of::<Z>(),
                TypeId::of::<Q>(),
            ];
            if sort.is_container_sort() || !ty.is_some_and(|ty| supported.contains(&ty)) {
                return Err(ComparisonError(format!(
                    "comparison export does not support sort {} yet",
                    sort.name()
                )));
            }
            let value = if ty == Some(TypeId::of::<F>()) {
                let float = self.value_to_base::<F>(value).0.0;
                // OrderedFloat equates both zeros and all NaNs. The interned
                // representative may otherwise depend on insertion order.
                if float == 0.0 {
                    "0.0".into()
                } else if float.is_nan() {
                    "NaN".into()
                } else {
                    format!("{float:?}")
                }
            } else {
                format!(
                    "{:?}",
                    BaseValuePrinter {
                        base: self.backend.base_values(),
                        ty: self.backend.base_values().get_ty_by_id(ty.unwrap()),
                        val: value
                    }
                )
            };
            Some(value)
        };
        database.classes.insert(
            id.clone(),
            Class {
                sort: self.comparison_sort_name(sort),
                literal,
            },
        );
        Ok(id)
    }

    fn comparison_relation_sort(&self, sort: &ArcSort) -> bool {
        // Normal desugaring creates reserved, non-unionable relation sorts.
        // User-declared :no-union sorts still have ordinary constructors.
        !self.type_info.is_sort_unionable(sort)
            && sort.is_eq_sort()
            && sort.name().starts_with(util::INTERNAL_SYMBOL_PREFIX)
    }

    fn comparison_sort_name(&self, sort: &ArcSort) -> String {
        if self.comparison_relation_sort(sort) {
            // Generated names can depend on declaration order (notably when
            // relation names differ only by dashes). Use their table names.
            let mut names: Vec<_> = self
                .functions
                .values()
                .filter(|f| {
                    !f.is_hidden()
                        && !f.is_let_binding()
                        && f.func_type().subtype == FunctionSubtype::Constructor
                        && f.func_type().output.name() == sort.name()
                })
                .map(Function::name)
                .collect();
            names.sort_unstable();
            format!(
                "{}relation:{}",
                util::INTERNAL_SYMBOL_PREFIX,
                serde_json::to_string(&names).unwrap()
            )
        } else {
            sort.name().to_owned()
        }
    }
}
