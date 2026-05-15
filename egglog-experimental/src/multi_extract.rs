//! An implementation of multi-extraction for egraphs.
//! Adds support for extracting multiple terms with a single command,
//! reducing the overhead of creating an extractor for each term.
//! The syntax for multi-extraction is `(multi-extract n t1 ... tm)`,
//! where n must be a positive i64.
//! This command will extract n lowest-cost variants of each of the m terms.
//! `(multi-extract 1 t)` is equivalent to `(extract t)`.

use egglog::{
    CommandOutput, EGraph, Error, TermDag, TermId, TypeError, UserDefinedCommand,
    ast::Expr,
    extract::{Cost, CostModel, Extractor},
};
use log::log_enabled;
use std::{fmt::Debug, marker::PhantomData};

#[derive(Debug)]
pub struct MultiExtractOutput {
    termdag: TermDag,
    terms: Vec<Vec<TermId>>,
}

impl std::fmt::Display for MultiExtractOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "(")?;
        for variants in &self.terms {
            writeln!(f, "   (")?;
            for expr in variants {
                writeln!(f, "      {}", self.termdag.to_string(*expr))?;
            }
            writeln!(f, "   )")?;
        }
        writeln!(f, ")")
    }
}

pub struct MultiExtract<C: Cost + Ord + Eq + Clone + Debug + Send + Sync, CM: CostModel<C> + Clone>
{
    cost_model: CM,
    _cost_t: PhantomData<C>,
}

impl<C: Cost + Ord + Eq + Clone + Debug + Send + Sync, CM: CostModel<C> + Clone>
    MultiExtract<C, CM>
{
    pub fn new(cost_model: CM) -> Self {
        MultiExtract {
            cost_model,
            _cost_t: PhantomData,
        }
    }
}

impl<
    C: Cost + Ord + Eq + Clone + Debug + Send + Sync,
    CM: CostModel<C> + Clone + Send + Sync + 'static,
> UserDefinedCommand for MultiExtract<C, CM>
{
    fn update(&self, egraph: &mut EGraph, args: &[Expr]) -> Result<Option<CommandOutput>, Error> {
        assert!(args.len() >= 2);

        let (variants_sort, variants_value) = egraph.eval_expr(&args[0])?;
        if variants_sort.name() != "i64" {
            return Err(Error::TypeError(TypeError::Mismatch {
                expr: args[0].clone(),
                expected: egraph.get_arcsort_by(|s| s.name() == "i64"),
                actual: variants_sort,
            }));
        }

        let n: i64 = egraph.value_to_base(variants_value);
        if n < 0 {
            panic!("Cannot extract negative number of variants");
        }

        let (sorts, values): (Vec<_>, Vec<_>) = args[1..]
            .iter()
            .map(|arg| egraph.eval_expr(arg))
            .collect::<Result<_, _>>()?;

        let mut termdag = TermDag::default();
        let extractor = Extractor::compute_costs_from_rootsorts(
            Some(sorts.clone()),
            egraph,
            self.cost_model.clone(),
        );

        let terms: Vec<Vec<_>> = values
            .into_iter()
            .zip(sorts)
            .map(|(value, sort)| {
                extractor
                    .extract_variants_with_sort(egraph, &mut termdag, value, n as usize, sort)
                    .into_iter()
                    .map(|e| e.1)
                    .collect()
            })
            .collect();

        if log_enabled!(log::Level::Info) {
            log::info!(
                "extracted {} variants for each of {} expressions",
                n,
                terms.len()
            );
        }

        Ok(Some(CommandOutput::UserDefined(std::sync::Arc::from(
            MultiExtractOutput { termdag, terms },
        ))))
    }
}
