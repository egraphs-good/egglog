//! egglog debugging tools.

use std::io::{stdin, stdout, BufRead, Write};

use crate::EGraph;

/// Allows user to directly interact with the egraph via commands on the command
/// line, for debugging purposes. egglog commands are taken over stdin, and
/// results are printed to stdout. Ctrl+D exits the debugging session.
///
/// The following example uses [`egraph_interact_via`] for testing purposes, but
/// the same code will work with [`egraph_interact`].
/// ```
/// use egglog::EGraph;
/// use egglog::debugging::egraph_interact_via;
/// use std::collections::VecDeque;
/// use std::io::{BufRead, Write};
///
/// let mut egraph = EGraph::default();
/// 
/// // Commands sent on the command line.
/// let mut input = VecDeque::from(Vec::from(
/// "(datatype Simple (Foo))
/// (Foo)
/// (print-function Foo 1)"));
/// 
/// let mut output = Vec::new();
/// 
/// // Equivalent to calling egraph_interact() and having the user run the above
/// // commands.
/// egraph_interact_via(&mut egraph, &mut input, &mut output).unwrap();
/// 
/// let output = String::from_utf8(output).unwrap().lines().map(|x| x.trim_start_matches("> ")).filter(|x| !x.is_empty()).map(|x| x.to_string()).collect::<Vec<_>>();
///
/// assert_eq!(output[0], "Now interacting with egraph. Type any egglog command. Use Ctrl+D to exit.");
/// assert_eq!(format!("{}\n{}\n{}", output[1], output[2], output[3]),
/// "(
///    (Foo) -> (Foo)
/// )");
/// ```
pub fn egraph_interact(egraph: &mut EGraph) -> std::io::Result<()> {
    egraph_interact_via(egraph, stdin().lock(), stdout())
}

/// Lower-level method for [`egraph_interact`] that allows the user to specify
/// their own input and output source/sink.
pub fn egraph_interact_via<R, W>(
    egraph: &mut EGraph,
    mut input: R,
    mut output: W,
) -> std::io::Result<()>
where
    R: BufRead,
    W: Write,
{
    writeln!(
        output,
        "Now interacting with egraph. Type any egglog command. Use Ctrl+D to exit."
    )?;

    'interact: loop {
        write!(output, "> ")?;
        output.flush()?;
        let mut buf = String::new();
        input.read_line(&mut buf)?;
        if buf.is_empty() {
            log::info!("EOF while interacting; exiting interact.");
            break 'interact;
        }
        let out = egraph.parse_and_run_program(None, &buf);
        if let Ok(out) = out {
            writeln!(output, "{}", out.join("\n"))?;
        } else {
            writeln!(output, "Error: {:?}", out)?;
        }
    }

    Ok(())
}
