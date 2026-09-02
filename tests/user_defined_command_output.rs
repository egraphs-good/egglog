use std::{fmt::Display, sync::Arc};

use egglog::UserDefinedCommandOutput;

#[test]
fn user_defined_command_output_can_be_downcast() {
    let output: Arc<dyn UserDefinedCommandOutput> = Arc::new(String::from("structured output"));
    let output = output.as_ref().as_any();

    assert_eq!(
        output.downcast_ref::<String>().map(String::as_str),
        Some("structured output")
    );
    assert!(output.downcast_ref::<usize>().is_none());
}

#[derive(Debug)]
struct BorrowedOutput<'a>(&'a str);

impl Display for BorrowedOutput<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

#[test]
fn borrowed_values_still_implement_user_defined_command_output() {
    fn accepts_user_defined_output<T: UserDefinedCommandOutput>(_: &T) {}

    let text = String::from("borrowed output");
    accepts_user_defined_output(&BorrowedOutput(&text));
}
