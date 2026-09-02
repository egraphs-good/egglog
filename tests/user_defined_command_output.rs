use std::{any::Any, sync::Arc};

use egglog::UserDefinedCommandOutput;

#[test]
fn user_defined_command_output_can_be_downcast() {
    let output: Arc<dyn UserDefinedCommandOutput> = Arc::new(String::from("structured output"));
    let output: &dyn Any = output.as_ref();

    assert_eq!(
        output.downcast_ref::<String>().map(String::as_str),
        Some("structured output")
    );
    assert!(output.downcast_ref::<usize>().is_none());
}
