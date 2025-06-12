mod ast;
fn main() {
    let bindings = ast::grammar::BindingsParser::new().parse("x: R; y: R; A: [R; 2x2]; B = (x + y) * A;").expect("parsing bindings");
    let core_bindings = match bindings.lower() {
        Ok(bindings) => bindings,
        Err(e) => {
            println!("Error: {:?}", e);
            return;
        }
    };
    println!("core_bindings: {:?}", core_bindings);
    println!("Hello, world!");
}
