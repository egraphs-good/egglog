use lalrpop_util::lalrpop_mod;

lalrpop_mod!(pub grammar);
use regex::Regex;

pub fn parse_matrix_type(input: &str) -> Type {
    let re = Regex::new(r":\s*\[R\s*;\s*([0-9]+)x([0-9]+)\s*\]").unwrap();
    let captures = re.captures(input).unwrap();
    let nrows = captures
        .get(1)
        .unwrap()
        .as_str()
        .parse::<usize>()
        .expect("Invalid matrix type");
    let ncols = captures
        .get(2)
        .unwrap()
        .as_str()
        .parse::<usize>()
        .expect("Invalid matrix type");
    Type::Matrix { nrows, ncols }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Binding {
    Bind { var: String, expr: Expr },
    Declare { var: String, ty: Type },
}

#[derive(Debug, PartialEq, Eq)]
pub enum Expr {
    Var(String),
    Num(i64),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
}

#[derive(Debug, PartialEq, Eq)]
pub enum Type {
    Scalar,
    Matrix { nrows: usize, ncols: usize },
}

#[test]
fn test_parser() {
    use Binding::*;
    use Expr::*;
    use Type::*;
    let bindings = grammar::BindingsParser::new().parse("x: R; y: R; A: [R; 2x2];");
    assert_eq!(
        bindings.unwrap(),
        vec![
            Binding::Declare {
                var: "x".to_string(),
                ty: Type::Scalar
            },
            Binding::Declare {
                var: "y".to_string(),
                ty: Type::Scalar
            },
            Binding::Declare {
                var: "A".to_string(),
                ty: Type::Matrix { nrows: 2, ncols: 2 }
            },
        ]
    );
    let bindings = grammar::BindingsParser::new().parse("x: R; y: R; A = x*y*A;");
    assert_eq!(
        bindings.unwrap(),
        vec![
            Binding::Declare {
                var: "x".to_string(),
                ty: Type::Scalar
            },
            Binding::Declare {
                var: "y".to_string(),
                ty: Type::Scalar
            },
            Binding::Bind {
                var: "A".to_string(),
                expr: Expr::Mul(
                    Box::new(Expr::Mul(
                        Box::new(Expr::Var("x".to_string())),
                        Box::new(Expr::Var("y".to_string()))
                    )),
                    Box::new(Expr::Var("A".to_string()))
                )
            },
        ]
    );
    let bindings = grammar::BindingsParser::new().parse("x: R; y: R; A = (1+1+x*y)*A;");

    assert_eq!(
        bindings.unwrap(),
        vec![
            Declare {
                var: "x".to_string(),
                ty: Scalar,
            },
            Declare {
                var: "y".to_string(),
                ty: Scalar,
            },
            Bind {
                var: "A".to_string(),
                expr: Mul(
                    Add(
                        Add(Num(1).into(), Num(1).into()).into(),
                        Mul(Var("x".to_string()).into(), Var("y".to_string()).into(),).into(),
                    )
                    .into(),
                    Var("A".to_string()).into(),
                ),
            },
        ]
    );
}
