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
pub struct Bindings(Vec<Binding>);

impl Bindings {
    pub fn lower(&self) -> Result<Vec<CoreBinding>, TypeError> {
        let mut core_bindings = vec![];
        let mut env = vec![];
        for binding in &self.0 {
            match binding {
                Binding::Bind { var, expr } => {
                    let (expr, ty) = expr.lower(&env)?;
                    core_bindings.push(CoreBinding::Bind {
                        var: var.clone(),
                        expr,
                    });
                    env.push((var.clone(), ty));
                }
                Binding::Declare { var, ty } => {
                    core_bindings.push(CoreBinding::Declare {
                        var: var.clone(),
                        ty: *ty,
                    });
                    env.push((var.clone(), *ty));
                }
            }
        }
        Ok(core_bindings)
    }
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

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Type {
    Scalar,
    Matrix { nrows: usize, ncols: usize },
}

#[derive(Debug, PartialEq, Eq)]
pub enum CoreBinding {
    Bind { var: String, expr: CoreExpr },
    Declare { var: String, ty: Type },
}

#[derive(Debug, PartialEq, Eq)]
pub enum CoreExpr {
    SVar(String),
    MVar {
        name: String,
        nrows: usize,
        ncols: usize,
    },
    Num(i64),
    SAdd(Box<CoreExpr>, Box<CoreExpr>),
    SMul(Box<CoreExpr>, Box<CoreExpr>),
    MAdd(Box<CoreExpr>, Box<CoreExpr>),
    MMul(Box<CoreExpr>, Box<CoreExpr>),
    Scale(Box<CoreExpr>, Box<CoreExpr>),
    SSub(Box<CoreExpr>, Box<CoreExpr>),
    SDiv(Box<CoreExpr>, Box<CoreExpr>),
}

#[derive(Debug)]
#[allow(dead_code)]
pub enum TypeError {
    ExpectedScalar,
    MatrixDimensionMismatch {
        op: &'static str,
        left: (usize, usize),
        right: (usize, usize),
    },
    UndeclaredVariable(String),
}

impl Expr {
    pub fn lower(&self, env: &[(String, Type)]) -> Result<(CoreExpr, Type), TypeError> {
        match self {
            Expr::Var(name) => {
                // Look up the variable in the environment
                match env.iter().find(|(var, _)| var == name) {
                    Some((_, Type::Scalar)) => Ok((CoreExpr::SVar(name.clone()), Type::Scalar)),
                    Some((_, Type::Matrix { nrows, ncols })) => Ok((
                        CoreExpr::MVar {
                            name: name.clone(),
                            nrows: *nrows,
                            ncols: *ncols,
                        },
                        Type::Matrix {
                            nrows: *nrows,
                            ncols: *ncols,
                        },
                    )),
                    None => Err(TypeError::UndeclaredVariable(name.clone())),
                }
            }
            Expr::Num(n) => Ok((CoreExpr::Num(*n), Type::Scalar)),
            Expr::Add(left, right) => {
                let l = left.lower(env)?;
                let r = right.lower(env)?;
                match (&l.1, &r.1) {
                    // Scalar addition
                    (Type::Scalar, Type::Scalar) => {
                        Ok((CoreExpr::SAdd(Box::new(l.0), Box::new(r.0)), Type::Scalar))
                    }
                    (
                        Type::Matrix { nrows, ncols },
                        Type::Matrix {
                            nrows: r2,
                            ncols: c2,
                        },
                    ) => {
                        if nrows == r2 && ncols == c2 {
                            Ok((
                                CoreExpr::MAdd(Box::new(l.0), Box::new(r.0)),
                                Type::Matrix {
                                    nrows: *nrows,
                                    ncols: *ncols,
                                },
                            ))
                        } else {
                            Err(TypeError::MatrixDimensionMismatch {
                                op: "add",
                                left: (*nrows, *ncols),
                                right: (*r2, *c2),
                            })
                        }
                    }
                    _ => Err(TypeError::ExpectedScalar),
                }
            }
            Expr::Mul(left, right) => {
                let l = left.lower(env)?;
                let r = right.lower(env)?;
                match (&l.1, &r.1) {
                    (Type::Scalar, Type::Scalar) => {
                        Ok((CoreExpr::SMul(Box::new(l.0), Box::new(r.0)), Type::Scalar))
                    }
                    (
                        Type::Matrix { nrows, ncols },
                        Type::Matrix {
                            nrows: r2,
                            ncols: c2,
                        },
                    ) => {
                        if nrows == r2 && ncols == c2 {
                            Ok((
                                CoreExpr::MMul(Box::new(l.0), Box::new(r.0)),
                                Type::Matrix {
                                    nrows: *nrows,
                                    ncols: *ncols,
                                },
                            ))
                        } else {
                            Err(TypeError::MatrixDimensionMismatch {
                                op: "mul",
                                left: (*nrows, *ncols),
                                right: (*r2, *c2),
                            })
                        }
                    }
                    (Type::Scalar, Type::Matrix { nrows, ncols }) => Ok((
                        CoreExpr::Scale(Box::new(l.0), Box::new(r.0)),
                        Type::Matrix {
                            nrows: *nrows,
                            ncols: *ncols,
                        },
                    )),
                    _ => Err(TypeError::ExpectedScalar),
                }
            }
            Expr::Sub(left, right) => {
                let l = left.lower(env)?;
                let r = right.lower(env)?;
                match (&l.1, &r.1) {
                    (Type::Scalar, Type::Scalar) => {
                        Ok((CoreExpr::SSub(Box::new(l.0), Box::new(r.0)), Type::Scalar))
                    }
                    _ => Err(TypeError::ExpectedScalar),
                }
            }
            Expr::Div(left, right) => {
                let l = left.lower(env)?;
                let r = right.lower(env)?;
                match (&l.1, &r.1) {
                    (Type::Scalar, Type::Scalar) => {
                        Ok((CoreExpr::SDiv(Box::new(l.0), Box::new(r.0)), Type::Scalar))
                    }
                    _ => Err(TypeError::ExpectedScalar),
                }
            }
        }
    }
}

#[test]
fn test_parser() {
    use Binding::*;
    use Expr::*;
    use Type::*;
    let bindings = grammar::BindingsParser::new().parse("x: R; y: R; A: [R; 2x2];");
    assert_eq!(
        bindings.unwrap().0,
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
        bindings.unwrap().0,
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
        bindings.unwrap().0,
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

#[test]
fn test_lowering() {
    use Expr::*;

    // Test environment
    let env = vec![
        ("x".to_string(), Type::Scalar),
        ("y".to_string(), Type::Scalar),
        ("A".to_string(), Type::Matrix { nrows: 2, ncols: 3 }),
        ("B".to_string(), Type::Matrix { nrows: 3, ncols: 2 }),
    ];

    // Test scalar operations
    let expr = Add(
        Box::new(Mul(Box::new(Var("x".to_string())), Box::new(Num(2)))),
        Box::new(Var("y".to_string())),
    );
    assert!(expr.lower(&env).is_ok());

    // Test matrix multiplication with correct dimensions
    let expr = Mul(
        Box::new(Var("A".to_string())),
        Box::new(Var("B".to_string())),
    );
    assert!(expr.lower(&env).is_ok());

    // Test matrix multiplication with incorrect dimensions
    let expr = Mul(
        Box::new(Var("B".to_string())),
        Box::new(Var("B".to_string())),
    );
    assert!(matches!(
        expr.lower(&env),
        Err(TypeError::MatrixDimensionMismatch { .. })
    ));

    // Test matrix-scalar multiplication
    let expr = Mul(
        Box::new(Var("A".to_string())),
        Box::new(Var("x".to_string())),
    );
    assert!(expr.lower(&env).is_ok());

    // Test undefined variable
    let expr = Var("z".to_string());
    assert!(matches!(
        expr.lower(&env),
        Err(TypeError::UndeclaredVariable(_))
    ));
}
