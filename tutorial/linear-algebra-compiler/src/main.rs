use lalrpop_util::lalrpop_mod;

lalrpop_mod!(pub grammar);

pub fn parse_matrix_type(input: &str) -> Type {
    todo!()
}

pub enum Binding {
    Bind { var: String, expr: Expr },
    Declare { var: String, ty: Type },
}

pub enum Expr {
    Var(String),
    Num(i64),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
}

pub enum Type {
    Scalar,
    Matrix { nrows: usize, ncols: usize },
}

fn main() {
    println!("Hello, world!");
}
