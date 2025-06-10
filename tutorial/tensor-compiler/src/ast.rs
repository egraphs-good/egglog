
lalrpop_mod!(grammar);

pub struct Program {
    pub inputs: Vec<Input>,
    pub exprs: Vec<Binding>,
}

pub struct Input {
    pub name: String,
    pub dimension: Vec<usize>,
    pub nnz: usize,
}

pub struct Binding {
    pub name: String,
    pub expr: Expr,
}

pub enum Expr {
    Tensor(String, Vec<String>),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Agg(String, Box<Expr>),
    Lit(egglog::sort::Q)
}


// f(i) = f(i, j, k) * f(k, h) + g(i, j)