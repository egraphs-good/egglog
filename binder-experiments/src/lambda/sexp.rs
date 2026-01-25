use super::{LambdaEnv, LambdaTables};
use egglog_bridge::{EGraph, TableAction};
use egglog_core_relations::Value;
use hashbrown::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub(crate) enum ParseError {
    #[error("unexpected end of input")]
    UnexpectedEof,
    #[error("unexpected token `{0}`")]
    UnexpectedToken(String),
    #[error("expected symbol, found `{0}`")]
    ExpectedSymbol(String),
    #[error("invalid integer literal `{0}`")]
    InvalidInteger(String),
    #[error("invalid form: {0}")]
    InvalidForm(String),
    #[error("unbound variable `{0}`")]
    UnboundVariable(String),
}

#[derive(Clone, Debug)]
enum Token {
    LParen,
    RParen,
    Atom(String),
}

#[derive(Clone, Debug)]
enum Sexp {
    Atom(String),
    List(Vec<Sexp>),
}

pub(crate) fn add_expr_from_sexp(
    egraph: &mut EGraph,
    env: &LambdaEnv,
    input: &str,
) -> Result<Value, ParseError> {
    let sexp = parse_sexp(input)?;
    let mut builder = Builder::new(egraph, &env.tables);
    let expr = builder.eval(&sexp)?;
    egraph.flush_updates();
    Ok(expr)
}

fn parse_sexp(input: &str) -> Result<Sexp, ParseError> {
    let tokens = tokenize(input)?;
    let mut idx = 0;
    let expr = parse_expr(&tokens, &mut idx)?;
    if idx != tokens.len() {
        return Err(ParseError::UnexpectedToken(format!("{:?}", tokens[idx])));
    }
    Ok(expr)
}

fn tokenize(input: &str) -> Result<Vec<Token>, ParseError> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();
    while let Some(&ch) = chars.peek() {
        match ch {
            '(' => {
                chars.next();
                tokens.push(Token::LParen);
            }
            ')' => {
                chars.next();
                tokens.push(Token::RParen);
            }
            c if c.is_whitespace() => {
                chars.next();
            }
            _ => {
                let mut atom = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_whitespace() || c == '(' || c == ')' {
                        break;
                    }
                    atom.push(c);
                    chars.next();
                }
                if atom.is_empty() {
                    return Err(ParseError::UnexpectedToken(ch.to_string()));
                }
                tokens.push(Token::Atom(atom));
            }
        }
    }
    Ok(tokens)
}

fn parse_expr(tokens: &[Token], idx: &mut usize) -> Result<Sexp, ParseError> {
    let Some(token) = tokens.get(*idx) else {
        return Err(ParseError::UnexpectedEof);
    };
    match token {
        Token::LParen => {
            *idx += 1;
            let mut elems = Vec::new();
            loop {
                let Some(token) = tokens.get(*idx) else {
                    return Err(ParseError::UnexpectedEof);
                };
                match token {
                    Token::RParen => {
                        *idx += 1;
                        break;
                    }
                    _ => elems.push(parse_expr(tokens, idx)?),
                }
            }
            Ok(Sexp::List(elems))
        }
        Token::RParen => Err(ParseError::UnexpectedToken(")".to_string())),
        Token::Atom(atom) => {
            *idx += 1;
            Ok(Sexp::Atom(atom.clone()))
        }
    }
}

struct Builder<'a> {
    egraph: &'a mut EGraph,
    tables: &'a LambdaTables,
    scopes: Vec<HashMap<String, Value>>,
}

impl<'a> Builder<'a> {
    fn new(egraph: &'a mut EGraph, tables: &'a LambdaTables) -> Self {
        Self {
            egraph,
            tables,
            scopes: Vec::new(),
        }
    }

    fn eval(&mut self, expr: &Sexp) -> Result<Value, ParseError> {
        match expr {
            Sexp::Atom(atom) => self.eval_atom(atom),
            Sexp::List(list) => self.eval_list(list),
        }
    }

    fn eval_atom(&mut self, atom: &str) -> Result<Value, ParseError> {
        if let Some(val) = self.lookup_var(atom) {
            return self.lookup_var_expr(val);
        }
        Err(ParseError::UnboundVariable(atom.to_string()))
    }

    fn eval_list(&mut self, list: &[Sexp]) -> Result<Value, ParseError> {
        if list.is_empty() {
            return Err(ParseError::InvalidForm("empty list".to_string()));
        }
        if let Sexp::Atom(head) = &list[0] {
            if head == "lam" {
                return self.eval_lam(list);
            }
            if head == "num" {
                return self.eval_num(list);
            }
        }

        let mut iter = list.iter();
        let mut expr = self.eval(
            iter.next()
                .expect("non-empty list should have a head"),
        )?;
        for arg in iter {
            let arg_id = self.eval(arg)?;
            expr = self.lookup_app_expr(expr, arg_id)?;
        }
        Ok(expr)
    }

    fn eval_lam(&mut self, list: &[Sexp]) -> Result<Value, ParseError> {
        let [_, name, body] = list else {
            return Err(ParseError::InvalidForm(
                "lam expects (lam <name> <body>)".to_string(),
            ));
        };
        let name = match name {
            Sexp::Atom(name) => name.as_str(),
            other => {
                return Err(ParseError::ExpectedSymbol(format!("{other:?}")));
            }
        };
        let lam_id = self.egraph.fresh_id();
        self.scopes.push(HashMap::from([(name.to_string(), lam_id)]));
        let body_id = self.eval(body)?;
        self.scopes.pop();

        let mut lam_action = TableAction::new(self.egraph, self.tables.lam);
        let lam_id_copy = lam_id;
        self.egraph.with_execution_state(|state| {
            lam_action.insert(state, [body_id, lam_id_copy].into_iter());
        });

        Ok(lam_id)
    }

    fn eval_num(&mut self, list: &[Sexp]) -> Result<Value, ParseError> {
        let [_, value] = list else {
            return Err(ParseError::InvalidForm(
                "num expects (num <int>)".to_string(),
            ));
        };
        let value = match value {
            Sexp::Atom(atom) => atom
                .parse::<i64>()
                .map_err(|_| ParseError::InvalidInteger(atom.to_string()))?,
            other => {
                return Err(ParseError::ExpectedSymbol(format!("{other:?}")));
            }
        };
        let num_action = TableAction::new(self.egraph, self.tables.num);
        let base_val = self.egraph.base_values().get(value);
        let num_id = self
            .egraph
            .with_execution_state(|state| num_action.lookup(state, &[base_val]));
        num_id.ok_or_else(|| ParseError::InvalidForm("num lookup failed".to_string()))
    }

    fn lookup_var(&self, name: &str) -> Option<Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.get(name) {
                return Some(*val);
            }
        }
        None
    }

    fn lookup_var_expr(&mut self, binder: Value) -> Result<Value, ParseError> {
        let var_action = TableAction::new(self.egraph, self.tables.var);
        let var_id = self
            .egraph
            .with_execution_state(|state| var_action.lookup(state, &[binder]));
        var_id.ok_or_else(|| ParseError::InvalidForm("var lookup failed".to_string()))
    }

    fn lookup_app_expr(&mut self, fun: Value, arg: Value) -> Result<Value, ParseError> {
        let app_action = TableAction::new(self.egraph, self.tables.app);
        let app_id = self
            .egraph
            .with_execution_state(|state| app_action.lookup(state, &[fun, arg]));
        app_id.ok_or_else(|| ParseError::InvalidForm("app lookup failed".to_string()))
    }
}
