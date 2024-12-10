use crate::core::{
    GenericCoreAction, GenericCoreActions, ResolvedAtomTerm, ResolvedCoreActions,
    SpecializedPrimitive,
};
use crate::{typechecking::FuncType, *};
use typechecking::TypeError;

use crate::{ast::Literal, core::ResolvedCall, ExtractReport, Value};

struct ActionCompiler<'a> {
    types: &'a IndexMap<Symbol, ArcSort>,
    locals: IndexSet<ResolvedVar>,
    instructions: Vec<Instruction>,
}

impl<'a> ActionCompiler<'a> {
    fn compile_action(&mut self, action: &GenericCoreAction<ResolvedCall, ResolvedVar>) {
        match action {
            GenericCoreAction::Let(_ann, v, f, args) => {
                self.do_call(f, args);
                self.locals.insert(v.clone());
            }
            GenericCoreAction::LetAtomTerm(_ann, v, at) => {
                self.do_atom_term(at);
                self.locals.insert(v.clone());
            }
            GenericCoreAction::Extract(_ann, e, b) => {
                let sort = self.do_atom_term(e);
                self.do_atom_term(b);
                self.instructions.push(Instruction::Extract(2, sort));
            }
            GenericCoreAction::Set(_ann, f, args, e) => {
                let ResolvedCall::Func(func) = f else {
                    panic!("Cannot set primitive- should have been caught by typechecking!!!")
                };
                for arg in args {
                    self.do_atom_term(arg);
                }
                self.do_atom_term(e);
                self.instructions.push(Instruction::Set(func.name));
            }
            GenericCoreAction::Change(_ann, change, f, args) => {
                let ResolvedCall::Func(func) = f else {
                    panic!("Cannot change primitive- should have been caught by typechecking!!!")
                };
                for arg in args {
                    self.do_atom_term(arg);
                }
                self.instructions
                    .push(Instruction::Change(*change, func.name));
            }
            GenericCoreAction::Union(_ann, arg1, arg2) => {
                let sort = self.do_atom_term(arg1);
                self.do_atom_term(arg2);
                self.instructions.push(Instruction::Union(2, sort));
            }
            GenericCoreAction::Panic(_ann, msg) => {
                self.instructions.push(Instruction::Panic(msg.clone()));
            }
        }
    }

    fn do_call(&mut self, f: &ResolvedCall, args: &[ResolvedAtomTerm]) {
        for arg in args {
            self.do_atom_term(arg);
        }
        match f {
            ResolvedCall::Func(f) => self.do_function(f),
            ResolvedCall::Primitive(p) => self.do_prim(p),
        }
    }

    fn do_atom_term(&mut self, at: &ResolvedAtomTerm) -> ArcSort {
        match at {
            ResolvedAtomTerm::Var(_ann, var) => {
                if let Some((i, ty)) = self.locals.get_full(var) {
                    self.instructions.push(Instruction::Load(Load::Stack(i)));
                    ty.sort.clone()
                } else {
                    let (i, _, ty) = self.types.get_full(&var.name).unwrap();
                    self.instructions.push(Instruction::Load(Load::Subst(i)));
                    ty.clone()
                }
            }
            ResolvedAtomTerm::Literal(_ann, lit) => {
                self.instructions.push(Instruction::Literal(lit.clone()));
                crate::sort::literal_sort(lit)
            }
            ResolvedAtomTerm::Global(_ann, _var) => {
                panic!("Global variables should have been desugared");
            }
        }
    }

    fn do_function(&mut self, func_type: &FuncType) {
        self.instructions
            .push(Instruction::CallFunction(func_type.name));
    }

    fn do_prim(&mut self, prim: &SpecializedPrimitive) {
        self.instructions
            .push(Instruction::CallPrimitive(prim.clone(), prim.input.len()));
    }
}

#[derive(Clone, Debug)]
enum Load {
    Stack(usize),
    Subst(usize),
}

/// The instruction set for the action VM.
#[derive(Clone, Debug)]
enum Instruction {
    /// Push a literal onto the stack.
    Literal(Literal),
    /// Push a value from the stack or the substitution onto the stack.
    Load(Load),
    /// Pop function arguments off the stack, calls the function,
    /// and push the result onto the stack.
    CallFunction(Symbol),
    /// Pop primitive arguments off the stack, calls the primitive,
    /// and push the result onto the stack.
    CallPrimitive(SpecializedPrimitive, usize),
    /// Pop function arguments off the stack and either deletes or subsumes the corresponding row
    /// in the function.
    Change(Change, Symbol),
    /// Pop the value to be set and the function arguments off the stack.
    /// Set the function at the given arguments to the new value.
    Set(Symbol),
    /// Union the last `n` values on the stack.
    Union(usize, ArcSort),
    /// Extract the best expression. `n` is always 2.
    /// The first value on the stack is the expression to extract,
    /// and the second value is the number of variants to extract.
    Extract(usize, ArcSort),
    /// Panic with the given message.
    Panic(String),
}

#[derive(Clone, Debug)]
pub struct Program(Vec<Instruction>);

impl EGraph {
    /// Takes `binding`, which is a set of variables bound during matching
    /// whose positions are captured by indices of the `IndexSet``, and a list of core actions.
    /// Returns a program compiled from core actions and a list of variables bound to `stack`
    /// (whose positions are described by IndexSet indices as well).
    pub(crate) fn compile_actions(
        &self,
        binding: &IndexSet<ResolvedVar>,
        actions: &GenericCoreActions<ResolvedCall, ResolvedVar>,
    ) -> Result<Program, Vec<TypeError>> {
        // TODO: delete types and just keep the ordering
        let mut types = IndexMap::default();
        for var in binding {
            types.insert(var.name, var.sort.clone());
        }
        let mut compiler = ActionCompiler {
            types: &types,
            locals: IndexSet::default(),
            instructions: Vec::new(),
        };

        for a in &actions.0 {
            compiler.compile_action(a);
        }

        Ok(Program(compiler.instructions))
    }

    // This is the ugly part. GenericCoreActions lowered from
    // expressions like `2` is an empty vector, because no action is taken.
    // So to explicitly obtain the return value of an expression, compile_expr
    // needs to also take a `target`.`
    pub(crate) fn compile_expr(
        &self,
        binding: &IndexSet<ResolvedVar>,
        actions: &ResolvedCoreActions,
        target: &ResolvedAtomTerm,
    ) -> Result<Program, Vec<TypeError>> {
        // TODO: delete types and just keep the ordering
        let mut types = IndexMap::default();
        for var in binding {
            types.insert(var.name, var.sort.clone());
        }
        let mut compiler = ActionCompiler {
            types: &types,
            locals: IndexSet::default(),
            instructions: Vec::new(),
        };

        for a in actions.0.iter() {
            compiler.compile_action(a);
        }
        compiler.do_atom_term(target);

        Ok(Program(compiler.instructions))
    }

    fn perform_set(
        &mut self,
        table: Symbol,
        new_value: Value,
        stack: &mut [Value],
    ) -> Result<(), Error> {
        let function = self.functions.get_mut(&table).unwrap();

        let new_len = stack.len() - function.schema.input.len();
        let args = &stack[new_len..];

        // We should only have canonical values here: omit the canonicalization step
        let old_value = function.get(args);

        if let Some(old_value) = old_value {
            if new_value != old_value {
                let merged: Value = match function.merge.clone() {
                    MergeFn::AssertEq => {
                        return Err(Error::MergeError(table, new_value, old_value));
                    }
                    MergeFn::Union => self.unionfind.union_values(
                        old_value,
                        new_value,
                        function.decl.schema.output,
                    ),
                    MergeFn::Expr(merge_prog) => {
                        let values = [old_value, new_value];
                        let mut stack = vec![];
                        self.run_actions(&mut stack, &values, &merge_prog)?;
                        stack.pop().unwrap()
                    }
                };
                if merged != old_value {
                    let args = &stack[new_len..];
                    let function = self.functions.get_mut(&table).unwrap();
                    function.insert(args, merged, self.timestamp);
                }
            }
        } else {
            function.insert(args, new_value, self.timestamp);
        }
        Ok(())
    }

    pub(crate) fn run_actions(
        &mut self,
        stack: &mut Vec<Value>,
        subst: &[Value],
        program: &Program,
    ) -> Result<(), Error> {
        for instr in &program.0 {
            match instr {
                Instruction::Load(load) => match load {
                    Load::Stack(idx) => stack.push(stack[*idx]),
                    Load::Subst(idx) => stack.push(subst[*idx]),
                },
                Instruction::CallFunction(f) => {
                    let function = self.functions.get_mut(f).unwrap();
                    let new_len = stack.len() - function.schema.input.len();
                    let values = &stack[new_len..];

                    #[cfg(debug_assertions)]
                    let output_tag = function.schema.output.name();

                    #[cfg(debug_assertions)]
                    for (ty, val) in function.schema.input.iter().zip(values) {
                        assert_eq!(ty.name(), val.tag);
                    }

                    let value = if let Some(out) = function.nodes.get(values) {
                        out.value
                    } else {
                        let ts = self.timestamp;
                        if function.decl.subtype == FunctionSubtype::Relation {
                            function.insert(values, Value::unit(), ts);
                            Value::unit()
                        } else if function.decl.subtype == FunctionSubtype::Constructor {
                            let value = Value {
                                #[cfg(debug_assertions)]
                                tag: function.schema.output.name(),
                                bits: self.unionfind.make_set(),
                            };
                            function.insert(values, value, ts);
                            value
                        } else {
                            return Err(Error::NotFoundError(NotFoundError(format!(
                                "No value found for {f} {:?}",
                                values
                            ))));
                        }
                    };

                    // cfg is necessary because debug_assert_eq still evaluates its
                    // arguments in release mode (is has to because of side effects)
                    #[cfg(debug_assertions)]
                    debug_assert_eq!(output_tag, value.tag);

                    stack.truncate(new_len);
                    stack.push(value);
                }
                Instruction::CallPrimitive(p, arity) => {
                    let new_len = stack.len() - arity;
                    let values = &stack[new_len..];
                    if let Some(value) =
                        p.primitive.apply(values, (&p.input, &p.output), Some(self))
                    {
                        stack.truncate(new_len);
                        stack.push(value);
                    } else {
                        return Err(Error::PrimitiveError(p.primitive.clone(), values.to_vec()));
                    }
                }
                Instruction::Set(f) => {
                    let function = self.functions.get_mut(f).unwrap();
                    // desugaring should have desugared
                    // set to union
                    let new_value = stack.pop().unwrap();
                    let new_len = stack.len() - function.schema.input.len();

                    self.perform_set(*f, new_value, stack)?;
                    stack.truncate(new_len)
                }
                Instruction::Union(arity, sort) => {
                    let new_len = stack.len() - arity;
                    let values = &stack[new_len..];
                    let first = self.unionfind.find(values[0].bits);
                    values[1..].iter().fold(first, |a, b| {
                        let b = self.unionfind.find(b.bits);
                        self.unionfind.union(a, b, sort.name())
                    });
                    stack.truncate(new_len);
                }
                Instruction::Extract(arity, sort) => {
                    let new_len = stack.len() - arity;
                    let values = &stack[new_len..];
                    let new_len = stack.len() - arity;
                    let mut termdag = TermDag::default();

                    let variants = values[1].bits as i64;
                    if variants == 0 {
                        let (cost, term) = self.extract(values[0], &mut termdag, sort);
                        let extracted = termdag.to_string(&term);
                        log::info!("extracted with cost {cost}: {extracted}");
                        self.print_msg(extracted);
                        self.extract_report = Some(ExtractReport::Best {
                            termdag,
                            cost,
                            term,
                        });
                    } else {
                        if variants < 0 {
                            panic!("Cannot extract negative number of variants");
                        }
                        let terms =
                            self.extract_variants(sort, values[0], variants as usize, &mut termdag);
                        log::info!("extracted variants:");
                        let mut msg = String::default();
                        msg += "(\n";
                        assert!(!terms.is_empty());
                        for expr in &terms {
                            let str = termdag.to_string(expr);
                            log::info!("   {str}");
                            msg += &format!("   {str}\n");
                        }
                        msg += ")";
                        self.print_msg(msg);
                        self.extract_report = Some(ExtractReport::Variants { termdag, terms });
                    }

                    stack.truncate(new_len);
                }
                Instruction::Panic(msg) => panic!("Panic: {msg}"),
                Instruction::Literal(lit) => match lit {
                    Literal::Int(i) => stack.push(Value::from(*i)),
                    Literal::Float(f) => stack.push(Value::from(*f)),
                    Literal::String(s) => stack.push(Value::from(*s)),
                    Literal::Bool(b) => stack.push(Value::from(*b)),
                    Literal::Unit => stack.push(Value::unit()),
                },
                Instruction::Change(change, f) => {
                    let function = self.functions.get_mut(f).unwrap();
                    let new_len = stack.len() - function.schema.input.len();
                    let args = &stack[new_len..];
                    match change {
                        Change::Delete => {
                            function.remove(args, self.timestamp);
                        }
                        Change::Subsume => {
                            if function.decl.merge.is_some() {
                                return Err(Error::SubsumeMergeError(*f));
                            }
                            function.subsume(args);
                        }
                    }
                    stack.truncate(new_len);
                }
            }
        }
        Ok(())
    }
}
