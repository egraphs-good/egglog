use crate::back::index::{IdxName, Trie};
use crate::*;

#[derive(PartialEq, Eq, Clone, Copy)]
pub struct IdxId(usize);
pub struct ValId(usize);

pub enum Instr {
    Load(IdxName),
    Intersect(Vec<IdxId>),
    Yield(Symbol, Vec<ValId>),
    // todo: make symbol a distinct structure FuncName
    Op(Symbol, Vec<ValId>),
}

pub struct Program(Vec<Instr>);

pub struct Machine<'a> {
    idx_buffer: Vec<&'a Trie>,
    val_buffer: Vec<DenseValue>,
    to_yield: HashMap<Symbol, Vec<DenseValue>>,
    arg_buffer: Vec<Value>,
}

impl<'a> Machine<'a> {
    pub fn run(&mut self, egraph: &'a mut EGraph, program: &Program) {
        self.idx_buffer.clear();
        self.val_buffer.clear();
        self.arg_buffer.clear();
        self.run_with_instrs(egraph, &program.0)
    }
    pub fn run_with_instrs(&mut self, egraph: &'a mut EGraph, instrs: &[Instr]) {
        if instrs.len() == 0 {
            return;
        }
        let instr = &instrs[0];
        let remaining_instrs = &instrs[1..];
        match instr {
            Instr::Load(idx_name) => {
                self.idx_buffer.push(egraph.get_index_mut(idx_name));
            }
            Instr::Intersect(idxs) => {
                let i_min = idxs
                    .iter()
                    .min_by_key(|i| self.idx_buffer[i.0].len())
                    .unwrap();
                let intersected = self.idx_buffer[i_min.0].0.keys().filter(|t| {
                    idxs.iter()
                        .all(|i| i == i_min || self.idx_buffer[i.0].0.contains_key(t))
                });
                self.val_buffer.push(DenseValue::null());
                let fake_trie = Trie::default();
                let n = self.idx_buffer.len();
                self.idx_buffer.extend((1..idxs.len()).map(|_| &fake_trie));

                for val in intersected {
                    *self.val_buffer.last_mut().unwrap() = *val;
                    for (i, idx) in idxs.iter().enumerate() {
                        self.idx_buffer[n + i] = self.idx_buffer[idx.0].0.get(val).unwrap();
                    }
                    self.run_with_instrs(egraph, remaining_instrs);
                }

                self.val_buffer.pop();
                self.idx_buffer.truncate(n);
            }
            Instr::Yield(rel, vals) => {
                self.to_yield
                    .get_mut(rel)
                    .expect("Relation does not found!")
                    .extend(vals.iter().map(|val_id| self.val_buffer[val_id.0]));
            }
            Instr::Op(op, vals) => {
                self.arg_buffer.extend(
                    vals.iter()
                        .map(|val_id| self.val_buffer[val_id.0])
                        .map(|dense_val| egraph.dict_encoding.get_by_right(&dense_val).unwrap())
                        .cloned(),
                );
                // todo: primitives should be recognized by unique identifier
                // rework primitives
                let val = egraph.primitives[op][0].apply(&self.arg_buffer);
                let dense_val = todo!(); // egraph.dict_encoding.entry()
                self.val_buffer.push(dense_val);
                self.val_buffer.clear();
            }
        }
    }
}
