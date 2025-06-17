use crate::ast::Command;

use super::*;
use dashmap::DashMap;
use egglog::{
    util::{IndexMap, IndexSet},
    EGraph, SerializeConfig,
};
use std::{collections::HashMap, path::PathBuf, sync::Mutex};

#[derive(Default)]
pub struct TxVT {
    egraph: Mutex<EGraph>,
    map: DashMap<Sym, WorkAreaNode>,
    /// used to store newly staged node among committed nodes (Not only the currently latest node but also nodes of old versions)
    staged_set_map: DashMap<Sym, Box<dyn EgglogNode>>,
    staged_new_map: Mutex<IndexMap<Sym, Box<dyn EgglogNode>>>,
    checkpoints: Mutex<Vec<CommitCheckPoint>>,
}

#[allow(unused)]
#[derive(Debug)]
pub struct CommitCheckPoint {
    committed_node_root: Sym,
    staged_set_nodes: Vec<Sym>,
    staged_new_nodes: Vec<Sym>,
}

pub enum TopoDirection {
    Up,
    Down,
}
/// Tx with version ctl feature
impl TxVT {
    pub fn to_dot(&self, file_name: PathBuf) {
        let egraph = self.egraph.lock().unwrap();
        let serialized = egraph.serialize(SerializeConfig::default());
        let dot_path = file_name.with_extension("dot");
        serialized
            .to_dot_file(dot_path.clone())
            .unwrap_or_else(|_| panic!("Failed to write dot file to {dot_path:?}"));
    }
    // collect all lastest ancestors of cur_sym, without cur_sym
    pub fn collect_latest_ancestors(&self, cur_sym: Sym, index_set: &mut IndexSet<Sym>) {
        let sym_node = self.map.get(&cur_sym).unwrap();
        let v = sym_node.preds.clone();
        drop(sym_node);
        for pred in v {
            // if pred has been accessed or it's not the lastest version
            if index_set.contains(&pred) || self.map.get(&pred).unwrap().next.is_some() {
                // do nothing
            } else {
                index_set.insert(pred.clone());
                self.collect_latest_ancestors(pred, index_set)
            }
        }
    }
    // collect all ancestors of cur_sym, without cur_sym
    pub fn collect_ancestors(&self, cur_sym: Sym, index_set: &mut IndexSet<Sym>) {
        let sym_node = self.map.get(&cur_sym).unwrap();
        let v = sym_node.preds.clone();
        drop(sym_node);
        for pred in v {
            // if pred has been accessed or it's not the lastest version
            if index_set.contains(&pred) {
                // do nothing
            } else {
                index_set.insert(pred.clone());
                self.collect_ancestors(pred, index_set)
            }
        }
    }
    // collect all strict descendants of cur_sym, without cur_sym
    pub fn collect_descendants(&self, cur_sym: Sym, index_set: &mut IndexSet<Sym>) {
        let succs = self
            .staged_set_map
            .get(&cur_sym)
            .map(|x| x.succs())
            .unwrap_or(self.map.get(&cur_sym).unwrap().succs());
        for succ in succs {
            if index_set.contains(&succ) || self.map.get(&succ).unwrap().next.is_some() {
                // do nothing this succ node has been accessed
            } else {
                index_set.insert(succ.clone());
                self.collect_descendants(succ, index_set)
            }
        }
    }
    /// topo all input nodes
    pub fn topo_sort(&self, index_set: &IndexSet<Sym>, direction: TopoDirection) -> Vec<Sym> {
        // init in degrees and out degrees
        let mut ins = Vec::new();
        let mut outs = Vec::new();
        ins.resize(index_set.len(), 0);
        outs.resize(index_set.len(), 0);
        for (i, (in_degree, out_degree)) in ins.iter_mut().zip(outs.iter_mut()).enumerate() {
            let sym = index_set[i];
            let node = self.map.get(&sym).unwrap();
            *in_degree = TxVT::degree_in_subgraph(node.preds().into_iter().map(|x| *x), index_set);
            *out_degree = TxVT::degree_in_subgraph(node.succs().into_iter(), index_set);
        }
        let (mut _ins, mut outs) = match direction {
            TopoDirection::Up => (ins, outs),
            TopoDirection::Down => (outs, ins),
        };
        let mut rst = Vec::new();
        let mut wait_for_release = Vec::new();
        // start node should not have any out edges in subgraph
        for (idx, _value) in outs.iter().enumerate() {
            if 0 == outs[idx] {
                wait_for_release.push(index_set[idx]);
            }
        }
        while !wait_for_release.is_empty() {
            let popped = wait_for_release.pop().unwrap();
            log::debug!(
                "popped is {} preds:{:?}",
                popped,
                &self.map.get(&popped).unwrap().preds
            );
            for target in &self.map.get(&popped).unwrap().preds {
                if let Some(idx) = index_set.get_index_of(target) {
                    outs[idx] -= 1;
                    if outs[idx] == 0 {
                        log::debug!("{} found to be 0", target);
                        wait_for_release.push(*target);
                    }
                }
            }
            rst.push(popped);
        }
        log::debug!("{:?}", rst);
        rst
    }
    /// calculate the edges in the subgraph
    pub fn degree_in_subgraph(nodes: impl Iterator<Item = Sym>, index_set: &IndexSet<Sym>) -> u32 {
        nodes.fold(0, |acc, item| {
            if index_set.contains(&item) {
                acc + 1
            } else {
                acc
            }
        })
    }
    pub fn new_with_type_defs(type_defs: Vec<Command>) -> Self {
        Self {
            egraph: Mutex::new({
                let mut e = EGraph::default();
                log::info!("{:?}", type_defs);
                e.run_program(type_defs).unwrap();
                e
            }),
            map: DashMap::default(),
            staged_set_map: DashMap::default(),
            staged_new_map: Mutex::new(IndexMap::default()),
            checkpoints: Mutex::new(vec![]),
        }
    }
    pub fn new() -> Self {
        Self::new_with_type_defs(collect_type_defs())
    }
    fn add_node(&self, mut node: WorkAreaNode, auto_latest: bool) {
        let sym = node.cur_sym();
        for node in node.succs_mut() {
            log::debug!("succ is {}", node);
            let latest = if auto_latest {
                &self.locate_latest(*node)
            } else {
                &*node
            };
            self.map
                .get_mut(node)
                .unwrap_or_else(|| panic!("node {} not found", latest.as_str()))
                .preds
                .push(sym);
            *node = *latest;
        }
        self.map.insert(node.cur_sym(), node);
    }

    /// update all ancestors recursively in guest and send updated term by egglog string repr to host
    /// when you update the node
    /// return all WorkAreaNodes created
    fn update_nodes(
        &self,
        root: Sym,
        staged_latest_syms_and_staged_nodes: Vec<(Sym, Box<dyn EgglogNode>)>,
    ) -> IndexSet<Sym> {
        // collect all ancestors that need copy
        let mut ancestors = IndexSet::default();
        for (latest_sym, _) in &staged_latest_syms_and_staged_nodes {
            log::debug!("collect ancestors of {:?}", latest_sym);
            // self.collect_latest_ancestors(*latest_sym, &mut latest_ancestors);
            self.collect_ancestors(*latest_sym, &mut ancestors);
        }
        let mut root_ancestors = IndexSet::default();
        self.collect_ancestors(root, &mut root_ancestors);
        if !root_ancestors.is_empty() {
            panic!("commit should be applied to root");
        }
        root_ancestors.insert(root);
        let mut root_descendants = IndexSet::default();
        self.collect_descendants(root, &mut root_descendants);
        root_descendants.insert(root);
        let intersection = IndexSet::from_iter(
            ancestors
                .intersection(&root_descendants)
                .cloned()
                .into_iter(),
        );
        let mut ancestors =
            IndexSet::from_iter(intersection.union(&root_ancestors).into_iter().cloned());
        let mut staged_latest_sym_map = IndexMap::default();
        // here we insert all staged_latest_sym because latest_ancestors do may not include all of them
        for (staged_latest_sym, staged_node) in staged_latest_syms_and_staged_nodes {
            ancestors.insert(staged_latest_sym);
            staged_latest_sym_map.insert(staged_latest_sym, staged_node);
        }

        // NB: ancestors set now contains all nodes that need to create
        log::trace!("all latest_ancestors {:?}", ancestors);

        let mut next_syms = IndexSet::default();
        for ancestor in ancestors {
            let mut latest_node = self.map.get_mut(&self.locate_latest(ancestor)).unwrap();
            let latest_sym = latest_node.cur_sym();
            let next_sym = latest_node.next_sym();
            let mut next_latest_node = latest_node.clone();
            // set prev, chain next latest version to latest version
            next_latest_node.prev = Some(latest_sym);
            // set next, chain latest version to next latest version
            latest_node.next = Some(next_sym);
            drop(latest_node);
            next_syms.insert(next_sym);
            if !staged_latest_sym_map.contains_key(&ancestor) {
                self.map.insert(next_sym, next_latest_node);
            } else {
                let mut staged_node = staged_latest_sym_map.get(&ancestor).unwrap().clone_dyn();
                *staged_node.cur_sym_mut() = next_sym;

                let mut staged_node = WorkAreaNode::new(staged_node);
                // set prev, chain next latest version to latest version
                staged_node.prev = Some(latest_sym);
                staged_node.preds = self.map.get(&ancestor).unwrap().preds.clone();
                self.map.insert(next_sym, staged_node);
            }
        }

        // update all preds
        let mut succ_preds_map = HashMap::new();
        for &next_sym in &next_syms {
            let sym_node = self.map.get(&next_sym).unwrap();
            for &sym in sym_node.preds() {
                let latest_sym = self.locate_latest(sym);
                if sym != latest_sym && !succ_preds_map.contains_key(&latest_sym) {
                    succ_preds_map.insert(sym, latest_sym);
                }
            }
            for sym in sym_node.succs() {
                let latest_sym = self.locate_latest(sym);
                if sym != latest_sym && !succ_preds_map.contains_key(&latest_sym) {
                    succ_preds_map.insert(sym, latest_sym);
                }
            }
        }
        log::debug!("preds 「map」to be {:?}", succ_preds_map);

        for &next_sym in &next_syms {
            let mut sym_node = self.map.get_mut(&next_sym).unwrap();
            for sym in sym_node.preds_mut() {
                if let Some(found) = succ_preds_map.get(sym) {
                    *sym = *found;
                }
            }
            for sym in sym_node.succs_mut() {
                if let Some(found) = succ_preds_map.get(sym) {
                    *sym = *found;
                }
            }
        }
        log::trace!("{:#?}", self.map);

        next_syms
    }
}

unsafe impl Send for TxVT {}
unsafe impl Sync for TxVT {}
impl VersionCtl for TxVT {
    /// locate the lastest version of the symbol
    fn locate_latest(&self, old: Sym) -> Sym {
        let map = &self.map;
        let mut cur = old;
        while let Some(newer) = map.get(&cur).unwrap().next {
            cur = newer;
        }
        cur
    }

    // locate next version
    fn locate_next(&self, node: Sym) -> Sym {
        let map = &self.map;
        let mut cur = node;
        if let Some(newer) = map.get(&cur).unwrap().next {
            cur = newer;
        } else {
            // do nothing because current version is the latest
        }
        cur
    }

    fn set_latest(&self, node: &mut Sym) {
        *node = self.locate_latest(*node);
    }

    fn set_next(&self, node: &mut Sym) {
        *node = self.locate_next(*node);
    }
    fn locate_prev(&self, node: Sym) -> Sym {
        let map = &self.map;
        let mut cur = node;
        if let Some(older) = map.get(&cur).unwrap().prev {
            cur = older;
        } else {
            // do nothing because current version is the oldest
        }
        cur
    }
    fn set_prev(&self, node: &mut Sym) {
        *node = self.locate_prev(*node);
    }
}

// MARK: Tx
impl Tx for TxVT {
    fn send(&self, received: TxCommand) {
        match received {
            TxCommand::StringCommand { command } => {
                log::info!("tx {}", command);
                let mut egraph = self.egraph.lock().unwrap();
                egraph.parse_and_run_program(None, command.as_str()).unwrap();
            }
            TxCommand::NativeCommand { command} => {
                let mut egraph = self.egraph.lock().unwrap();
                egraph.run_program(vec![command]).unwrap();
            },
        }
    }

    fn on_new(&self, node: &(impl EgglogNode + 'static)) {
        self.staged_new_map
            .lock()
            .unwrap()
            .insert(node.cur_sym(), node.clone_dyn());
    }

    fn on_set(&self, _node: &mut (impl EgglogNode + 'static)) {
        // do nothing, this operation has been delayed to commit
    }

    fn on_func_set<'a, F: EgglogFunc>(
        &self,
        input: <F::Input as EgglogFuncInputs>::Ref<'a>,
        output: <F::Output as EgglogFuncOutput>::Ref<'a>,
    ) {
        let input_nodes = input.as_nodes();
        let input_syms = input_nodes.iter().map(|x| x.cur_sym());
        let output = output.as_node().cur_sym();
        self.send(TxCommand::StringCommand {
            command: format!(
                "(set ({} {}) {} )",
                F::FUNC_NAME,
                input_syms.map(|x| x.as_str()).collect::<String>(),
                output
            ),
        });
    }
}

impl TxCommit for TxVT {
    /// commit behavior:
    /// 1. commit all descendants (if you also call set fn on subnodes they will also be committed)
    /// 2. commit basing the latest version of the working graph (working graph record all versions)
    /// 3. if TxCommit is implemented you can only change egraph by commit things. It's lazy.
    fn on_commit<T: EgglogNode>(&self, commit_root: &T) {
        let check_point = CommitCheckPoint {
            committed_node_root: commit_root.cur_sym(),
            staged_set_nodes: self.staged_set_map.iter().map(|a| *a.key()).collect(),
            staged_new_nodes: self
                .staged_new_map
                .lock()
                .unwrap()
                .iter()
                .map(|a| *a.0)
                .collect(),
        };
        log::debug!("{:?}", check_point);
        self.checkpoints.lock().unwrap().push(check_point);

        // process new nodes
        let mut news = self.staged_new_map.lock().unwrap();
        let mut backup_staged_new_syms = IndexSet::default();
        let len = news.len();
        for (new, new_node) in news.drain(0..len) {
            self.add_node(WorkAreaNode::new(new_node.clone_dyn()), false);
            backup_staged_new_syms.insert(new);
        }
        // send egglog string to egraph
        backup_staged_new_syms.into_iter().for_each(|sym| {
            self.send(TxCommand::NativeCommand {
                command: self.map.get(&sym).unwrap().egglog.to_egglog(),
            })
        });

        let all_staged = IndexSet::from_iter(self.staged_set_map.iter().map(|a| *a.key()));
        // // check any absent node
        // let mut panic_list = IndexSet::default();
        // for &sym in &all_staged{
        //     if !self.map.contains_key(&sym){
        //         panic_list.insert(sym);
        //     }
        // }
        // if panic_list.len()>0 {panic!("node {:?} not exist",panic_list )};

        let mut descendants = IndexSet::default();
        self.collect_descendants(commit_root.cur_sym(), &mut descendants);
        descendants.insert(commit_root.cur_sym());

        let staged_descendants_old = descendants.intersection(&all_staged).collect::<Vec<_>>();
        let staged_descendants_latest = staged_descendants_old
            .iter()
            .map(|x| self.locate_latest(**x))
            .collect::<Vec<_>>();

        let iter_impl = staged_descendants_latest.iter().cloned().zip(
            staged_descendants_old
                .iter()
                .map(|x| self.staged_set_map.remove(*x).unwrap().1),
        );
        let created = self.update_nodes(commit_root.cur_sym(), iter_impl.collect());
        log::debug!("created {:#?}", created);

        log::debug!("nodes to topo:{:?}", created);
        self.topo_sort(&created, TopoDirection::Up)
            .into_iter()
            .for_each(|sym| {
                self.send(TxCommand::NativeCommand {
                    command: self.map.get(&sym).unwrap().egglog.to_egglog(),
                })
            });
    }

    fn on_stage<T: EgglogNode + ?Sized>(&self, node: &T) {
        self.staged_set_map.insert(node.cur_sym(), node.clone_dyn());
    }
}
