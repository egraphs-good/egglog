use super::*;
use crate::{ast::Command, Function};
use core_relations::Value;
use dashmap::DashMap;
use egglog::{
    util::{IndexMap, IndexSet},
    EGraph, SerializeConfig,
};
use petgraph::{
    dot::{Config, Dot},
    prelude::{StableDiGraph, StableGraph},
    EdgeType,
};
use std::{collections::HashMap, fs::File, io::Write, path::PathBuf, sync::Mutex};

pub struct TxRxVT {
    pub egraph: Mutex<EGraph>,
    pub map: DashMap<Sym, WorkAreaNode>,
    /// used to store newly staged node among committed nodes (Not only the currently latest node but also nodes of old versions)
    pub staged_set_map: DashMap<Sym, Box<dyn EgglogNode>>,
    pub staged_new_map: Mutex<IndexMap<Sym, Box<dyn EgglogNode>>>,
    checkpoints: Mutex<Vec<CommitCheckPoint>>,
    registry: EgglogTypeRegistry,
}

#[allow(unused)]
#[derive(Debug)]
pub struct CommitCheckPoint {
    committed_node_root: Sym,
    staged_set_nodes: Vec<Sym>,
    staged_new_nodes: Vec<Sym>,
}

/// Tx with version ctl feature
impl TxRxVT {
    pub fn egraph_to_dot(&self, file_name: PathBuf) {
        let egraph = self.egraph.lock().unwrap();
        let serialized = egraph.serialize(SerializeConfig::default());
        let dot_path = file_name;
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
            *in_degree =
                TxRxVT::degree_in_subgraph(node.preds().into_iter().map(|x| *x), index_set);
            *out_degree = TxRxVT::degree_in_subgraph(node.succs().into_iter(), index_set);
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
    pub fn new() -> Self {
        Self {
            egraph: Mutex::new({
                let mut e = EGraph::default();
                let type_defs = EgglogTypeRegistry::collect_type_defs();
                log::trace!("{:?}", type_defs);
                e.run_program(type_defs).unwrap();
                e
            }),
            registry: EgglogTypeRegistry::new_with_inventory(),
            map: DashMap::new(),
            staged_set_map: DashMap::new(),
            staged_new_map: Mutex::new(IndexMap::default()),
            checkpoints: Mutex::new(vec![]),
        }
    }
    pub fn pack_actions(actions: Vec<EgglogAction>) -> Vec<Command> {
        let mut v = vec![];
        for egglog_action in actions {
            v.push(Command::Action(egglog_action))
        }
        v
        // static COUNTER: OnceLock<AtomicU32> = OnceLock::new();
        // let counter = COUNTER.get_or_init(|| AtomicU32::new(0));
        // let rule_set_name = format!(
        //     "anonymous_rule_set_{}",
        //     counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        // );
        // let new_rule_set = Command::AddRuleset(span!(), rule_set_name.clone());
        // let rule = GenericRule {
        //     span: span!(),
        //     head: GenericActions::new(actions),
        //     body: vec![],
        // };
        // let action_rule_set = Command::Rule {
        //     name: format!(""),
        //     ruleset: rule_set_name.clone(),
        //     rule,
        // };
        // let run = Command::RunSchedule(GenericSchedule::Run(
        //     span!(),
        //     GenericRunConfig {
        //         ruleset: rule_set_name.clone(),
        //         until: None,
        //     },
        // ));
        // vec![new_rule_set, action_rule_set, run]
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
        log::debug!("map insert {:?}", node.egglog);
        if let Some(node) = self.map.insert(node.cur_sym(), node) {
            panic!("repeat insertion of node {:?}", node);
        }
    }
    /// update all ancestors recursively in guest and send updated term by egglog native command to host
    /// when you update the node
    /// return all WorkAreaNodes created
    fn update_nodes(
        &self,
        root: Sym,
        staged_latest_syms_and_staged_nodes: Vec<(Sym, Box<dyn EgglogNode>)>,
    ) -> IndexSet<Sym> {
        if staged_latest_syms_and_staged_nodes.len() == 0 {
            return IndexSet::default();
        }

        log::debug!("update_nodes:{:#?}", self.map);
        // collect all ancestors that need copy
        let mut ancestors = IndexSet::default();
        for (latest_sym, _) in &staged_latest_syms_and_staged_nodes {
            log::debug!("collect ancestors of {:?}", latest_sym);
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
        log::debug!("all latest_ancestors {:?}", ancestors);

        let mut next_syms = IndexSet::default();
        for ancestor in ancestors {
            let mut latest_node = self.map.get_mut(&self.locate_latest(ancestor)).unwrap();
            let latest_sym = latest_node.cur_sym();
            let mut next_latest_node = latest_node.clone();
            let next_sym = next_latest_node.roll_sym();

            // set next, chain latest version to next latest version
            latest_node.next = Some(next_sym);
            drop(latest_node);

            // set prev, chain next latest version to latest version
            next_latest_node.prev = Some(latest_sym);

            next_syms.insert(next_sym);
            if !staged_latest_sym_map.contains_key(&ancestor) {
                log::debug!("map insert {},{:?}", next_sym, next_latest_node);
                if let Some(node) = self.map.insert(next_sym, next_latest_node) {
                    panic!("repeat insertion of node {:?}", node);
                }
            } else {
                let mut staged_node = staged_latest_sym_map.get(&ancestor).unwrap().clone_dyn();
                *staged_node.cur_sym_mut() = next_sym;

                let mut staged_node = WorkAreaNode::new(staged_node);
                // set prev, chain next latest version to latest version
                staged_node.prev = Some(latest_sym);
                staged_node.preds = self.map.get(&ancestor).unwrap().preds.clone();

                log::debug!("map insert {},{:?}", next_sym, staged_node);
                if let Some(node) = self.map.insert(next_sym, staged_node) {
                    panic!("repeat insertion of node {:?}", node);
                }
            }
        }
        log::debug!("mid update_nodes:{:#?}", self.map);

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
        log::debug!("after update_nodes:{:#?}", self.map);
        next_syms
    }
    /// transform WorkAreaGraph into dot file
    pub fn wag_to_dot(&self, name: String) {
        pub fn generate_dot_by_graph<N: std::fmt::Debug, E: std::fmt::Debug, Ty: EdgeType>(
            g: &StableGraph<N, E, Ty>,
            name: String,
            graph_config: &[Config],
        ) {
            let dot_name = name.clone();
            let mut f = File::create(dot_name.clone()).unwrap();
            let dot_string = format!("{:?}", Dot::with_config(&g, &graph_config));
            f.write_all(dot_string.as_bytes()).expect("写入失败");
        }
        let g = self.build_petgraph();
        generate_dot_by_graph(&g, name, &[]);
    }
    pub fn build_petgraph(&self) -> StableDiGraph<WorkAreaNode, ()> {
        // 1. 收集所有节点
        let v = self
            .map
            .iter()
            .map(|x| x.value().clone())
            .collect::<Vec<_>>();
        let mut g = StableDiGraph::new();
        let mut idxs = Vec::new();
        // 2. 建立 WorkAreaNode 的 cur_sym 到 petgraph::NodeIndex 的映射
        use std::collections::HashMap;
        let mut sym2idx = HashMap::new();
        log::debug!("map:{:?}", self.map);
        for node in &v {
            let idx = g.add_node(node.clone());
            idxs.push(idx);
            sym2idx.insert(node.egglog.cur_sym(), idx);
            log::debug!("sym2idx insert {}", node.egglog.cur_sym());
        }
        // 3. 添加边（succs）
        for node in &v {
            let from = node.egglog.cur_sym();
            let from_idx = sym2idx[&from];
            log::debug!("succs of {} is {:?}", from, node.egglog.succs());
            for to in node.egglog.succs() {
                if let Some(&to_idx) = sym2idx.get(&to) {
                    g.add_edge(from_idx, to_idx, ());
                } else {
                    panic!("{} not found in wag", to)
                }
            }
        }
        g
    }
}

unsafe impl Send for TxRxVT {}
unsafe impl Sync for TxRxVT {}
impl VersionCtl for TxRxVT {
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
impl Tx for TxRxVT {
    fn send(&self, transmitted: TxCommand) {
        let mut egraph = self.egraph.lock().unwrap();
        match transmitted {
            TxCommand::StringCommand { command } => {
                log::info!("{}", command);
                egraph.parse_and_run_program(None, &command).unwrap();
            }
            TxCommand::NativeCommand { command } => {
                log::info!("{}", command.to_string());
                egraph.run_program(vec![command]).unwrap();
            }
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

    #[track_caller]
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

    fn on_union(&self, node1: &(impl EgglogNode + 'static), node2: &(impl EgglogNode + 'static)) {
        self.send(TxCommand::StringCommand {
            command: format!("(union {} {})", node1.cur_sym(), node2.cur_sym()),
        });
    }
}

impl TxCommit for TxRxVT {
    /// commit behavior:
    /// 1. commit all descendants (if you also call set fn on subnodes they will also be committed)
    /// 2. commit basing the latest version of the working graph (working graph record all versions)
    /// 3. if TxCommit is implemented you can change egraph by `commit` rather than `set`. It's lazy because it uses a buffer to store all `staged set`.
    /// 4. if you didn't stage `set` on nodes, it will do nothing on commited node only flush all staged_new_node buffer
    fn on_commit<T: EgglogNode>(&self, commit_root: &T) {
        log::debug!("on_commit {:?}", commit_root.to_egglog_string());
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
        log::debug!("staged_set_map:{:?}", self.staged_set_map);
        log::debug!("staged_new_map:{:?}", self.staged_new_map.lock().unwrap());
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
        let actions = backup_staged_new_syms
            .into_iter()
            .map(|sym| self.map.get(&sym).unwrap().egglog.to_egglog())
            .collect::<Vec<_>>();
        let commands = Self::pack_actions(actions);
        for command in commands {
            self.send(TxCommand::NativeCommand { command });
        }

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
        log::trace!("created {:#?}", created);

        log::trace!("nodes to topo:{:?}", created);
        let actions = self
            .topo_sort(&created, TopoDirection::Up)
            .into_iter()
            .map(|sym| self.map.get(&sym).unwrap().egglog.to_egglog())
            .collect::<Vec<_>>();
        for command in Self::pack_actions(actions) {
            self.send(TxCommand::NativeCommand { command })
        }
    }

    fn on_stage<T: EgglogNode + ?Sized>(&self, node: &T) {
        self.staged_set_map.insert(node.cur_sym(), node.clone_dyn());
    }
}

// MARK: Rx
impl Rx for TxRxVT {
    fn on_func_get<'a, 'b, F: EgglogFunc>(
        &self,
        input: <F::Input as EgglogFuncInputs>::Ref<'a>,
    ) -> F::Output {
        let input_nodes = input.as_nodes();
        let output = {
            let egraph = &self.egraph.lock().unwrap();
            input_nodes.iter().for_each(|x| {
                println!(
                    "input:({},{:?})",
                    x.cur_sym(),
                    get_value(egraph, x.cur_sym().into())
                )
            });
            let output = get_value(egraph, F::FUNC_NAME);
            output
        };
        let sym = self.on_pull_value::<F::OutputTy>(output);
        let node = &self.map.get(&sym).unwrap().egglog;
        let output: &F::Output =
            unsafe { &*(node.as_ref() as *const dyn EgglogNode as *const F::Output) };
        output.clone()
    }

    fn on_funcs_get<'a, 'b, F: EgglogFunc>(
        &self,
        _max_size: Option<usize>,
    ) -> Vec<(
        <F::Input as EgglogFuncInputs>::Ref<'b>,
        <F::Output as EgglogFuncOutput>::Ref<'b>,
    )> {
        todo!()
    }
    fn on_pull_value<T: EgglogTy>(&self, value: Value) -> Sym {
        log::debug!("pulling value {:?}", value);
        let egraph = self.egraph.lock().unwrap();
        let sort = egraph.get_sort_by_name(T::TY_NAME).unwrap();
        let mut term2sym = HashMap::new();
        let (term_dag, start_term, cost) = egraph.extract_value(sort, value).unwrap();

        let root_idx = term_dag.lookup(&start_term);
        let mut ret_sym = None;

        let topo = topo_sort(&term_dag);
        for &i in &topo {
            let new_fn = self
                .registry
                .get_fn(i, &term_dag)
                .unwrap_or_else(|| panic!("didn't found fn of term {:?}", term_dag.get(i)));
            let boxed_node = new_fn(i, &term_dag, &mut term2sym);
            if i == root_idx {
                ret_sym = Some(boxed_node.cur_sym())
            }
            self.add_node(WorkAreaNode::new(boxed_node), false);
        }
        log::debug!(
            "term:{:?}, term_dag:{:?}, cost:{}",
            start_term,
            term_dag,
            cost
        );
        ret_sym.unwrap()
    }
    fn on_pull_sym<T: EgglogTy>(&self, sym: Sym) -> Sym {
        let value = get_value(&self.egraph.lock().unwrap(), sym.into());
        self.on_pull_value::<T>(value)
    }
}
fn get_function(egraph: &EGraph, name: &str) -> Function {
    egraph.functions.get(name).unwrap().clone()
}
fn get_value(egraph: &EGraph, name: &str) -> Value {
    log::trace!("get_value_of_func {}", name);
    let mut out = None;
    let id = get_function(egraph, name).backend_id;
    egraph.backend.for_each(id, |row| out = Some(row.vals[0]));
    out.unwrap_or_else(|| panic!("do not have any output"))
}

impl std::fmt::Debug for Box<dyn EgglogNode> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{},{}", self.cur_sym(), self.to_egglog_string())
    }
}
