use crate::{
    extract::Node,
    function::ValueVec,
    termdag::{Term, TermDag},
    util::HashMap,
    EGraph, Symbol, UnionFind, Value,
};

struct ProofChecker {
    proven_equal: UnionFind,
    dag: TermDag,
    to_check: Vec<Term>,
}

const EQ_GRAPH_NAME: &str = "EqGraph__";

pub fn check_proof(egraph: &mut EGraph) {
    let mut proven_equal = UnionFind::default();
    for _i in 0..egraph.unionfind.num_ids() {
        proven_equal.make_set();
    }

    let mut termdag = TermDag::default();
}
/*let to_check = egraph.extract_variants(

    ProofChecker { proven_equal }.check();
}

impl ProofChecker {
    fn check(&mut self) {
        let eq_graph_func = self
            .egraph
            .functions
            .get::<Symbol>(&EQ_GRAPH_NAME.into())
            .unwrap();
        let mut proofs = eq_graph_func
            .nodes
            .iter()
            .map(|(_k, output)| output.value)
            .collect::<Vec<_>>();
        proofs.sort_by_key(|proof| self.get_proof_with_age_age(*proof));

        // sorted ascending by age, so that DemandEq
        // edges should already be proven
        for proof_with_age_id in proofs {
            let proof_with_age = self.get(proof_with_age_id);
            //let proof = self.get(proof_with_age.inputs[0]);
            //self.check_proof(&proof);
        }
    }

    fn get_proof_with_age_age(&mut self, value: Value) -> i64 {
        let sort = self.egraph.get_sort(&value).unwrap();
        assert!(sort.name() == "ProofWithAge__".into());
        assert!(sort.is_eq_sort());
        let node = self.get(value);
        assert!(node.sym == "MakeProofWithAge__".into());
        assert!(node.inputs.len() == 2);
        let age = node.inputs[1];
        assert!(age.tag == "i64".into());
        return age.bits as i64;
    }

    // get the only enode from this eclass
    fn get(&mut self, value: Value) -> Node {
        let nodes = self.value_to_nodes.get(&value).unwrap();
        println!("{}", self.egraph.clone().extract(value).1);
        assert_eq!(nodes.len(), 1);
        nodes[0].clone()
    }
}
*/
