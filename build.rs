use std::env;
use std::io::Write;

fn main() {
    println!("cargo:rerun-if-changed=src/ast/parse.lalrpop");
    lalrpop::process_root().unwrap();
    let mdir = std::env!("CARGO_MANIFEST_DIR");
    generate_tests(&format!("{mdir}/tests/**/*.egg"));
}

fn generate_tests(glob: &str) {
    let out_dir = env::var("OUT_DIR").unwrap();
    let mut out = std::fs::File::create(format!("{out_dir}/files.rs")).unwrap();
    for f in glob::glob(glob).unwrap() {
        let f = f.unwrap();
        let name = f
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .replace(['.', '-', ' '], "_");
        // write a normal test
        writeln!(out, "#[test] fn {name}() {{ run({:?}, false); }}", f).unwrap();

        // write a test with proofs enabled
        // TODO: re-enable herbie, unsound, and eqsolve when proof extraction is faster
        if !(name == "herbie" || name == "repro_unsound" || name == "eqsolve") {
            writeln!(
                out,
                "#[test] fn {name}_with_proofs() {{ run({:?}, true); }}",
                f
            )
            .unwrap();
        }
    }
}
