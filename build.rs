use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/ast/parse.lalrpop");
    lalrpop::process_root().unwrap();
    let mdir = std::env!("CARGO_MANIFEST_DIR");

    let out_dir = env::var("OUT_DIR").unwrap();
    let test_file = &mut File::create(format!("{out_dir}/files.rs")).unwrap();
    generate_tests(test_file, &format!("{mdir}/tests/**/*.egg"));
}

fn generate_tests(file: &mut File, glob: &str) {
    let paths: Vec<PathBuf> = glob::glob(glob)
        .unwrap()
        .collect::<Result<_, glob::GlobError>>()
        .unwrap();

    writeln!(file, "const N_TEST_FILES: usize = {};", paths.len()).unwrap();

    for f in paths {
        let name = f
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .replace(['.', '-', ' '], "_");

        let should_fail = f.to_string_lossy().contains("fail-typecheck");

        // write a normal test
        writeln!(
            file,
            r#" #[test] 
            fn {name}() {{ 
                Run {{
                    path: {f:?},
                    should_fail: {should_fail},
                    test_proofs: false,
                }}.run(); 
            }}"#,
        )
        .unwrap();

        // write a test with proofs enabled
        // TODO: re-enable herbie, unsound, and eqsolve when proof extraction is faster
        writeln!(
            file,
            r#" #[test] 
            fn {name}_with_proofs() {{ 
                Run {{
                    path: {f:?},
                    should_fail: {should_fail},
                    test_proofs: true,
                }}.run(); 
            }}"#,
        )
        .unwrap();
    }
}
