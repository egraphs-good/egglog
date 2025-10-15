use egglog::*;
use std::env;
use std::ffi::OsStr; // Add this import for file extension checking
use std::fs;
use std::path::{Path, PathBuf};

/// Run an egglog (.egg) file in a TimedEgraph and output the produced timeline to a JSON file
///
/// # Arguments
///
/// * `egg_file_path` - Path to the .egg file to run
/// * `json_output_path` - Path where the timeline JSON should be written
///
/// # Returns
///
/// Returns the command outputs from running the egglog program
///
/// # Example
///
/// ```rust
/// use egglog::tests::poach_testing::run_egglog_file_with_timeline_output;
///
/// // Run an egglog file and save timeline
/// let outputs = run_egglog_file_with_timeline_output("my_program.egg", "timeline.json").unwrap();
/// println!("Executed {} commands", outputs.len());
/// ```
pub fn run_egglog_file_with_timeline_output<P: AsRef<Path>>(
    egg_file_path: P,
    json_output_path: P,
) -> Result<Vec<CommandOutput>, Error> {
    // Create a new TimedEgraph with default configuration
    let mut timed_egraph = TimedEgraph::new();

    // Read the .egg file
    let egg_content = fs::read_to_string(&egg_file_path)
        .map_err(|e| Error::IoError(egg_file_path.as_ref().to_path_buf(), e, span!()))?;

    // Get the filename for error reporting
    let filename = egg_file_path
        .as_ref()
        .file_name()
        .and_then(|name| name.to_str())
        .map(|s| s.to_string());

    // Run the program
    let outputs = timed_egraph.parse_and_run_program(filename, &egg_content)?;

    // Serialize the timeline to JSON
    let timeline_json = timed_egraph
        .serialized_timeline()
        .map_err(|e| Error::BackendError(format!("Failed to serialize timeline: {}", e)))?;

    // Write the JSON to the output file
    fs::write(&json_output_path, timeline_json)
        .map_err(|e| Error::IoError(json_output_path.as_ref().to_path_buf(), e, span!()))?;

    Ok(outputs)
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        eprintln!("Usage: {} <filename.egg | dir/> <output_dir>", args[0]);
        std::process::exit(1);
    }

    let input = &args[1];
    let output_dir = PathBuf::from(&args[2]);

    // Ensure the output directory exists
    if !output_dir.exists() {
        fs::create_dir_all(&output_dir).unwrap_or_else(|e| {
            eprintln!(
                "Failed to create output directory {}: {}",
                output_dir.display(),
                e
            );
            std::process::exit(1);
        });
    }

    if input.ends_with(".egg") {
        let input_path = PathBuf::from(input);
        if !input_path.exists() {
            eprintln!("File not found: {}", input);
            std::process::exit(1);
        }

        let output_path = output_dir.join(format!(
            "{}.json",
            input_path.file_stem().unwrap().to_string_lossy()
        ));

        match run_egglog_file_with_timeline_output(&input_path, &output_path) {
            Ok(_) => println!("Output written to {}", output_path.display()),
            Err(e) => {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        let input_path = PathBuf::from(input);
        if !input_path.is_dir() {
            eprintln!("Directory not found: {}", input);
            std::process::exit(1);
        }

        // Iterate over .egg files in the directory
        match fs::read_dir(&input_path) {
            Ok(entries) => {
                for entry in entries {
                    if let Ok(entry) = entry {
                        let path = entry.path();
                        if path.extension() == Some(OsStr::new("egg")) {
                            let output_path = output_dir.join(format!(
                                "{}.json",
                                path.file_stem().unwrap().to_string_lossy()
                            ));

                            match run_egglog_file_with_timeline_output(&path, &output_path) {
                                Ok(_) => println!("Output written to {}", output_path.display()),
                                Err(e) => {
                                    eprintln!("Error processing {}: {}", path.display(), e);
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to read directory {}: {}", input, e);
                std::process::exit(1);
            }
        }
    }
}
