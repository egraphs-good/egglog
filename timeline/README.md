# Using the TimedEgraph

TimedEgraph is a wrapper around the usual EGraph struct. While running an egglog file, it keeps a timeline of events including the start and end times for each command. The timeline can then be serialized to JSON with `egraph.serialized_timeline()`. The implementation of TimedEgraph and IO for it are in `src/`, while Python file(s) for transforming and analyzing the data are in `timeline/`.

To benchmark a .egg file, or all .egg files in a directory, and save the recorded timelines to .json, run the following command:

```
cargo run --bin timeline -- path/to/<file.egg|in_dir/> path/to/out_dir/
```

Running

```
python3 timeline/transform.py path/to/in_dir/ path/to/out_dir/
```

will add data about the concrete s-expression, egglog command, and total time for each event to the JSON.