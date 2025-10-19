import os
import json
import matplotlib.pyplot as plt

RUN_CMDS = ["run", "run-schedule"]
EXTRACT_CMDS = ["extract"]

def sum_command_runtimes(directory_path):
	"""
	Sums the runtime in milliseconds of each type of command
	(run, extract, and other) for JSON files in the given directory.

	Returns
		[[filename, run_time, extract_time, other_time, num_runs, num_extracts, num_others]
		for filename in directory_path]
	"""
	totals = []

	for filename in os.listdir(directory_path):
		if filename.endswith(".json"):
			file_path = os.path.join(directory_path, filename)
			with open(file_path, "r") as file:
				try:
					data = json.load(file)
				except json.JSONDecodeError:
					print(f"Could not decode JSON in file {filename}")
					continue

				file_times = [filename, 0.0, 0.0, 0.0, 0, 0, 0]
				for entry in data[0]["evts"]:
					entry_ms = entry["total_time"]["secs"] * 1e3 + entry["total_time"]["nanos"] / 1e6
					command = entry["cmd"]
					if command in RUN_CMDS:
						file_times[1] += entry_ms
						file_times[4] += 1
					elif command in EXTRACT_CMDS:
						file_times[2] += entry_ms
						file_times[5] += 1
					else:
						file_times[3] += entry_ms
						file_times[6] += 1

				totals += [file_times]

	return totals

def plot_run_extract(totals, dataset, output_path):
    """
    Saves a scatter plot of total run time vs total extract time.
    """
    run_times = [entry[1] for entry in totals]
    extract_times = [entry[2] for entry in totals]

    plt.figure(figsize=(8, 6))
    plt.scatter(run_times, extract_times, alpha=0.7, edgecolors='w', color='blue')
    plt.title(f"Total Run vs Total Extract Time ({dataset})")
    plt.xlabel("Total Run Time (ms)")
    plt.ylabel("Total Extract Time (ms)")
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.savefig(output_path)
    plt.close()

def main(in_dir, dataset):
	total_times = sum_command_runtimes(in_dir)
	output_path = os.path.join(in_dir, "total_run_total_extract.png")
	plot_run_extract(total_times, dataset, output_path)

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Get totals for each type of command.")
	parser.add_argument("in_dir", type=str, help="Path to the input directory containing JSON files")
	parser.add_argument("dataset", type=str, help="The label for the benchmark dataset")

	args = parser.parse_args()
	main(args.in_dir, args.dataset)