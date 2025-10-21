import json
import re
import os

def load_json(path):
  with open(path, 'r') as file:
    data = json.load(file)
  return data

def save_json(path, data):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'w') as file:
    json.dump(data, file, indent=4)

def merge_start_end_events(timeline):
    """
    Merges pairs of start and end events into a single event with start_time, end_time, and total_time fields.

    Args:
        timeline (list): The JSON data to process.

    Returns:
        list: The updated JSON data.
    """
    for entry in timeline:
        if 'evts' not in entry:
            raise KeyError("Each JSON object must contain 'evts' key.")

        events = entry['evts']
        merged_events = []

        for i in range(0, len(events), 2):
            start_event = events[i]
            end_event = events[i + 1]

            if start_event['evt'] != 'start' or end_event['evt'] != 'end':
                raise ValueError("Events must alternate between 'start' and 'end'.")

            merged_event = {
                'sexp_idx': start_event['sexp_idx'],
                'start_time': start_event['time'],
                'end_time': end_event['time'],
                'total_time': {
                    'secs': end_event['time']['secs'] - start_event['time']['secs'],
                    'nanos': end_event['time']['nanos'] - start_event['time']['nanos']
                }
            }

            if merged_event['total_time']['nanos'] < 0:
                merged_event['total_time']['secs'] -= 1
                merged_event['total_time']['nanos'] += 1_000_000_000

            merged_events.append(merged_event)

        entry['evts'] = merged_events

    return timeline

def add_sexp_strs(timeline):
    """
    Adds the concrete s-expression corresponding to the sexp_id of each event.

    Args:
        timeline (list): The JSON data to process.

    Returns:
        list: The updated JSON data.
    """
    def parse_top_level_s_expressions(program_text):
        # Remove comments
        program_text = '\n'.join(line.split(';', 1)[0].strip() for line in program_text.splitlines())

        stack = []
        current = ''
        expressions = []

        for char in program_text:
            if char == '(':
                if not stack:
                    current = ''
                stack.append('(')
                current += char
            elif char == ')':
                if stack:
                    stack.pop()
                current += char
                if not stack:
                    expressions.append(current)
            else:
                current += char

        return expressions

    for entry in timeline:
        if 'program_text' not in entry or 'evts' not in entry:
            raise KeyError("Each JSON object must contain 'program_text' and 'evts' keys.")

        program_text = parse_top_level_s_expressions(entry['program_text'])
        events = entry['evts']

        for event in events:
            if 'sexp_idx' in event:
                sexp_idx = event['sexp_idx']
                if 0 <= sexp_idx < len(program_text):
                    event['sexp'] = program_text[sexp_idx]
                else:
                    raise IndexError(f"sexp_idx {sexp_idx} is out of bounds for program_text.")

    return timeline

def add_egglog_cmds(timeline):
  """
    Parses the egglog command present in each s-expression.

    Args:
        timeline (list): The JSON data to process.

    Returns:
        list: The updated JSON data.
    """
  for entry in timeline:
    events = entry['evts']

    for event in events:
      if 'sexp' not in event:
        raise KeyError("Event is missing the concrete s-expression.")
      event['cmd'] = re.search(r"[^\(\s]+", event['sexp']).group()

  return timeline

def main(input_dir, output_dir):
    """
    Processes all JSON files in the input directory, applying each transformation in order,
    and writes the results to the output directory.

    Args:
        input_dir (str): Path to the input directory containing JSON files.
        output_dir (str): Path to the output directory to save processed JSON files.
    """
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    save_json(os.path.join(output_dir, "list.json"), files)

    for filename in files:
        input_file_path = os.path.join(input_dir, filename)
        output_file_path = os.path.join(output_dir, filename)

        data = load_json(input_file_path)

        data = merge_start_end_events(data)
        data = add_sexp_strs(data)
        data = add_egglog_cmds(data)

        save_json(output_file_path, data)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process JSON egraph timeline files.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing JSON files.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory to save processed JSON files.")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir)

