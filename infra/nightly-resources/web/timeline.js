var AGGREGATED_DATA = {};
// Top-level load function for the timeline page.
function load_timeline() {
  // list.json contains a list of all the json files (one for each egg benchmark)
  fetch("data/list.json")
    .then((response) => response.json())
    .then(buildGlobalData);
}

function buildGlobalData(names) {
  // map from filename to timeline data
  const allData = {};

  const promises = names.map((name) =>
    fetch(`data/${name}`)
      .then((r) => r.json())
      // @Noah -- why does each data file contain a singleton list?
      // Would there ever be multiple elements? If not, should we make it just the object, not a list?
      .then((d) => (allData[name] = d[0]))
  );

  Promise.all(promises).then(() => {
    const RUN_CMDS = ["run", "run-schedule"];
    const EXT_CMDS = ["extract"];
    Object.keys(allData).forEach((filename) => {
      times = {
        runs: [],
        exts: [],
        others: [],
      };
      const entries = allData[filename].evts;
      entries.forEach((entry) => {
        // compute event duration in ms from recorded seconds and nanoseconds
        // @Noah- should we just record ms directly?
        const ms = toMs(entry.total_time);
        const cmd = entry.cmd;

        // group commands by type (run, extract, other)
        if (RUN_CMDS.includes(cmd)) {
          times.runs.push(ms);
        } else if (EXT_CMDS.includes(cmd)) {
          times.exts.push(ms);
        } else {
          times.others.push(ms);
        }
      });
      AGGREGATED_DATA[filename] = times;
    });

    plot();
  });
}

function toMs(duration) {
  return duration.secs * 1e3 + duration.nanos / 1e6;
}

function aggregate(times, mode) {
  switch (mode) {
    case "average":
      return times.reduce((a, b) => a + b) / times.length;

    case "total":
      return times.reduce((a, b) => a + b);

    case "max":
      return Math.max(...times);

    default:
      console.warn("Unknown selection:", mode);
      return 0;
  }
}

function plot() {
  const ctx = document.getElementById("chart").getContext("2d");

  const mode = document.querySelector('input[name="mode"]:checked').value;

  const points = Object.values(AGGREGATED_DATA).map((entry) => {
    return { x: aggregate(entry.runs, mode), y: aggregate(entry.exts, mode) };
  });

  new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Herbie",
          data: points,
          backgroundColor: "blue",
          pointRadius: 4,
        },
      ],
    },
    options: {
      title: {
        display: false,
      },
      scales: {
        xAxes: [
          {
            type: "linear",
            position: "bottom",
            scaleLabel: {
              display: true,
              labelString: "Run Time (ms)",
            },
          },
        ],
        yAxes: [
          {
            scaleLabel: {
              display: true,
              labelString: "Extract Time (ms)",
            },
          },
        ],
      },
    },
  });
}
