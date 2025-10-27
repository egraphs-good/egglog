var CHART = null;

var BENCH_SUITES = [
  {
    name: "Herbie",
    dir: "herbie-hamming",
    color: "blue",
  },
  {
    name: "Easteregg",
    dir: "easteregg",
    color: "red",
  }
];

// Top-level load function for the timeline page.
function load_timeline() {
  const suitePromises = BENCH_SUITES.map((suite) => {
    return fetch(`data/${suite.dir}/list.json`)
      .then((response) => response.json())
      .then((names) => {
        return getDatapoints(suite.dir, names).then((data) => {
          return {
            ...suite,
            data: data,
          };
        });
      });
  });

  Promise.all(suitePromises).then((resolvedSuites) => {
    BENCH_SUITES = resolvedSuites;
    plot();
  });
}

function getDatapoints(suite, names) {
  // map from filename to timeline data
  const allData = {};
  const aggregatedData = {};

  const promises = names.map((name) =>
    fetch(`data/${suite}/${name}`)
      .then((r) => r.json())
      .then((d) => (allData[name] = d[0]))
  );

  return Promise.all(promises).then(() => {
    const RUN_CMDS = ["run", "run-schedule"];
    const EXT_CMDS = ["extract"];
    Object.keys(allData).forEach((filename) => {
      const times = {
        runs: [],
        exts: [],
        others: [],
      };
      const entries = allData[filename].evts;
      entries.forEach((entry) => {
        const ms = entry.total_ms;
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
      aggregatedData[filename] = times;
    });
    return aggregatedData;
  });
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
  if (CHART !== null) {
    CHART.destroy();
  }

  const ctx = document.getElementById("chart").getContext("2d");

  const mode = document.querySelector('input[name="mode"]:checked').value;

  const datasets = BENCH_SUITES.map((suite) => { return {
    label: suite.name,
    data: Object.values(suite.data).map((entry) => {
      return { x: aggregate(entry.runs, mode), y: aggregate(entry.exts, mode) };
    }),
    backgroundColor: suite.color,
    pointRadius: 4,
  };});

  CHART = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: datasets,
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
