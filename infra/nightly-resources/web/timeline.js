const BENCH_SUITES = [
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

let chart = null;
let loadedData = [];

// Top-level load function for the timeline page.
function loadTimeline() {
  Promise.all(BENCH_SUITES.map(
    (suite) => fetch(`data/${suite.dir}/list.json`)
      .then((response) => response.json())
      .then((names) => getDatapoints(suite.dir, names)
        .then((data) => ({ ...suite, data }))
      )
  )).then((results) => {
    loadedData = results;
    plot();
  });
}

function getDatapoints(suite, names) {
  const RUN_CMDS = ["run", "run-schedule"];
  const EXT_CMDS = ["extract"];

  const datapoints = names.map((name) =>
    fetch(`data/${suite}/${name}`)
      .then((response) => response.json())
      .then((data) => data[0].evts)
      .then((events) => {
        const times = {
          runs: [],
          exts: [],
          others: [],
        };

        events.forEach((entry) => {
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

        return times;
      })
  );

  return Promise.all(datapoints);
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
  if (chart !== null) {
    chart.destroy();
  }

  const ctx = document.getElementById("chart").getContext("2d");

  const mode = document.querySelector('input[name="mode"]:checked').value;

  const datasets = loadedData.map((suite) => ({
    label: suite.name,
    data: Object.values(suite.data).map((entry) => {
      return { x: aggregate(entry.runs, mode), y: aggregate(entry.exts, mode) };
    }),
    backgroundColor: suite.color,
    pointRadius: 4,
  }));

  chart = new Chart(ctx, {
    type: "scatter",
    data: { datasets },
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
