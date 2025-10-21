const RUN_CMDS = ["run", "run-schedule"];
const ACT_CMDS = ["extract"];

// Top-level load function for the timeline page.
function load_timeline() {
  fetch("data/list.json")
    .then((response) => response.json())
    .then(buildGlobalData);
}

function buildGlobalData(names) {
  const allData = {};
  const promises = names.map((name) =>
    fetch(`data/${name}`)
      .then((r) => r.json())
      .then((d) => (allData[name] = d[0]))
  );

  Promise.all(promises).then(() => {
    const aggregated = aggregate(allData);
    plot(aggregated);
  });
}

function aggregate(data) {
  const aggregated = {};
  Object.keys(data).forEach((filename) => {
    times = {
      runs: [],
      acts: [],
      others: [],
    };
    const entries = data[filename].evts;
    entries.forEach((entry) => {
      const ms = toMs(entry.total_time);
      const cmd = entry.cmd;
      if (RUN_CMDS.includes(cmd)) {
        times.runs.push(ms);
      } else if (ACT_CMDS.includes(cmd)) {
        times.acts.push(ms);
      } else {
        times.others.push(ms);
      }
    });
    aggregated[filename] = times;
  });

  return aggregated;
}

function toMs(duration) {
  return duration.secs * 1e3 + duration.nanos / 1e6;
}

function plot(data) {
  const ctx = document.getElementById("chart").getContext("2d");

  const points = Object.values(data).map((entry) => {
    const runTime = entry.runs.reduce((a, b) => a + b);
    const extractTime = entry.acts.reduce((a, b) => a + b);
    return { x: runTime, y: extractTime };
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
