import { convertToTable } from "./table.js";

const STATE = {
  activeSuite: null,
};

const GLOBAL_DATA = {
  data: null,
  suites: null,
};

load();

async function load() {
  const statusNode = document.querySelector("#status");

  const response = await fetch("./data/data.json");
  if (!response.ok) {
    statusNode.textContent = `Failed to load data/data.json: ${error}`;
    return;
  }

  GLOBAL_DATA.data = await response.json();
  GLOBAL_DATA.suites = [
    ...new Set(GLOBAL_DATA.data.passing_benchmarks.map((x) => x.suite_name)),
  ].sort();
  STATE.activeSuite = GLOBAL_DATA.suites[0] ?? null;

  statusNode.textContent = "Loaded data/data.json";
  renderSummary();

  renderSuiteSelectors();
  renderTable();
}

function renderSummary() {
  let ruleMs = 0;
  let extractMs = 0;
  let otherMs = 0;

  for (const reportSummary of GLOBAL_DATA.data.passing_benchmarks) {
    ruleMs += reportSummary.report.rule_ms;
    extractMs += reportSummary.report.extraction_ms;
    otherMs += reportSummary.report.other_ms;
  }

  const numPassing = GLOBAL_DATA.data.passing_benchmarks.length;
  const numFailing = GLOBAL_DATA.data.failing_benchmarks.length;
  const totalTime = GLOBAL_DATA.data.passing_benchmarks
    .map((x) => x.wall_time_s)
    .reduce((a, b) => a + b, 0);

  document.querySelector("#summary-text").textContent =
    `Passing Benchmarks: ${numPassing} | ` +
    `Failing Benchmarks: ${numFailing} | ` +
    `Nightly time: ${totalTime.toFixed(1)} s | ` +
    `Rule running: ${(ruleMs / 1000).toFixed(1)} s | ` +
    `Extraction: ${(extractMs / 1000).toFixed(1)} s | ` +
    `Other: ${(otherMs / 1000).toFixed(1)} s`;
}

function renderSuiteSelectors() {
  document.querySelector("#suite-tabs").innerHTML = GLOBAL_DATA.suites
    .map(
      (suite) =>
        `<button
    type="button"
    class="suite-tab ${suite === STATE.activeSuite ? " is-active" : ""}"
    data-suite-name="${suite}"
      >
      ${suite}
      </button>
      `,
    )
    .join("");

  for (const button of document.querySelectorAll(".suite-tab")) {
    button.addEventListener("click", () => {
      STATE.activeSuite = button.dataset.suiteName;

      for (const btn of document.querySelectorAll(".suite-tab")) {
        btn.classList.toggle(
          "is-active",
          btn.dataset.suiteName === STATE.activeSuite,
        );
      }

      renderTable();
    });
  }
}

function renderTable() {
  const benchmarks = GLOBAL_DATA.data.passing_benchmarks.filter(
    (x) => x.suite_name === STATE.activeSuite,
  );
  const totalTime = benchmarks
    .map((x) => x.wall_time_s)
    .reduce((a, b) => a + b, 0);

  document.querySelector("#active-suite-summary").innerHTML = `
  <div>
    <h3>${STATE.activeSuite}</h3>
    <p>${benchmarks.length} benchmarks | ${totalTime} s</p>
  </div>`;

  const columns = [
    "Benchmark",
    "Wall Time (s)",
    "Rules (ms)",
    "Extraction (ms)",
    "Other (ms)",
  ];

  const rows = benchmarks.map((b) => ({
    Benchmark: b.benchmark_name,
    "Wall Time (s)": b.wall_time_s,
    "Rules (ms)": b.report.rule_ms,
    "Extraction (ms)": b.report.extraction_ms,
    "Other (ms)": b.report.other_ms,
  }));

  const tableDiv = document.querySelector("#active-suite-table");
  tableDiv.innerHTML = "";
  tableDiv.appendChild(convertToTable(columns, rows));
}
