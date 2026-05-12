/**
 * Given a list of column names and a list of row objects, produce an
 * HTML table that displays the data
 *
 * @param {String[]} columns
 * Defines the set of columns to be displayed and their order.
 * If empty, an empty table element will be returned
 *
 * @param {Object[]} rows
 * Each object corresponds to one row in the table.
 * Each object maps column name -> cell value.
 * Values should be numbers or strings only.
 * Rows may contain keys not listed in `columns` and may be missing `columns`
 * Extra values are ignored. Missing values are displayed as `-` in the table
 * If empty, a table with only the header row will be returned
 *
 * @return An HTML <table> DOM element
 * Sortable by column, defaults to sorted by first column
 */
export function convertToTable(columns, rows) {
  const table = document.createElement("table");

  const STATE = {
    sortCol: columns[0],
    sortDir: "asc",
  };

  function sortedRows() {
    const dir = STATE.sortDir === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => {
      // Look up value to sort by in each row
      const av = a[STATE.sortCol];
      const bv = b[STATE.sortCol];

      // First check if either/both are missing. Sort missing values to the bottom
      const aMissing = av === undefined || av === null;
      const bMissing = bv === undefined || bv === null;
      if (aMissing && bMissing) return 0;
      if (aMissing) return 1;
      if (bMissing) return -1;

      // Compare
      return (av > bv ? 1 : av < bv ? -1 : 0) * dir;
    });
  }

  function toggleSortDir() {
    if (STATE.sortDir === "asc") {
      STATE.sortDir = "desc";
    } else {
      STATE.sortDir = "asc";
    }
  }

  function render() {
    // Reset table element before rebuilding
    table.innerHTML = "";

    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    for (const column of columns) {
      const th = document.createElement("th");
      const arrow =
        column === STATE.sortCol ? (STATE.sortDir === "asc" ? " ▲" : " ▼") : "";
      th.textContent = column + arrow;
      th.style.cursor = "pointer";
      th.addEventListener("click", () => {
        if (STATE.sortCol === column) {
          toggleSortDir();
        } else {
          STATE.sortCol = column;
          STATE.sortDir = "asc";
        }
        render();
      });
      headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    for (const row of sortedRows()) {
      const tr = document.createElement("tr");
      for (const column of columns) {
        const td = document.createElement("td");
        td.textContent = row[column] ?? "-"; // We want to leave 0 and "" intact, so don't use ||
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
  }

  render();
  return table;
}
