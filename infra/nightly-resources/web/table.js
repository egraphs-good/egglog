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
 */
export function convertToTable(columns, rows) {
  const table = document.createElement("table");

  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  for (const column of columns) {
    const th = document.createElement("th");
    th.textContent = column;
    headerRow.appendChild(th);
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const column of columns) {
      const td = document.createElement("td");
      td.textContent = row[column] ?? "-"; // We want to leave 0 and "" intact, so don't use ||
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);

  return table;
}
