// Top-level load function for the serialize page.
function load_serialize() {
  fetch("/data/summary.json")
    .then((response) => response.json())
    .then(display);
}

function display(data) {
  const success_container = document.getElementById("success");
  data.success.forEach((s) => {
    const elt = document.createElement("li");
    elt.textContent = s;
    success_container.appendChild(elt);
  });

  const fail_container = document.getElementById("fail");
  data.fail.forEach((s) => {
    const elt = document.createElement("li");
    elt.textContent = s;
    fail_container.appendChild(elt);
  });
}
