let { run } = require("./pkg/");

if (run() !== 0) {
  throw new Error("run() should return 0");
}

console.log("Successfully ran egglog program");
