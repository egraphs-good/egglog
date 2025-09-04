import { run } from "./pkg/egglog_webasm_test.js";

if (run() !== 0) {
  throw new Error("run() should return 0");
}

console.log("Successfully ran egglog program");
