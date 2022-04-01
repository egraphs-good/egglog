importScripts("web_demo.js")
console.log("I'm in the worker")

let { run_program } = wasm_bindgen;
async function work() {
    await wasm_bindgen("web_demo_bg.wasm");

    // Set callback to handle messages passed to the worker.
    self.onmessage = async event => {
        try {
            let result = run_program(event.data);
            console.log("Got result from worker");
            self.postMessage(result);
        } catch (error) {
            console.log(error);
            self.postMessage("Something panicked! Check the console logs...");
        }
    };
}

work()