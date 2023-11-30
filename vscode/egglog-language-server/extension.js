"use strict";
const vscode = require("vscode");
const languageclient = require("vscode-languageclient");
const { exec } = require("node:child_process");

let client;

function activate(context) {
  try {
    const serverOptions = {
      command: "cargo",
      args: [
        "run",
        "--release",
        "--manifest-path",
        context.extensionPath + "/Cargo.toml",
      ],
    };
    const clientOptions = {
      documentSelector: [
        {
          scheme: "file",
          language: "egglog",
        },
      ],
    };
    client = new languageclient.LanguageClient(
      "egglog",
      serverOptions,
      clientOptions
    );
    context.subscriptions.push(client.start());
  } catch (e) {
    vscode.window.showErrorMessage(
      "egglog-language-server couldn't be started."
    );
  }

  context.subscriptions.push(
    vscode.commands.registerCommand("egglog.egglog_run", async function () {
      const document = vscode.window.activeTextEditor.document;
      document.save().then(() => {
        const relativeFile = document.uri.fsPath;

        let process_exec = new vscode.ProcessExecution("egglog", [
          relativeFile
        ]);

        const task = new vscode.Task({ type: "process" }, vscode.TaskScope.Workspace, "egglog", "egglog", process_exec);
        // https://github.com/microsoft/vscode/issues/157756
        task.definition.command = "egglog";

        vscode.tasks.executeTask(task);
      });
    })
  );
  context.subscriptions.push(
    vscode.commands.registerCommand("egglog.egglog_desugar", async function () {
      const document = vscode.window.activeTextEditor.document;
      document.save().then(() => {
        const relativeFile = document.uri.fsPath;

        let process_exec = new vscode.ProcessExecution("egglog", [
          "--desugar",
          relativeFile
        ]);

        const task = new vscode.Task({ type: "process" }, vscode.TaskScope.Workspace, "egglog-desugar", "egglog", process_exec);
        task.definition.command = "egglog";

        vscode.tasks.executeTask(task);
      });
    })
  );
  context.subscriptions.push(
    vscode.commands.registerCommand(
      "egglog.egglog_dot_preview",
      async function () {
        const document = vscode.window.activeTextEditor.document;
        document.save().then(() => {
          const relativeFile = document.uri.fsPath;

          const command = `egglog --to-dot ${relativeFile}`;
          exec(command).on("exit", (code) => {
            if (code === 0) {
              const dotFile = vscode.Uri.parse(
                relativeFile.replace(/\.egg$/, "") + ".dot"
              );
              vscode.workspace.openTextDocument(dotFile).then((doc) => {
                vscode.window.showTextDocument(doc, 1, false);
              });
            } else {
              vscode.window.showErrorMessage(
                `${command} exited with code ${code}`
              );
            }
          });
        });
      }
    )
  );
}

function deactivate() {
  if (client) return client.stop();
}

module.exports = { activate, deactivate };
