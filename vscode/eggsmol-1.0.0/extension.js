// The module 'vscode' contains the VS Code extensibility API
const vscode = require('vscode');
const { exec } = require('node:child_process');

/**
* @param {vscode.ExtensionContext} context
*/
function activate(context) {
	eggsmolChannel = vscode.window.createOutputChannel("Eggsmol output");
	eggsmolChannel.show(true);
	context.subscriptions.push(vscode.commands.registerCommand('eggsmol.Eggsmol_run', async function () {
		eggsmolChannel.clear();
		eggsmolChannel.show(true);
		const document = vscode.window.activeTextEditor.document;
		document.save().then(() => {
			var folder;
			if (vscode.workspace.workspaceFolders != null && vscode.workspace.workspaceFolders.length > 0) {
				folder = vscode.workspace.workspaceFolders[0].uri.fsPath;
			} else {
				folder = ".";
			}
			const relativeFile = document.uri.fsPath;
			eggsmolChannel.appendLine("Running '" + `target/debug/eggsmol ${relativeFile}` + "' in " + folder);
			exec(`target/debug/eggsmol ${relativeFile}`, {cwd: folder}, (err, stdout, stderr) => {
				eggsmolChannel.show(true);
				if (err) {
					eggsmolChannel.append(err);
				}
				if (stdout != "") {
					eggsmolChannel.append(stdout);
				}
				if (stderr != "") {
					eggsmolChannel.append(stderr);
				}
			});
		});
	}));

}

// This method is called when your extension is deactivated
function deactivate() {}

module.exports = {
	activate,
	deactivate
}
