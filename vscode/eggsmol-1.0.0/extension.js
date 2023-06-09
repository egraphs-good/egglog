// The module 'vscode' contains the VS Code extensibility API
const vscode = require('vscode');
const { exec } = require('node:child_process');

/**
* @param {vscode.ExtensionContext} context
*/
function activate(context) {
	egglogChannel = vscode.window.createOutputChannel("egglog output");
	egglogChannel.show(true);
	context.subscriptions.push(vscode.commands.registerCommand('egglog.egglog_run', async function () {
		egglogChannel.clear();
		egglogChannel.show(true);
		const document = vscode.window.activeTextEditor.document;
		document.save().then(() => {
			var folder;
			if (vscode.workspace.workspaceFolders != null && vscode.workspace.workspaceFolders.length > 0) {
				folder = vscode.workspace.workspaceFolders[0].uri.fsPath;
			} else {
				folder = ".";
			}
			const relativeFile = document.uri.fsPath;
			egglogChannel.appendLine("Running '" + `cargo run ${relativeFile}` + "' in " + folder);
			exec(`cargo run ${relativeFile}`, { cwd: folder }, (err, stdout, stderr) => {
				egglogChannel.show(true);
				if (err) {
					egglogChannel.append(err);
				}
				if (stdout != "") {
					egglogChannel.append(stdout);
				}
				if (stderr != "") {
					egglogChannel.append(stderr);
				}
			});
		});
	}));

}

// This method is called when your extension is deactivated
function deactivate() { }

module.exports = {
	activate,
	deactivate
}
