# egglog language server

A language server and VScode extension for [egglog](https://github.com/egraphs-good/egglog).

## Features

- [x] Press F7 to run egglog file
- [x] Press F8 to desugar egglog file
- [x] Press F9 to run egglog file with `--to-dot` and show dot file
- [x] Syntax highlighting
- [x] Prser Diagnostics
- [x] Formatting
- [x] Hover
    - [x] Definition
    - [x] Desugar
- [x] Completion
    - [x] Keywords
    - [x] Global variables
- [x] GOTO Definition

## Installation

## Prerequisites

- cargo
- `npm install -g @vscode/vsce`

To fully functional this extension, you need to install `egglog` command on your $PATH.

## Build

Install from [Marketplace](https://marketplace.visualstudio.com/items?itemName=hatookov.egglog-language). Or

```bash
npm i
vsce package
```

and install .vsix file to your vscode.

### Note

Language server will be compiled at first time you open an egglog file. It may take a while.

## License

Codes are based on original [egglog](https://github.com/egraphs-good/egglog/tree/main/vscode/eggsmol-1.0.0) extension.
```text
MIT License

Copyright (c) 2022 Max Willsey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```