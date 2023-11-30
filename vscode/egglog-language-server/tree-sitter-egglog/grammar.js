// This grammar is one to one translation with few exceptions from original Egglog grammar
// https://github.com/egraphs-good/egglog/blob/8fc012f784cc810f79e8d6907f362955e46b559f/src/ast/parse.lalrpop

// Its licence is as follows:
/*
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
*/

const list = ($, item) => seq($.lparen, repeat(item), $.rparen);

module.exports = grammar({
  name: "egglog",

  extras: ($) => [$.comment, $.ws],

  rules: {
    // TODO: add the actual grammar rules
    source_file: ($) => repeat($.command),

    comment: ($) => token(seq(";", /.*/)),
    ws: ($) => token(/\s+/),

    lparen: ($) => token(choice("(", "[")),
    rparen: ($) => token(choice(")", "]")),

    comma: ($) => repeat1(seq($.expr, ",")),

    command: ($) =>
      choice(
        seq($.lparen, "set-option", $.ident, $.expr, $.rparen),
        seq($.lparen, "datatype", $.ident, repeat($.variant), $.rparen),
        seq(
          $.lparen,
          "sort",
          $.ident,
          $.lparen,
          $.ident,
          repeat($.expr),
          $.rparen,
          $.rparen
        ),
        seq($.lparen, "sort", $.ident, $.rparen),
        seq(
          $.lparen,
          "function",
          $.ident,
          $.schema,
          optional($.cost),
          optional(":unextractable"),
          optional(seq(":on_merge", list($, $.action))),
          optional(seq(":merge", $.expr)),
          optional(seq(":default", $.expr)),
          $.rparen
        ),
        seq($.lparen, "declare", $.ident, $.type, $.rparen),
        seq($.lparen, "relation", $.ident, list($, $.type), $.rparen),
        seq($.lparen, "ruleset", $.ident, $.rparen),
        seq(
          $.lparen,
          "rule",
          list($, $.fact),
          list($, $.action),
          optional(seq(":ruleset", $.ident)),
          optional(seq(":name", $.string)),
          $.rparen
        ),
        seq(
          $.lparen,
          "rewrite",
          $.expr,
          $.expr,
          optional(seq(":when", list($, $.fact))),
          optional(seq(":ruleset", $.ident)),
          $.rparen
        ),
        seq(
          $.lparen,
          "birewrite",
          $.expr,
          $.expr,
          optional(seq(":when", list($, $.fact))),
          optional(seq(":ruleset", $.ident)),
          $.rparen
        ),
        seq($.lparen, "let", $.ident, $.expr, $.rparen),
        $.nonletaction,
        seq(
          $.lparen,
          "run",
          $.unum,
          optional(seq(":until", repeat($.fact))),
          $.rparen
        ),
        seq(
          $.lparen,
          "run",
          $.ident,
          $.unum,
          optional(seq(":until", repeat($.fact))),
          $.rparen
        ),
        seq($.lparen, "simplify", $.schedule, $.expr, $.rparen),
        seq(
          $.lparen,
          "calc",
          $.lparen,
          repeat($.identsort),
          $.rparen,
          repeat1($.expr),
          $.rparen
        ),
        seq(
          $.lparen,
          "query-extract",
          optional(seq(":variants", $.unum)),
          $.expr,
          $.rparen
        ),
        seq($.lparen, "check", repeat($.fact), $.rparen),
        seq($.lparen, "check-proof", $.rparen),
        seq($.lparen, "run-schedule", repeat($.schedule), $.rparen),
        seq($.lparen, "print-stats", $.rparen),
        seq($.lparen, "push", optional($.unum), $.rparen),
        seq($.lparen, "pop", optional($.unum), $.rparen),
        seq($.lparen, "print-function", $.ident, $.unum, $.rparen),
        seq($.lparen, "print-size", optional($.ident), $.rparen),
        seq($.lparen, "input", $.ident, $.string, $.rparen),
        seq($.lparen, "output", $.string, repeat1($.expr), $.rparen),
        seq($.lparen, "fail", $.command, $.rparen),
        seq($.lparen, "include", $.string, $.rparen)
      ),

    schedule: ($) =>
      choice(
        seq($.lparen, "saturate", repeat($.schedule), $.rparen),
        seq($.lparen, "seq", repeat($.schedule), $.rparen),
        seq($.lparen, "repeat", $.unum, repeat($.schedule), $.rparen),
        seq($.lparen, "run", optional(seq(":until", repeat($.fact))), $.rparen),
        seq(
          $.lparen,
          "run",
          $.ident,
          optional(seq(":until", repeat($.fact))),
          $.rparen
        ),
        seq($.ident)
      ),

    cost: ($) => seq(":cost", $.unum),

    nonletaction: ($) =>
      choice(
        seq(
          $.lparen,
          $.lparen,
          "set",
          $.ident,
          repeat($.expr),
          $.rparen,
          $.expr,
          $.rparen
        ),
        seq(
          $.lparen,
          "delete",
          $.lparen,
          $.ident,
          repeat($.expr),
          $.rparen,
          $.rparen
        ),
        seq($.lparen, "union", $.expr, $.expr, $.rparen),
        seq($.lparen, "panic", $.string, $.rparen),
        seq($.lparen, "extract", $.expr, $.rparen),
        seq($.lparen, "extract", $.expr, $.expr, $.rparen),
        $.callexpr
      ),

    action: ($) =>
      choice(seq($.lparen, "let", $.ident, $.expr, $.rparen), $.nonletaction),

    name: ($) => seq("[", $.ident, "]"),

    fact: ($) =>
      choice(seq($.lparen, "=", repeat1($.expr), $.expr, $.rparen), $.callexpr),

    schema: ($) => seq(list($, $.type), $.type),

    expr: ($) => choice($.literal, $.ident, $.callexpr),

    literal: ($) => choice($.unit, $.bool, $.num, $.f64, $.symstring),

    callexpr: ($) => seq($.lparen, $.ident, repeat($.expr), $.rparen),

    exprlist: ($) => seq($.lparen, repeat($.expr), $.rparen),

    variant: ($) =>
      seq($.lparen, $.ident, repeat($.type), optional($.cost), $.rparen),

    // Using slightly different regex due to tree-sitter's bug(?)
    type: ($) => /(([[a-zA-Za]][\w-]*)|([-+*/?!=<>&|^/%_]))+/,

    identsort: ($) => seq($.lparen, $.ident, $.type, $.rparen),
    unit: ($) => seq($.lparen, $.rparen),
    bool: ($) => choice("true", "false"),
    num: ($) => /(-)?[0-9]+/,
    unum: ($) => /[0-9]+/,
    f64: ($) =>
      choice("NaN", /(-)?[0-9]+\.[0-9]+(e(\+)?(-)?[0-9]+)?/, "inf", "-inf"),
    ident: ($) => /(([[a-zA-Z]][\w-]*)|([-+*/?!=<>&|^/%_]))+/,
    symstring: ($) => $.string,
    string: ($) =>
      token(
        seq(
          '"',
          repeat(
            alias(token.immediate(prec(1, /[^\\"\n]+/)), $.string_content)
          ),
          '"'
        )
      ),
  },
});
