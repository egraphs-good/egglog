[(lparen) (rparen)] @punctuation.bracket
["set-option" "datatype" "sort" "function" "declare" "relation" "ruleset" "rule" "rewrite" "birewrite" "let" "run" "simplify" "calc" "query-extract" "check" "check-proof" "run-schedule" "print-stats" "push" "pop" "print-function" "print-size" "input" "output" "fail" "include"] @keyword
["saturate" "set" "repeat" "delete" "union" "panic" "extract" "let" "true" "false"] @keyword
(fact ("=" @keyword))
[":unextractable" ":on_merge" ":merge" ":default" ":ruleset" ":name" ":when" ":until" ":variants" ":cost"] @attribute

(comment) @comment
(callexpr ((ident) @function))

(command ("sort" (ident) @type))
(command ("datatype" (ident) @type))
(command ("relation" (ident) @function))
(command ("function" (ident) @function))
(command ((lparen) "declare" (ident) @variable (type) @type (rparen)))
(variant ((ident) @function))

[(num) (unum)] @number
(string) @string
(type) @type
(ident) @variable
