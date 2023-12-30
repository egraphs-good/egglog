#lang scribble/manual

@(require scribble-math/dollar)
@(require redex)
@(require "semantics.rkt")
@(require (except-in pict table))
@(require scribble/core
          scribble/html-properties
          racket/runtime-path
          (only-in xml cdata))

@(define head-google
  (head-extra (cdata #f #f "<link rel=\"stylesheet\"
          href=\"https://fonts.googleapis.com/css?family=Nunito+Sa
          ns\">")))

@(define-runtime-path css-path "style.css")
@(define css-object (css-addition css-path))

@(define html5-style
   (with-html5 manual-doc-style))
@(define title-style
   (struct-copy style html5-style
                [properties
                 (append
                  (style-properties html5-style)
                  (list css-object head-google))]))

@title[#:style  title-style]{ Egglog Semantics }

@section{Egglog Grammar}

@(render-language Egglog)

An egglog @code{Program} is a sequence of top-level @code{Cmd}s.
Egglog keeps track of a global @code{Database}
as it runs.
An @code{Action} adds directly to the database.
A @code{Rule} is added to the list of rules,
and a @code{(run)} statement runs the rules.

Rules have two parts: a @code{Query} and @code{Actions}.
The @code{Query} is a set of patterns that must
match terms in the database for the rule to fire.
The @code{Actions} add to the database for every valid match.

@(render-language Egglog+Database)

Egglog's global state is a @code{Database}, containing:
@itemlist[
 @item{A set of @code{Terms}}
 @item{A set of equalities @code{Congr}, which forms a congruence closure}
 @item{A set of global bindings @code{Env}}
 @item{A set of rules @code{Rules}, which are run when @code{(run)} is called}
]

Running an @code{Action} may add terms and equalities to the database.
In-between each command, the congruence closure over these terms is completed.

@section{Running Commands}

@(define stepped
  (hc-append 10
   (render-term Egglog
    (Cmd_1 Database_1))
   -->_Command
   (render-term Egglog
    (Cmd_stepped Database_2))))

    
@(with-unquote-rewriter
  (lambda (old)
   (struct-copy lw old
     [e stepped]))
  (render-reduction-relation Egglog-Reduction))
