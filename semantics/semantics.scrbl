#lang scribble/manual

@(require racket/match)
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

@(reduction-relation-rule-separation 10)

@(define (set-notation lws)
  (match lws
    [`(,paren ,tset ,elements ... ,paren2)
      (append  `("{")
                elements
                `("}"))]
    [else (error "bad tset")]))

@(define (render redex-object)
  (with-compound-rewriters
   (['tset set-notation]
    ['congr set-notation]
    )
    (cond
     [(reduction-relation? redex-object)
       (render-reduction-relation redex-object)]
     [else
      (render-language redex-object)])))


@title[#:style  title-style]{ Egglog Semantics }

@section{Egglog Grammar}

@(render Egglog)

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

@(render Egglog+Database)

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

Egglog's top-level reduction relation @-->_Program runs a sequence of commands.

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
  (render Egglog-Reduction))

The @-->_Command reduction relation defines
how each of these commands is run.

@(render Command-Reduction)

The next two sections will cover evaluating actions (@code{Eval-Action}) and evaluating
queries (@code{Rule-Envs}).

@section{Evaluating Actions}

Given an environment @code{Env}, egglog's
actions add new terms and equalities to
the global database.
An action is either a let binding,
an expression,
or a union between two expressions.
At the top level, let bindings add new global
variables to the environment.
In the actions of a rule, let bindings add
new local variables.

@(render-metafunction Eval-Action)

Since actions only add terms to the set of terms
and equalities to the congruence relation,
the order of evaluation of these actions
does not matter.
Actions that bind new local variables
could also be inlined, thus avoiding any dependency between actions.


