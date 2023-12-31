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
@(metafunction-rule-gap-space 10)

@(define (set-notation lws)
  (match lws
    [`(,paren ,tset ,elements ... ,paren2)
      (append  `("{")
                elements
                `("}"))]
    [else (error "bad tset")]))

@(define (my-render-reduction-relation reduction-relation)
  (render
   (lambda ()
    (render-reduction-relation reduction-relation))))

@(define (my-render-language language)
  (render
   (lambda ()
    (render-language language))))

@(define-syntax-rule (my-render-judgement judgement)
  (render
   (lambda ()
    (render-judgment-form judgement))))

@(define-syntax-rule (my-render-metafunction metafunction)
  (render
   (lambda ()
    (render-metafunction metafunction))))

@(define (render func)
  (parameterize ([rule-pict-style 'compact-vertical]
                 [metafunction-fill-acceptable-width 10])
    (with-compound-rewriters
     (['tset set-notation]
      ['congr set-notation])
     (func))))


@title[#:style  title-style]{ Egglog Semantics }

@section{Egglog Grammar}

@(my-render-language Egglog)

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

@(my-render-language Egglog+Database)

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
  (my-render-reduction-relation Egglog-Reduction))

The @-->_Command reduction relation defines
how each of these commands is run.

@(my-render-reduction-relation Command-Reduction)

The next two sections will cover evaluating actions (@code{Eval-Action}) and evaluating
queries (@code{Rule-Envs}).

@section{Evaluating Actions}

Evaluating expressions is straitforward-
expressions construct a single term, looking
up variables in the environment.
While evaluating an expression, each intermediate
term is also added to a resulting database.

@(my-render-metafunction Eval-Expr)

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

The resulting database after evaluating expressions
contains all intermidate terms.
These are added to the database using the
@code{Database-Union} metafunction, which
unions the terms, equalities, environments,
and rules of two databases.

@(my-render-metafunction Eval-Action)


Evaluating a sequence of actions is done
by evaluating each action in turn.
Since actions only add terms to the set of terms
and equalities to the congruence relation,
the order of evaluation of these actions
does not matter (besides needing variables to be bound first).

@(my-render-metafunction Eval-Global-Actions)

In order to evaluate local actions for
a rule, we evaluate the actions in the global
scope, then forget the resulting environment.

@(my-render-metafunction Eval-Local-Actions)

Given a set of substitutions, we can evaluate
a rule's actions by applying the actions 
multiple times, once for each 

@(my-render-metafunction Eval-Rule-Actions)

@section{Evaluating Queries}

To evaluate a rule, egglog first finds
all valid substitutions for the rule's @code{Query}.

@(my-render-metafunction Rule-Envs)

The @code{valid-query-subst} judgement defines
which substitutions are valid for a query.
A substitution is valid for a query if it
is valid for each pattern in the query.

@(my-render-judgement valid-query-subst)


The @code{valid-subst} judgement defines
when a substitution is valid for a pattern.
It also provides a particular witness term
from the database that the pattern @bold{e-matches}.
A pattern and substitution @bold{e-matches} a witness term if it is equal to that term
modulo the congruence relation.

@(my-render-judgement valid-subst)



