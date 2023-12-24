#lang racket

(require redex)
(require rackunit)

(define-language Egglog
  (cmds (cmd ...))
  (cmd action)
  (action expr
          (let var expr))
  (expr number
        (constructor expr ...)
        var)
  (constructor variable-not-otherwise-mentioned)
  (var variable-not-otherwise-mentioned))

(define-extended-language
  Egglog+Database
  Egglog
  [Database (Terms)]
  [Terms (Term ...)]
  [Term number
        (constructor Term ...)])

(check-not-false
 (redex-match Egglog
  cmds
  (term
    ((let a 2) a))))

;;(define-judgment-form
;;  Egglog
;;  #:contract (

