#lang racket

(define binary `(Add Sub Mul Div))

(define program
  `((datatype Math
      (Add Math Math)
      (Sub Math Math)
      (Mul Math Math)
      (Div Math Math)
      (Const Rational)
      (Var String))

    (function Parent (Math) Math :merge new)

    ;; rebuilding rules
    (ruleset parent)
    (rule ((Parent a b)
          (Parent b c))
          ((Parent a c)))

    (ruleset rebuilding1)

    ;; make the left child canonical and add it to the eclass
    ,@(for/list ([op binary])
       `(rule ((= e (,op a b))
              (Parent e ep)
              (Parent a p)
              (!= a p))
              ((Parent (Add p b) ep))
              :ruleset rebuilding1))

    (ruleset rebuilding2)

    ;; make the right child canonical and add it to the eclass
    ,@(for/list ([op binary])
       `(rule ((= e (,op a b))
              (Parent e ep)
              (Parent b p)
              (!= b p))
              ((Parent (Add a p) ep))
              :ruleset rebuilding2))


    ;; rules   
  ))


(displayln
  ";; This file does equality saturation without using
;; union, and only sets relations
;; So it is eqsat using datalog, rulesets, and subsumption
;; in the Parent function (:merge new)")

(for ([line program])
  (pretty-write line))