#lang racket

(define binary `(Add Sub Mul Div))

(define program
  `((datatype Math
      (Add Math Math)
      (Sub Math Math)
      (Mul Math Math)
      (Div Math Math)
      (Const i64)
      (Var String))

    (relation Parent (Math Math))

    ;; rebuilding rules
    (ruleset parent-subsume)
    (rule ((Parent a a)
          (Parent a c)
          (!= a c))
        ((delete (Parent a a)))
        :ruleset parent-subsume)

    (ruleset parent)
    (rule ((Parent a b)
           (Parent b c))
          ((Parent a c))
          :ruleset parent)

    (ruleset rebuilding1)

    ;; make the left child canonical and add it to the eclass
    ;; TODO subsume the old one
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
    (ruleset just-add)
    (ruleset do-union)

    ;; commutativity of addition
    (rule ((Add a b))
          ((let rhs (Add b a))
           (Parent rhs rhs))
          :ruleset just-add)
    (rule ((= lhs (Add a b))
           (= rhs (Add b a))
           (Parent a a)
           (Parent b b)
           (Parent lhs p1)
           (Parent rhs p2)
           (= 1 (ordering-less p1 p2))
           )
          ;; set lhs parent to rhs parent
          ((Parent p1 p2))
          :ruleset do-union)

    (let a (Add (Const 1) (Const 2)))
    (Parent a a)
    (Parent (Const 1) (Const 1))
    (Parent (Const 2) (Const 2))


    ;; time to run!
    (run-schedule
      (repeat 10
        (saturate parent-subsume)
        (saturate parent)
        (saturate rebuilding1)
        (saturate rebuilding2)
        (saturate just-add)
        (saturate parent-subsume)
        (saturate do-union)))
    
    (run-schedule (saturate parent-subsume))

    (let b (Add (Const 2) (Const 1)))

    (check (Parent a p1) (Parent b p2) (= p1 p2))
  ))


(displayln
  ";; This file does equality saturation without using
;; union, and only sets relations
;; So it is eqsat using datalog, rulesets, and subsumption
;; in the Parent function (:merge new)")

(for ([line program])
  (pretty-write line))