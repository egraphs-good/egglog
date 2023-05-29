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

    (function Parent (Math) Math :merge (ordering-less old new))

    ;; rebuilding rules
    (ruleset parent)
    (rule ((= (Parent a) b)
           (= (Parent b) c))
          ((set (Parent a) c))
          :ruleset parent)

    (ruleset rebuilding)

    ;; make the children canonical and add it to the eclass
    ,@(for/list ([op binary])
       `(rule ((= e (,op a b)))
              ((let lhs (Add (Parent a) (Parent b)))
               (let rhs (Parent e))
                (set (Parent lhs) rhs)
                (set (Parent rhs) lhs))
              :ruleset rebuilding))

    ;; commutativity of addition
    (rule ((= lhs (Add a b))
           (= (Parent a) a)
           (= (Parent b) b))
          ;; set lhs parent to rhs parent
          ((set (Parent lhs) (Add b a))
           (set (Parent (Add b a)) lhs)))

    (let a (Add (Const 1) (Const 2)))
    (set (Parent a) a)
    (set (Parent (Const 1)) (Const 1))
    (set (Parent (Const 2)) (Const 2))


    ;; time to run!
    (run-schedule
      (repeat 10
        (saturate parent)
        (saturate rebuilding)
        (saturate (run 1))))
    
    (run-schedule (saturate parent))
    (run-schedule (saturate rebuilding))

    (let b (Add (Const 2) (Const 1)))

    (check (= (Parent a) p1) (= (Parent b) p2) (= p1 p2))
  ))


(displayln
  ";; This file does equality saturation without using
;; union, and only sets relations
;; So it is eqsat using datalog, rulesets, and subsumption
;; in the Parent relation")

(for ([line program])
  (pretty-write line))