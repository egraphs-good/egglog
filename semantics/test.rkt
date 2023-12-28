#lang racket

(require redex)
(require rackunit)
(require "semantics.rkt")

(module+ test
  (check-false
   (judgment-holds (typed-expr a ())))
  (check-false
   (judgment-holds
    (typed (a) TypeEnv)))
  (check-not-false
   (judgment-holds
    (typed ((let a 2) a)
           TypeEnv)))

  (check-not-false
   (redex-match Egglog
                Program
                (term
                 ((let a 2) a))))

  (check-not-false
   (redex-match
    Egglog+Database
    (Program Database)
    (term
     ((skip) (() () ())))))

  (check-not-false
   (redex-match
    Egglog+Database
    Command+Database
    (term
     ((Add 2 3) (() () ())))))


  (check-equal?
   (apply-reduction-relation*
    Command-Reduction
    (term
     ((Add 2 3) (() () ()))))
   (list (term (skip (((Add 2 3) 2 3) () ())))))

  (check-equal?
   (term (Eval-Expr 2 ()))
   (term (2 ((2) () ()))))
  (check-equal?
   (term (Eval-Expr (Add 2 3) ()))
   (term ((Add 2 3) (((Add 2 3) 2 3) () ()))))

  (check-equal?
   (restore-congruence
    (term
     (()
      ((= 1 2)
       (= 2 3))
      ())))
   '(() ((= 3 1)
         (= 2 2)
         (= 2 1)
         (= 3 3)
         (= 3 2)
         (= 1 3)
         (= 1 2)
         (= 2 3)) ()))

  (check-equal?
   (restore-congruence
    (term ((1) () ())))
   (term ((1) ((= 1 1)) ())))

  (check-equal?
   (restore-congruence
    (term
     (((Num 1) (Num 2) 1 2)
      ((= 1 2))
      ())))
   '(((Num 1) (Num 2) 1 2)
     ((= 1 1) (= (Num 2) (Num 2)) (= (Num 1) (Num 1)) (= 2 2) (= 2 1) (= 1 2))
     ()))


  (redex-check Egglog
               Program
               (executes? (term Program))
               #:attempts 10000)
  )
