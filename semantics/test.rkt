#lang racket

(require redex)
(require rackunit)
(require "semantics.rkt")

(define num-executed 0)

(define (types? e)
  (judgment-holds (typed-program ,e TypeEnv)))

(define (has-many-run-statements? prog)
  (>
   (count
    (lambda (cmd)
       (equal? cmd `(run)))
     prog)
   3))

(define (execute prog)
 (cond
    [(not (types? prog))
     (error 'execute "Program does not type check")]
    [else
     (define res
       (apply-reduction-relation* Egglog-Reduction (term (,prog (() () () ())))))
     (match res
       [`((() ,database))
        database]
       [else (error 'execute (format "Program did not terminate. Got ~a" res))])]))

(define (executes? prog)
  ;;(println prog)
  (cond
    [(not (types? prog))
     #t]
    [(has-many-run-statements? prog)
     #t]
    [else
     (define res
       (apply-reduction-relation* Egglog-Reduction (term (,prog (() () () ())))))
     (match res
       [`((() ,database))
        (set! num-executed (+ 1 num-executed))
        #t]
       [_ #f])]))

(module+ test
  (check-false
   (judgment-holds (typed-expr v1 ())))
  (check-false
   (judgment-holds
    (typed-program (v1) TypeEnv)))
  (check-not-false
   (judgment-holds
    (typed-program ((let v1 2) v1)
           TypeEnv)))

  (check-not-false
   (judgment-holds
    (typed-action
      (cadd v1 v1)
      ((v1 : no-type))
      ((v1 : no-type)))))
  (check-not-false
   (judgment-holds
    (typed-actions ((cadd v1 v1)) ((v1 : no-type)) ((v1 : no-type)))))
  (check-not-false
   (judgment-holds
    (typed-query ((= v1 2)) () ((v1 : no-type)))))
  (check-not-false
    (judgment-holds
      (typed-rule
       (rule ((= v1 2)) ((cadd v1 v1)))
       ())))

  (check-false
    (judgment-holds
      (typed-rule
       (rule ((= v1 2)) ((cadd v1 v2)))
       ())))
  (check-not-false
   (judgment-holds
    (typed-query-expr
      v1
      ((v2 : no-type))
      ((v1 : no-type) (v2 : no-type)))))
  (check-not-false
   (judgment-holds
    (typed-query ((= v1 2))
                 ((v2 : no-type))
                 TypeEnv)))
  (check-not-false
    (judgment-holds
      (typed-rule
       (rule ((= v1 2)) ((cadd v1 v2)))
       ((v2 : no-type)))))

      

  (check-not-false
   (redex-match Egglog
                Program
                (term
                 ((let v1 2) v1))))

  (check-not-false
   (redex-match
    Egglog+Database
    Database (term ((1 2) ((= 1 2)) () ()))))


  (check-not-false
   (redex-match
    Egglog+Database
    (Program Database)
    (term
     ((skip) (() () () ())))))

  (check-not-false
   (redex-match
    Egglog+Database
    Command+Database
    (term
     ((cadd 2 3) (() () () ())))))


  (check-equal?
   (apply-reduction-relation*
    Command-Reduction
    (term
     ((cadd 2 3) (() () () ()))))
   (list (term (skip (((cadd 2 3) 2 3) () () ())))))

  (check-equal?
   (term (Eval-Expr 2 ()))
   (term (2 ((2) () () ()))))
  (check-equal?
   (term (Eval-Expr (cadd 2 3) ()))
   (term ((cadd 2 3) (((cadd 2 3) 2 3) () () ()))))

  (check-equal?
   (restore-congruence
    (term
     (()
      ((= 1 2)
       (= 2 3))
      () ())))
   '(() ((= 3 1)
         (= 2 2)
         (= 2 1)
         (= 3 3)
         (= 3 2)
         (= 1 3)
         (= 1 2)
         (= 2 3)) () ()))

  (check-equal?
   (restore-congruence
    (term ((1) () () ())))
   (term ((1) ((= 1 1)) () ())))

  (check-equal?
   (restore-congruence
    (term
     (((cnum 1) (cnum 2) 1 2)
      ((= 1 2))
      () ())))
   '(((cnum 1) (cnum 2) 1 2)
     ((= 1 1) (= (cnum 2) (cnum 2)) (= (cnum 1) (cnum 1)) (= 2 2) (= 2 1) (= 1 2))
     () ()))

  (check-equal?
    (judgment-holds
      (valid-query-subst
        ((1) () () ()) (v1) Env)
      Env)
    '(((v1 -> 1))))

  (check-equal?
   (term
    (Eval-Local-Actions ()
      ((+inf.0)
      ((= +inf.0 +inf.0))
      ((v1 -> +inf.0))
      ((rule () ())))
      ()))
   (term
     ((+inf.0)
      ((= +inf.0 +inf.0))
      ((v1 -> +inf.0))
      ((rule () ())))))

  (check-equal?
    (execute
     (term ((Add 1 2)
            (rule ((Add a b))
                  ((Add b a)))
            (run))))
    '(((Add 1 2) (Add 2 1) 2 1)
      ((= (Add 2 1) (Add 2 1)) (= 2 2) (= 1 1) (= (Add 1 2) (Add 1 2)))
      ()
      ((rule ((Add a b)) ((Add b a))))))

  (check-equal?
   (execute
    (term ((rule ()
                 ((let S 16)
                  (let w (M))
                  (num w)))
           (run))))
    '(((num (M)) (M) 16)
      ((= 16 16) (= (M) (M)) (= (num (M)) (num (M))))
      ()
      ((rule () ((let S 16) (let w (M)) (num w))))))


  (redex-check Egglog
               Program
               (executes? (term Program))
               #:attempts 100000)

  (displayln (format "Executed ~a programs" num-executed))
  )
