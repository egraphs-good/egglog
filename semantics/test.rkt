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
       (apply-reduction-relation* Egglog-Reduction (term (,prog ((tset) (congr) () ())))))
     (match res
       [`((() ,database))
        database]
       [else (error 'execute (format "Program did not terminate. Got ~a" res))])]))

(define (executes? prog)
  (cond
    [(not (types? prog))
     #t]
    [(has-many-run-statements? prog)
     #t]
    [else
     (define res
       (apply-reduction-relation* Egglog-Reduction (term (,prog ((tset) (congr) () ())))))
     (match res
       [`((() ,database))
        (set! num-executed (+ 1 num-executed))
        #t]
       [_ #f])]))

(define (check-equal-databases? db1 db2)
 (match (list db1 db2)
  [`(((tset ,terms1 ...) (congr ,congr1 ...) ,env1 ,rules1)
     ((tset ,terms2 ...) (congr ,congr2 ...) ,env2 ,rules2))
    (and
      (check-equal? (list->set terms1) (list->set terms2))
      (check-equal? (list->set congr1) (list->set congr2))
      (check-equal? env1 env2)
      (check-equal? rules1 rules2))]
  [_ (check-equal? db1 db2)]))

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
    Database (term ((tset 1 2) (congr (= 1 2)) () ()))))


  (check-not-false
   (redex-match
    Egglog+Database
    (Program Database)
    (term
     ((skip) ((tset) (congr) () ())))))

  (check-not-false
   (redex-match
    Egglog+Database
    Command+Database
    (term
     ((cadd 2 3) ((tset) (congr) () ())))))


  (check-equal?
   (apply-reduction-relation*
    Command-Reduction
    (term
     ((cadd 2 3) ((tset) (congr) () ()))))
   (list (term (skip ((tset (cadd 2 3)) (congr) () ())))))

  (check-equal?
   (term (Eval-Expr 2 ()))
   (term 2))
  (check-equal?
   (term (Eval-Expr (cadd 2 3) ()))
   (term (cadd 2 3)))

  (check-equal-databases?
   (term
    (restore-congruence
      ((tset)
        (congr (= 1 2)
        (= 2 3))
        () ())))
   '((tset) (congr (= 3 1)
         (= 2 2)
         (= 2 1)
         (= 3 3)
         (= 3 2)
         (= 1 3)
         (= 1 2)
         (= 2 3)) () ()))

  (check-equal?
   (term (restore-congruence
    ((tset 1) (congr) () ())))
   (term ((tset 1) (congr (= 1 1)) () ())))
  
  (check-true
    (judgment-holds
      (tset-is-subset (tset 1) (tset 1) (tset)))) 
  (check-equal?
   (judgment-holds
     (valid-env (v1)
                ((tset 1) (congr (= 1 1)) () ())
                Env)
     Env)
   `(((v1 -> 1))))

  (check-equal?
    (term (free-vars v1 ()))
    (term (v1)))

  (check-equal?
    (judgment-holds (tset-element 1 (tset 1)))
    #t)
  
  (check-equal?
    (term (Eval-Expr v1 ((v1 -> 1))))
    (term 1))

  (check-equal?
    (judgment-holds
      (valid-env (free-vars v1 ()) ((tset 1) (congr (= 1 1)) () ()) Env)
      Env)
    '(((v1 -> 1))))

  (check-equal?
    (judgment-holds
      (valid-subst
        ((tset 1) (congr (= 1 1)) () ()) v1 Env)
      Env)
    '(((v1 -> 1))))

  (check-equal?
   (term
    (Eval-Local-Actions ()
      ((tset +inf.0)
      (congr (= +inf.0 +inf.0))
      ((v1 -> +inf.0))
      ((rule () ())))
      ()))
   (term
     ((tset +inf.0)
      (congr (= +inf.0 +inf.0))
      ((v1 -> +inf.0))
      ((rule () ())))))

  (check-equal-databases?
    (execute
     (term ((Add 1 2)
            (rule ((Add a b))
                  ((Add b a)))
            (run))))
    '((tset (Add 1 2) 1 2 (Add 2 1))
      (congr (= (Add 2 1) (Add 2 1)) (= 2 2) (= 1 1) (= (Add 1 2) (Add 1 2)))
      ()
      ((rule ((Add a b)) ((Add b a))))))

  (check-equal-databases?
   (execute
    (term ((rule ()
                 ((let S 16)
                  (let w (M))
                  (num w)))
           (run))))
    '((tset 16 (M) (num (M)))
      (congr (= (num (M)) (num (M))) (= (M) (M)) (= 16 16))
      ()
      ((rule () ((let S 16) (let w (M)) (num w))))))

  (check-equal?
   (execute
    (term ((let v (b 1)) (union 7 7) (union v 4))))
   (term
    ((tset (b 1) 4 7 1)
  (congr (= 4 4) (= 4 (b 1)) (= (b 1) 4) (= 7 7) (= (b 1) (b 1)) (= 1 1))
  ((v -> (b 1)))
  ())))

  (check-equal?
    (execute
     (term ((union (r) 22)
            (rule ((r 1) 0 (= (D (q)) 6)) ())
            (run))))
    (term ((tset (r) 22)
        (congr (= (r) (r)) (= 22 22) (= 22 (r)) (= (r) 22))
        ()
        ((rule ((r 1) 0 (= (D (q)) 6)) ())))))

  
  (check-equal?
   (term
    (restore-congruence 
     ((tset 1 2 3)
      (congr (= 1 2) (= 2 3))
      () ())))
   (term ((tset 1 2 3)
          (congr
          (= 1 1)
          (= 3 1)
          (= 2 2)
          (= 2 1)
          (= 3 3)
          (= 3 2)
          (= 1 3)
          (= 1 2)
          (= 2 3))
          ()
          ())))

  (check-equal?
   (term
    (restore-congruence
     ((tset 1 2 (wrapper 2) (wrapper 1))
      (congr (= 1 2)) () ())))
   (term ((tset 1 2 (wrapper 2) (wrapper 1))
          (congr
            (= (wrapper 2) (wrapper 2))
            (= 1 1)
            (= (wrapper 1) (wrapper 1))
            (= (wrapper 1) (wrapper 2))
            (= 2 2)
            (= (wrapper 2) (wrapper 1))
            (= 2 1)
            (= 1 2))
          ()
          ())))

  (check-equal?
   (judgment-holds
     (valid-subst ((tset 1 2 (wrapper 2))
                   (congr (= 1 2) (= 2 1) (= 1 1) (= 2 2) (= (wrapper 2) (wrapper 2)))
                   () ())
                  (wrapper 1)
                  Env)
    Env)
   '(()))

  (check-equal?
    (execute
     (term
       ((Wrapper (Add 1 2))
        (rule ((Add a b)) ((union (Add a b) (Add b a))))
        (rule ((= (Wrapper (Add 1 2))
                  (Wrapper (Add 2 1))))
              ((success)))
        (run)
        (run))))
    'a)



  (redex-check Egglog
               Program
               (executes? (term Program))
               #:attempts 200000)

  (displayln (format "Executed ~a programs" num-executed)))