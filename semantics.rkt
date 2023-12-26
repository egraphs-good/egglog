#lang racket

(require redex)
(require rackunit)

(define-language Egglog
  (Program
   (cmd ...))
  (cmd action
       skip)
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
  ;; a database is a list of terms
  ;; the terms list is finite, but combined with
  ;; the congruence closure it can represent an infinite set of terms
  [Database (Terms Congr)]
  [Congr
   (Eq ...)]
  [Eq
   (= Term Term)]
  [Terms
   (Term ...)]
  [Term number
        (constructor Term ...)]
  [Program+Database
   (Program Database)]
  [Command+Database
   (cmd Database)]
  [Env empty-env
       (var -> Term Env)]
  )

;; No rule for if the variable is not found
;; Therefore it fails in such cases
(define-metafunction Egglog+Database
  Lookup : var Env -> Term
  [(Lookup var (var -> Term Env))
   Term]
  [(Lookup var_1 (var_2 -> Term Env))
   (Lookup var Env)
   (side-condition (not (equal? (term var_1)
                                (term var_2))))])

(define-metafunction Egglog+Database
  Database-Union : Database ... -> Database
  [(Database-Union Database)
   Database]
  [(Database-Union ((Term_s ...) (Eq_s ...)) Database_s ...)
   ((Term_s ... Term_s2 ...) (Eq_s ... Eq_s2 ...))
   (where ((Term_s2 ...) (Eq_s2 ...))
          (Database-Union Database_s ...))])



(define-metafunction Egglog+Database
  Eval-Expr : expr Env -> (Term Database)
  [(Eval-Expr number Env)
   (number ((number) ()))]
  [(Eval-Expr (constructor expr_s ...) Env)
   ,(let ([results (term ((Eval-Expr expr_s Env) ...))])
      (term
       ((constructor ,@(map first results))
        (Database-Union ,@(map second results)))))])

#;(define-metafunction Egglog+Database
    Eval-Action : action Env Database -> Database
    [(Eval-Action (let var expr) Env Database)])

(define Command-Reduction
  (reduction-relation
   Egglog+Database
   #:domain Command+Database
   ;; running actions
   (-->
    (action Database)
    (skip Database))))




(define Egglog-Reduction
  (reduction-relation
   Egglog+Database
   #:domain Program+Database
   (-->
    ((cmd_1 cmd_s ...)
     Database)
    ((cmd_stepped cmd_s ...)
     Database_2)
    (where
     (cmd_stepped Database_2)
     ,(first
       (apply-reduction-relation
        Command-Reduction
        (term (cmd_1 Database))))))
   (-->
    ((skip cmd_s ...)
     Database)
    ((cmd_s ...)
     Database))))


;;(define-judgment-form
;;  Egglog
;;  #:contract (evals-to Database Cmds)
;;  [-------------------------
;;   (evals-to empty-terms empty-cmds)]
;;  [(evals-to
;;   -------------------------




(module+ test
  (check-not-false
   (redex-match Egglog
                Program
                (term
                 ((let a 2) a))))

  (check-not-false
   (redex-match
    Egglog+Database
    Program+Database
    (term
     ((skip) (() ())))))

  (check-not-false
   (redex-match
    Egglog+Database
    Command+Database
    (term
     ((Add 2 3) (() ())))))


  (check-equal?
   (apply-reduction-relation*
    Command-Reduction
    (term
     ((Add 2 3) (() ()))))
   (list (term (skip (() ())))))

  (check-equal?
   (term (Eval-Expr 2 empty-env))
   (term (2 ((2) ()))))
  (check-equal?
   (term (Eval-Expr (Add 2 3) empty-env))
   (term ((Add 2 3) (((Add 2 3) 2 3) ()))))
  )