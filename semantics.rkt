#lang racket

(require redex)
(require rackunit)
(require pict)


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
  [Database (Terms Congr Env)]
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
  [Env (Assign ...)]
  [Assign (var -> Term)]
  )

;; No rule for if the variable is not found
;; Therefore it fails in such cases
(define-metafunction Egglog+Database
  Lookup : var Env -> Term
  [(Lookup var ((var -> Term) (var_s -> Term_s) ...))
   Term]
  [(Lookup var_1 ((var_2 -> Term) (var_s -> Term_s) ...))
   (Lookup var ((var_s -> Term_s) ...))
   (side-condition (not (equal? (term var_1)
                                (term var_2))))])

(define-metafunction Egglog+Database
  Database-Union : Database ... -> Database
  [(Database-Union Database)
   Database]
  [(Database-Union ((Term_s ...) (Eq_s ...) (Assign_s ...)) Database_s ...)
   ((Term_s ... Term_s2 ...) (Eq_s ... Eq_s2 ...) (Assign_s ... Assign_s2 ...))
   (where ((Term_s2 ...) (Eq_s2 ...) (Assign_s2 ...))
          (Database-Union Database_s ...))])



(define-metafunction Egglog+Database
  Eval-Expr : expr Env -> (Term Database)
  [(Eval-Expr number Env)
   (number ((number) () ()))]
  [(Eval-Expr (constructor expr_s ...) Env)
   ((constructor Term_c ...)
        (Database-Union
         (((constructor Term_c ...)) () ())
         Database_c ...))
   (where ((Term_c Database_c) ...)
          ((Eval-Expr expr_s Env) ...))])


(define (dump-pict pict name)
  (send (pict->bitmap pict)
        save-file
        name
        'png))


(define-metafunction Egglog+Database
    Eval-Action : action Database -> Database
    [(Eval-Action (let var expr) (Terms Congr Env))
     ((var -> Term)
      (Database-Union (Terms Congr Env) Database_2))
     (where (Term Database_2)
            (Eval-Expr expr Env))]
    [(Eval-Action expr (Terms Congr Env))
     (Database-Union (Terms Congr Env) Database_2)
     (where (Term Database_2)
            (Eval-Expr expr Env))])

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


(define-syntax-rule (save-metafunction func ...)
  (begin
   (dump-pict (render-metafunction func)
               (format "~a.png" 'func))
   ...
   (void)))


(define (save-semantics)
  (save-metafunction Eval-Expr Lookup Database-Union Eval-Action))

(save-semantics)

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
   (list (term (skip (() () ())))))

  (check-equal?
   (term (Eval-Expr 2 ()))
   (term (2 ((2) () ()))))
  (check-equal?
   (term (Eval-Expr (Add 2 3) ()))
   (term ((Add 2 3) (((Add 2 3) 2 3) () ()))))
  )


;; TODO
;; Make ast typed and add type checking?
;;    Perhaps typed ast should be separate for simplicity
;; Running rules, schedules
;; Merge functions, on_merge
;; Rebuilding semantics
;; Seminaive evaluation- which substitutions are allowed to be returned
;; Handle globals
;; Extraction meaning and guarantees
;; Set-opion

