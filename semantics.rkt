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
  [Command+Database
   (cmd Database)]
  [Env (Binding ...)]
  [Binding (var -> Term)]
  [TypeEnv (TypeBinding ...)]
  [TypeBinding (var : Type)]
  ;; TODO add types- right now we just
  ;; check there are no free variables
  [Type no-type])

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
  [(Database-Union ((Term_s ...) (Eq_s ...) (Binding_s ...)) Database_s ...)
   ((Term_s ... Term_s2 ...) (Eq_s ... Eq_s2 ...) (Binding_s ... Binding_s2 ...))
   (where ((Term_s2 ...) (Eq_s2 ...) (Binding_s2 ...))
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
          ((Eval-Expr expr_s Env) ...))]
  [(Eval-Expr var Env)
   ((Lookup var Env) (((Lookup var Env)) () ()))])


(define (dump-pict pict name)
  (send (pict->bitmap pict)
        save-file
        name
        'png))


(define-metafunction Egglog+Database
    Eval-Action : action Database -> Database
    [(Eval-Action (let var expr) (Terms Congr (Binding_s ...)))
     (Database-Union (Terms Congr ((var -> Term) Binding_s ...)) Database_2)
     (where (Term Database_2)
            (Eval-Expr expr (Binding_s ...)))]
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
    (skip (Eval-Action action Database)))))

(define (single-element-or-false lst)
  (match lst
    [`(,x) x]
    [`() #f]
    [_ (error "Expected single element, got ~a" lst)]))

(define Egglog-Reduction
  (reduction-relation
   Egglog+Database
   #:domain (Program Database)
   (-->
    ((cmd_1 cmd_s ...)
     Database)
    ((cmd_stepped cmd_s ...)
     Database_2)
    (where
     (cmd_stepped Database_2)
     ,(single-element-or-false
       (apply-reduction-relation
        Command-Reduction
        (term (cmd_1 Database))))))
   (-->
    ((skip cmd_s ...)
     Database)
    ((cmd_s ...)
     Database))))


(define-judgment-form
  Egglog+Database
  #:contract (typed-expr expr TypeEnv)
  #:mode (typed-expr I I)
  [----------------------------
   (typed-expr number TypeEnv)]
  [(side-condition ,(member (term (var : no-type))
                           (term TypeEnv)))
   ----------------------------
   (typed-expr var TypeEnv)]
  [(typed-expr expr_s TypeEnv) ...
   ----------------------------
   (typed-expr (constructor expr_s ...) TypeEnv)])


(define-judgment-form
  Egglog+Database
  ;; Types an action in a given environment and
  ;; returns the resulting environment
  #:contract (typed-action action TypeEnv TypeEnv)
  #:mode (typed-action I I O)
  [(side-condition ,(not
                     (member 
                      (term (TypeBinding ...))
                      (term (TypeBinding ...)))))
   (typed-expr expr (TypeBinding ...))
   ----------------------------
   (typed-action (let var expr)
                 (TypeBinding ...)
                 ((var : no-type)
                  TypeBinding ...))]
  [(typed-expr expr TypeEnv)
   ----------------------------
   (typed-action expr TypeEnv TypeEnv)])


(define-judgment-form
  Egglog+Database
  #:contract (typed Program TypeEnv)
  #:mode (typed I O)
  [---------------------------
   (typed () ())]
  [(typed (cmd_p ...) TypeEnv)
   (typed-action action TypeEnv TypeEnv_res)
   ---------------------------
   (typed (cmd_p ... action) TypeEnv_res)])

(define (types? e)
  (judgment-holds (typed ,e TypeEnv)))

(define (executes? prog)
  (cond
    [(not (types? prog))
     #t]
    [else
     (define res
      (apply-reduction-relation* Egglog-Reduction (term (,prog (() () ())))))
     (match res
      [`((() ,database))
        #t]
      [_ #f])]))


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

  (redex-check Egglog
    Program
    (executes? (term Program))
    #:attempts 100000)
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

