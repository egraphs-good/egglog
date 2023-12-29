#lang racket

(require redex)
(require rackunit)
(require pict)

(provide (all-defined-out))

(define-language Egglog
  (Program
   (Cmd ...))
  (Cmd Action
       Rule
       (run) ;; TODO add schedules
       skip)
  (Rule
    (rule Query Actions))
  (Query (Pattern ...)) ;; Query is a list of Pattern
  (Pattern (= expr expr) ;; equality constraint
           expr)
  (Actions (Action ...))
  (Action expr
          (let var expr)
          (union expr expr))
  (expr number
        (constructor expr ...)
        var)
  (constructor variable-not-otherwise-mentioned)
  (var variable-not-otherwise-mentioned))

(define-extended-language
  Egglog+Database
  Egglog
  ;; Egglog maintains a Database as global state.
  ;; The database contains terms, a congruence
  ;; closure over those terms,
  ;; a set of global variable bindings,
  ;; and a set of rules (only run by run commands).
  ;; The environment is also used for
  ;; local variable bindings when
  ;; running rules.
  [Database (Terms Congr Env Rules)]
  [Congr
   (Eq ...)]
  [Eq
   (= Term Term)]
  [Terms
   (Term ...)]
  [Term number
        (constructor Term ...)]
  [Rules (Rule ...)]
  [Command+Database
   (Cmd Database)]
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
  [(Database-Union
    ((Term_i ...) (Eq_i ...) (Binding_i ...) (Rule_i ...))
    Database_i ...)
   ((Term_i ... Term_j ...)
    (Eq_i ... Eq_j ...)
    (Binding_i ... Binding_j ...)
    (Rule_i ... Rule_j ...))
   (where ((Term_j ...) (Eq_j ...) (Binding_j ...) (Rule_j ...))
          (Database-Union Database_i ...))])



(define-metafunction Egglog+Database
  Eval-Expr : expr Env -> (Term Database)
  [(Eval-Expr number Env)
   (number ((number) () () ()))]
  [(Eval-Expr (constructor expr_s ...) Env)
   ((constructor Term_c ...)
    (Database-Union
     (((constructor Term_c ...)) () () ())
     Database_c ...))
   (where ((Term_c Database_c) ...)
          ((Eval-Expr expr_s Env) ...))]
  [(Eval-Expr var Env)
   ((Lookup var Env) (((Lookup var Env)) () () ()))])


(define (dump-pict pict name)
  (send (pict->bitmap pict)
        save-file
        name
        'png))


(define-metafunction Egglog+Database
  Eval-Action : Action Database -> Database
  [(Eval-Action (let var expr) (Terms Congr (Binding_s ...) Rules))
   (Database-Union (Terms Congr ((var -> Term) Binding_s ...) Rules) Database_2)
   (where (Term Database_2)
          (Eval-Expr expr (Binding_s ...)))]
  [(Eval-Action expr (Terms Congr Env Rules))
   (Database-Union (Terms Congr Env Rules) Database_2)
   (where (Term Database_2)
          (Eval-Expr expr Env))]
  [(Eval-Action (union expr_1 expr_2) (Terms (Eq ...) Env Rules))
   (Database-Union
    (() ((= Term_1 Term_2) Eq ...) () ())
    Database_1
    Database_2)
   (where (Term_1 Database_1) (Eval-Expr expr_1 Env))
   (where (Term_2 Database_2) (Eval-Expr expr_2 Env))])


(define-metafunction Egglog+Database
  Add-Equality : Eq Congr -> Congr
  [(Add-Equality Eq (Eq_i ...))
   (Eq Eq_i ...)])

(define-metafunction
  Egglog+Database
  subset : Congr Congr -> ()
  [(subset () ())
   ()]
  [(subset (Eq_1 Eq_i ...) (Eq_j ... Eq_1 Eq_k ...))
   (subset (Eq_i ...) (Eq_j ... Eq_1 Eq_k ...))]
  [(subset (Eq_1 Eq_i ...) (Eq_j ...))
   #f
   (side-condition
    (not (member (term Eq_1) (term (Eq_j ...)))))])

;; A reduction relation that restores the congruence
;; relation after some equalities have been added
;; TODO would be much cleaner if we had sets
;; in our terms somehow
(define Congruence-Reduction
  (reduction-relation
   Egglog+Database
   #:domain Database
   (-->
    ((Term_i ... Term_j Term_k ...) Congr Env Rules)
    ((Term_i ... Term_j Term_k ...)
     (Add-Equality (= Term_j Term_j) Congr)
     Env
     Rules)
    (side-condition
     (not
      (member (term (= Term_j Term_j))
              (term Congr))))
    "reflexivity")
   (-->
    (Terms Congr Env Rules)
    (Terms (Add-Equality (= Term_2 Term_1) Congr) Env Rules)
    (where (Eq_i ... (= Term_1 Term_2) Eq_j ...) Congr)
    (side-condition
     (not
      (member (term (= Term_2 Term_1))
              (term Congr))))
    "symmetry"
    )
   (-->
    (Terms Congr Env Rules)
    (Terms (Add-Equality (= Term_1 Term_3) Congr) Env Rules)
    (where
     (Eq_i ... (= Term_1 Term_2)
      Eq_j ... (= Term_2 Term_3)
      Eq_k ...)
     Congr)
    (side-condition
     (not
      (member (term (= Term_1 Term_3))
              (term Congr))))
    "transitivity"
    )
   (-->
    (Terms Congr Env Rules)
    (Terms (Add-Equality
            (= (constructor Term_i ...)
               (constructor Term_j ...))
            Congr) Env Rules)
    (where (Term_k ... (constructor Term_i ...)
            Term_l ... (constructor Term_j ...)
            Term_m ...)
           Terms)
    (where
     (subset ((= Term_i Term_j) ...)
             Congr)
     ())
    (side-condition
     (not
      (member (term (= (constructor Term_i ...)
                       (constructor Term_j ...)))
              (term Congr))))
    )))

;; Like apply-reduction-relation*, but only
;; returns the first path found.
;; Instead of returning a list, it returns
;; a single term.
(define (apply-reduction-relation-one-path relation term)
  (match (apply-reduction-relation relation term)
    [`(,reduced ,others ...)
     (apply-reduction-relation-one-path relation reduced)]
    [`()
     term]))

(define (restore-congruence database-term)
  (apply-reduction-relation-one-path Congruence-Reduction database-term))

(define Command-Reduction
  (reduction-relation
   Egglog+Database
   #:domain Command+Database
   ;; running actions
   (-->
    (Action Database)
    (skip (Eval-Action Action Database)))
   (-->
    (Rule (Terms Congr Env (Rule_i ...)))
    (skip (Terms Congr Env (Rule Rule_i ...))))))




(define (try-apply-reduction-relation relation term)
  (define lst (apply-reduction-relation relation term))
  (match lst
    [`(,x) x]
    [`() #f]
    [_ (error "Expected single element, got ~a" lst)]))

(define Egglog-Reduction
  (reduction-relation
   Egglog+Database
   #:domain (Program Database)
   (-->
    ((Cmd_1 Cmd_s ...)
     Database)
    ((Cmd_stepped Cmd_s ...)
     ,(restore-congruence (term Database_2)))
    (where
     (Cmd_stepped Database_2)
     ,(try-apply-reduction-relation
       Command-Reduction
       (term (Cmd_1 Database)))))
   (-->
    ((skip Cmd_s ...)
     Database)
    ((Cmd_s ...)
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
  #:contract (typed-query-expr expr TypeEnv TypeEnv)
  #:mode (typed-query-expr I I O)
  [----------------------------
   (typed-query-expr number TypeEnv TypeEnv)]
  ;; no need to add it, refers to global
  ;; or previously defined
  [(side-condition ,(member (term (var : no-type))
                            (term TypeEnv)))
   ----------------------------
   (typed-query-expr var TypeEnv TypeEnv)]
  ;; add it to environment if not bound 
  [(side-condition ,(not
                     (member (term
                              (var : no-type))
                             (term
                              (TypeBinding ...)))))
   ----------------------------
   (typed-query-expr var
                     (TypeBinding ...)
                     ((var : no-type)
                      TypeBinding ...))]
  [(typed-query-expr expr_s TypeEnv (TypeBinding ...)) ...
   ----------------------------
   (typed-query-expr
    (constructor expr_s ...)
    TypeEnv
    (TypeBinding ... ...))])


(define-judgment-form
  Egglog+Database
  ;; Types an Action in a given environment and
  ;; returns the resulting environment
  #:contract (typed-action Action TypeEnv TypeEnv)
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
   (typed-action expr TypeEnv TypeEnv)]
  [(typed-expr expr_1 TypeEnv)
   (typed-expr expr_2 TypeEnv)
   ----------------------------
   (typed-action (union expr_1 expr_2) TypeEnv TypeEnv)])

(define-judgment-form
  Egglog+Database
  #:contract (typed-pattern Pattern TypeEnv TypeEnv)
  #:mode (typed-pattern I I O)
  [(typed-query-expr expr TypeEnv TypeEnv_2)
    ----------------------------
   (typed-pattern expr TypeEnv TypeEnv_2)]
  [(typed-query-expr expr_1 TypeEnv TypeEnv_2)
   (typed-query-expr expr_2 TypeEnv_2 TypeEnv_3)
   ----------------------------
   (typed-pattern (= expr_1 expr_2)
                  TypeEnv TypeEnv_3)])

(define-judgment-form
  Egglog+Database
  #:contract (typed-query Query TypeEnv TypeEnv)
  #:mode (typed-query I I O)
  [------------------------
   (typed-query () TypeEnv TypeEnv)]
  [(typed-query (Pattern_i ...) TypeEnv TypeEnv_2)
   (typed-pattern Pattern TypeEnv_2 TypeEnv_3)
   ------------------------
   (typed-query (Pattern_i ... Pattern) TypeEnv TypeEnv_3)])

(define-judgment-form
  Egglog+Database
  #:contract (typed-actions Actions TypeEnv TypeEnv)
  #:mode (typed-actions I I O)
  [------------------------
   (typed-actions () TypeEnv TypeEnv)]
  [(typed-actions (Action_i ...) TypeEnv TypeEnv_2)
   (typed-action Action TypeEnv_2 TypeEnv_3)
   ------------------------
   (typed-actions
    (Action_i ... Action)
    TypeEnv TypeEnv_3)])

(define-judgment-form
  Egglog+Database
  #:contract (typed-rule Rule TypeEnv)
  #:mode (typed-rule I I)
  [(typed-query Query TypeEnv TypeEnv_2)
   (typed-actions Actions TypeEnv_2 TypeEnv_3)
   ----------------------------
   (typed-rule (rule Query Actions) TypeEnv)])

(define-judgment-form
  Egglog+Database
  #:contract (typed-program Program TypeEnv)
  #:mode (typed-program I O)
  [---------------------------
   (typed-program () ())]
  [(typed-program (Cmd_p ...) TypeEnv)
   (typed-action Action TypeEnv TypeEnv_res)
   ---------------------------
   (typed-program (Cmd_p ... Action) TypeEnv_res)]
  [(typed-program (Cmd_p ...) TypeEnv)
   (typed-rule Rule TypeEnv)
   ---------------------------
   (typed-program (Cmd_p ... Rule) TypeEnv)])


(define-syntax-rule (save-metafunction func ...)
  (begin
    (dump-pict (render-metafunction func)
               (format "~a.png" 'func))
    ...
    (void)))


(define (save-semantics)
  (save-metafunction Eval-Expr Lookup Database-Union Eval-Action))

(save-semantics)




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

