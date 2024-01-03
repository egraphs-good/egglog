#lang racket

(require redex)
(require pict)

(provide (all-defined-out))


;; here's how to set the style of everything
;; for example, modern (monospace) style:
#;((default-style 'modern)
(grammar-style 'modern)
(label-style 'modern)
(paren-style 'modern)
(literal-style (cons 'bold 'modern))
(metafunction-style 'modern)
(non-terminal-style 'modern)
(non-terminal-subscript-style 'modern)
(non-terminal-superscript-style 'modern))

(define-language Egglog
  [Program
   (Cmd ...)]
  [Cmd Action
       Rule
       (run) ;; TODO add schedules
       skip]
  [Rule
    (rule Query Actions)]
  [Query (Pattern ...)] ;; Query is a list of Pattern
  [Pattern (= expr expr) ;; equality constraint
           expr]
  [Actions (Action ...)]
  [Action expr
          (let var expr)
          (union expr expr)]
  [expr number
        (constructor expr ...)
        var]
  [constructor variable-not-otherwise-mentioned]
  [var variable-not-otherwise-mentioned]
  [ReservedSymbol -> :]) ;; variable-not-otherwise-mentioned needs to not use these since we use them in Database

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
   (congr Eq ...)]
  [Eq
   (= Term Term)]
  [Terms
   (tset Term ...)]
  [Term number
        (constructor Term ...)]
  [Rules (Rule ...)]
  [Command+Database
   (Cmd Database)]
  [Envs (Env ...)]
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
  Lookup : var Env -> Term ∨ #f
  [(Lookup var_1 ())
   #f]
  [(Lookup var_1 ((var_1 -> Term_1) (var_s -> Term_s) ...))
   Term_1]
  [(Lookup var_1 ((var_2 -> Term_1) (var_s -> Term_s) ...))
   (Lookup var_1 ((var_s -> Term_s) ...))
   (side-condition (not (equal? (term var_1)
                                (term var_2))))])


(define-metafunction Egglog+Database
  Dedup : Database -> Database
  [(Dedup
    ((tset Term_i ...)
     (congr Eq_i ...)
     (Binding_i ...)
     (Rule_i ...)))
   (,(cons 'tset (remove-duplicates (term (Term_i ...))))
    ,(cons 'congr (remove-duplicates (term (Eq_i ...))))
    ,(remove-duplicates (term (Binding_i ...)))
    ,(remove-duplicates (term (Rule_i ...))))])

(define-metafunction Egglog+Database
  Database-Union : Database ... -> Database
  [(Database-Union)
   ((tset) (congr) () ())]
  [(Database-Union
    ((tset Term_i ...)
     (congr Eq_i ...)
     (Binding_i ...)
     (Rule_i ...)) ...)
   (Dedup
    ((tset Term_i ... ...)
     (congr Eq_i ... ...)
     (Binding_i ... ...)
     (Rule_i ... ...)))])


(define-metafunction Egglog+Database
 tset-union : Terms ... -> Terms
 [(tset-union (tset Term_i ...) ...)
  (tset Term_i ... ...)])

(define-judgment-form Egglog+Database
  #:contract (tset-element Term Terms)
  #:mode (tset-element O I)
  [----------------------
   (tset-element Term_a (tset Term_i ... Term_a Term_j ...))])
  

(define-metafunction Egglog+Database
  Eval-Expr : expr Env -> Term
  [(Eval-Expr number Env)
   number]
  [(Eval-Expr var Env)
   (Lookup var Env)]
  [(Eval-Expr (constructor expr_s ...) Env)
   (constructor Term_c ...)
   (where (Term_c ...)
          ((Eval-Expr expr_s Env) ...))])


(define (dump-pict pict name)
  (send (pict->bitmap pict)
        save-file
        name
        'png))

(define-metafunction Egglog+Database
  Eval-Action : Action Database -> Database
  [(Eval-Action (let var expr) (Terms Congr (Binding_s ...) Rules_1))
   ((tset-union (tset Term_res) Terms) Congr ((var -> Term_res) Binding_s ...) Rules_1)
   (where Term_res
          (Eval-Expr expr (Binding_s ...)))]
  [(Eval-Action expr (Terms Congr Env Rules_1))
   ((tset-union (tset Term_res) Terms) Congr Env Rules_1)
   (where Term_res
          (Eval-Expr expr Env))]
  [(Eval-Action (union expr_1 expr_2) (Terms_1 (congr Eq ...) Env_1 Rules_1))
   ((tset-union (tset Term_1 Term_2) Terms_1) (congr (= Term_1 Term_2) Eq ...) Env_1 Rules_1)
   (where Term_1 (Eval-Expr expr_1 Env_1))
   (where Term_2 (Eval-Expr expr_2 Env_1))])


(define-metafunction Egglog+Database
  Add-Equality : Eq Congr -> Congr
  [(Add-Equality Eq (congr Eq_i ...))
   (congr Eq Eq_i ...)])

(define-metafunction
  Egglog+Database
  subset : Congr Congr -> #t ∨ #f
  [(subset (congr) (congr Eq_i ...))
   #t]
  [(subset (congr Eq_1 Eq_i ...) (congr Eq_j ... Eq_1 Eq_k ...))
   (subset (congr Eq_i ...) (congr Eq_j ... Eq_1 Eq_k ...))]
  [(subset (congr Eq_1 Eq_i ...) (congr Eq_j ...))
   #f
   (side-condition
    (not (member (term Eq_1) (term (Eq_j ...)))))])

(define-metafunction
  Egglog+Database
  tset-subset : Terms Terms -> #t ∨ #f
  [(tset-subset (tset Term_i ...) (tset Term_j ...))
   ,(subset? (list->set (term (Term_i ...)))
             (list->set (term (Term_j ...))))])

;; A reduction relation that restores the congruence
;; relation after some equalities have been added
(define Congruence-Reduction
  (reduction-relation
   Egglog+Database
   #:domain Database
   (-->
    ((tset Term_i ... Term_j Term_k ...) Congr Env Rules_1)
    ((tset Term_i ... Term_j Term_k ...)
     (Add-Equality (= Term_j Term_j) Congr)
     Env
     Rules_1)
    (side-condition/hidden ;; side condition so redex terminates
     (not
      (member (term (= Term_j Term_j))
              (term Congr))))
    "reflexivity")
   (-->
    (Terms Congr Env Rules_1)
    (Terms (Add-Equality (= Term_2 Term_1) Congr) Env Rules_1)
    (where (congr Eq_i ... (= Term_1 Term_2) Eq_j ...) Congr)
    (side-condition/hidden
     (not
      (member (term (= Term_2 Term_1))
              (term Congr))))
    "symmetry"
    )
   (-->
    (Terms Congr Env Rules_1)
    (Terms (Add-Equality (= Term_1 Term_3) Congr) Env Rules_1)
    (where
     (congr Eq_i ... (= Term_1 Term_2)
      Eq_j ... (= Term_2 Term_3)
      Eq_k ...)
     Congr)
    (side-condition/hidden
     (not
      (member (term (= Term_1 Term_3))
              (term Congr))))
    "transitivity"
    )
   (-->
    (Terms Congr Env Rules_1)
    (Terms (Add-Equality
            (= (constructor Term_i ...)
               (constructor Term_j ...))
            Congr) Env Rules_1)
    (where (Term_k ... (constructor Term_i ...)
            Term_l ... (constructor Term_j ...)
            Term_m ...)
           Terms)
    (where
     (subset (congr (= Term_i Term_j) ...)
             Congr)
     ())
    (side-condition/hidden
     (not
      (member (term (= (constructor Term_i ...)
                       (constructor Term_j ...)))
              (term Congr))))
    "congruence"
    )
   (-->
    (Terms Congr Env Rules)
    ((tset-union (tset Term_c) Terms)
     Congr Env Rules)
    (judgment-holds
     (tset-element (constructor Term_i ... Term_c Term_j ...) Terms))
    (side-condition/hidden
     (not (member (term Term_c) (term Terms))))
    "presence of children"
     )
    ))

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

(define-metafunction
  Egglog+Database
  restore-congruence : Database -> Database
  [(restore-congruence Database)
   ,(apply-reduction-relation-one-path Congruence-Reduction (term Database))])

(define-metafunction
  Egglog+Database
  Eval-Global-Actions : Actions Database -> Database
  [(Eval-Global-Actions () Database)
   Database]
  [(Eval-Global-Actions (Action_1 Action_i ...)
                 Database_1)
   (Eval-Global-Actions (Action_i ...)
                 (Eval-Action Action_1 Database_1))])

(define-metafunction
  Egglog+Database
  Eval-Local-Actions : Actions Database Env -> Database
  [(Eval-Local-Actions Actions
                       (Terms_1 Congr_1 Env_1 Rules_1)
                       Env_local)
   (Terms_2 Congr_2 Env_1 Rules_1)
   (where (Terms_2 Congr_2 Env_2 Rules_2)
          (Eval-Global-Actions
           Actions
           (Terms_1 Congr_1
            (Env-Union Env_1 Env_local)
            Rules_1)))])
                           
  

(define-metafunction
  Egglog+Database
  Eval-Rule-Actions : Rule Database Envs -> Database
  [(Eval-Rule-Actions
    (rule Query Actions)
    Database
    (Env_i ...))
   (Database-Union
    (Eval-Local-Actions Actions Database Env_i)
    ...)])

(define-metafunction Egglog+Database
  unbound : var Env -> #t ∨ #f
  [(unbound var_1 Env)
   ,(not (member (term var_1)
                 (map first (term Env))))])

;; Returns #f if the environments are 
;; inconsistent
(define-metafunction
  Egglog+Database
  Env-Union : Env ... -> Env
  [(Env-Union)
   ()]
  [(Env-Union Env)
   Env]
  [(Env-Union Env_1 Env_2 Env_i ...)
   (Env-Union (Env-Union2 Env_1 Env_2) Env_i ...)])

(define-metafunction
  Egglog+Database
  Env-Union2 : Env Env -> Env
  [(Env-Union2 () Env)
   Env]
  [(Env-Union2 ((var -> Term) Binding_i ...) Env)
   ((var -> Term) Binding_r ...)
   (where (Binding_r ...)
          (Env-Union2 (Binding_i ...) Env))
   (side-condition
    (or (equal? (term (Lookup var Env))
                (term Term))
        (term (unbound var Env))))]
  [(Env-Union2 ((var -> Term) Binding_i ...) Env)
   #f
   (side-condition
     (not (or (equal? (term (Lookup var Env))
                      (term Term))
              (term (unbound var Env)))))])

(define-metafunction
  Egglog+Database
  free-vars : expr -> (var ...)
  [(free-vars number)
   ()]
  [(free-vars (constructor expr_i ...))
   (free-vars expr_i ...)]
  [(free-vars var)
   (var)])


;; For a database, pattern, term, and environment,
;; valid-subst judges that the pattern e-matches
;; the term with local substitution given by the environment.
;; `valid-subst` defines e-matching by specifying
;; which environments satisfy a query.
(define-judgment-form
  Egglog+Database
  #:contract (valid-subst Database Pattern Term Env)
  #:mode (valid-subst I I O O)
  [(where #t (unbound var Env))
   ----------------------------
   (valid-subst ((tset Term_i ... Term_1 Term_j ...) Congr Env Rules)
                var
                Term_1
                ((var -> Term_1)))]
  [-----------------------------
   (valid-subst
    (Terms Congr (Binding_i ... (var -> Term_1) Binding_j ...) Rules)
    var
    Term_1
    ())]
  [-----------------------------
   (valid-subst Database number number ())]
  [(valid-subst Database Pattern_i Term_i Env_i) ...
   (where ((tset Term_x ... 
     (constructor Term_j ...)
     Term_z ...) Congr Env Rules) Database)
   (where #t (subset (congr (= Term_i Term_j) ...) Congr))
   (where Env_r (Env-Union Env_i ...))
   -----------------------------
   (valid-subst
    Database
    (constructor Pattern_i ...)
    (constructor Term_j ...)
    Env_r)])


;; If a local environment is valid
;; for a set of patterns, it is valid for
;; the overall query.
(define-judgment-form
  Egglog+Database
  #:contract (valid-query-subst Database Query Env)
  #:mode (valid-query-subst I I O)
  [(valid-subst Database Pattern_i Term_i Env_i) ...
   ---------------------------
   (valid-query-subst Database (Pattern_i ...) (Env-Union Env_i ...))])


;; Performs the rule's query, returning a set
;; of environments that satisfy the query
(define-metafunction
  Egglog+Database
  Rule-Envs : Database Rule -> Envs
  [(Rule-Envs Database (rule Query Actions))
   ;; Perform e-matching using redex
   ,(judgment-holds (valid-query-subst Database Query Env) Env)])

(define -->_Command (render-term Egglog -->_Command))
(set-arrow-pict! '-->_Command
 (lambda () -->_Command))
(define Command-Reduction
  (reduction-relation
   Egglog+Database
   #:domain Command+Database
   #:arrow -->_Command
   ;; running actions
   (-->_Command
    (Action Database)
    (skip (Eval-Action Action Database)))
   (-->_Command
    (Rule (Terms Congr Env (Rule_i ...)))
    (skip (Terms Congr Env (Rule Rule_i ...))))
   (-->_Command
    ((run) Database)
    (skip (Database-Union Database Database_i ...))
    (where (Terms Congr Env (Rule ...))
           Database)
    (where (Envs ...)
           ((Rule-Envs Database Rule) ...))
    (where (Database_i ...)
           ((Eval-Rule-Actions Rule Database Envs) ...)))))


(define (try-apply-reduction-relation relation term)
  (define lst (apply-reduction-relation relation term))
  (match lst
    [`(,x) x]
    [`() #f]
    [_ (error "Expected single element, got ~a" lst)]))


(define -->_Program (render-term Egglog -->_Program))
(set-arrow-pict! '-->_Program
 (lambda () -->_Program))
(define Egglog-Reduction
  (reduction-relation
   Egglog+Database
   #:domain (Program Database)
   #:arrow -->_Program
   (-->_Program
    ((Cmd_1 Cmd_s ...) Database)
    ((Cmd_stepped Cmd_s ...) (restore-congruence Database_2))
    (where/hidden ;; hide this from typesetting
     (Cmd_stepped Database_2)
     ,(try-apply-reduction-relation
       Command-Reduction
       (term (Cmd_1 Database))))
    (side-condition #t) ;; use this instead
       
       )
   (-->_Program
    ((skip Cmd_s ...) Database)
    ((Cmd_s ...) Database))))


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
   (typed-program (Cmd_p ... Rule) TypeEnv)]
  [(typed-program (Cmd_p ...) TypeEnv)
   ---------------------------
   (typed-program (Cmd_p ... (run)) TypeEnv)])

;; TODO
;; Add type checking?
;;   -Perhaps typed ast should be separate for simplicity
;; Schedules
;; Merge functions, on_merge
;; Seminaive evaluation- egglog actually finds a subset of the valid substitutions
;; Extraction meaning and guarantees
;; Set-opion
;; Containers?
;; Subsumption

