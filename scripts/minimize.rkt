#lang racket

(require racket/runtime-path)

;; timeout in seconds
(define TIMEOUT 5)
(define ITERATIONS 1)
(define RANDOM-SAMPLE-FACTOR 1)
(define MUST-NOT-STRINGS `())
(define TARGET-STRINGS `("Intersect failed"))
(define DONT-REMOVE (list->set `(set-option check keep define)))


(define should-transform-rewrites
  (make-parameter #f))

(define (read-lines port)
  (define line (read port))
  (if (eof-object? line)
      '()
      (cons line (read-lines port))))

;; don't remove any check statements
(define (remove-at n lst)
  (define-values (head tail) (split-at lst n))
  (define line (car tail))
  (cond
    [(and (list? line)
          (set-member? DONT-REMOVE (first line)))
      lst]
    [(should-transform-rewrites)
     (define new-line
      (match line
        [`(rewrite ,lhs ,rhs ,whatever ...)
          `(rule (,lhs) (,rhs) ,@whatever)]
        [else line]))
     (append head (cons new-line (cdr tail)))
    ]
    [else
      (append head (cdr tail))]))

(define-runtime-path egglog-binary
  "../target/release/egg-smol")


(define (desugar line)
  (match line
    [`(keep ,body)
     body]
    [else line]))

(define (desired-error? program)
  (displayln (format "Trying program of size ~a" (length program)))
  (flush-output)
  (define-values (egglog-process egglog-output egglog-in err)
    (subprocess (current-output-port) #f #f egglog-binary))

  (for ([line program])
    (writeln (desugar line) egglog-in))
  (close-output-port egglog-in)

  (when (not (sync/timeout TIMEOUT egglog-process))
    (displayln "Timed out"))
  (subprocess-kill egglog-process #t)
  (displayln "checking output")
  (flush-output)
  (define err-str (read-string 10000 err))
  (close-input-port err)
  (define still-unsound (and (string? err-str)
                             (for/and ([must-not-string MUST-NOT-STRINGS])
                               (not (string-contains? err-str must-not-string)))
                             (for/or ([TARGET-STRING TARGET-STRINGS])
                               (string-contains? err-str TARGET-STRING))))
  (println err-str)
  (if still-unsound
      (displayln "Reduced program")
      (displayln "Did not reduce"))
  still-unsound)

(define (min-program program index)
  (fprintf (current-output-port) "Trying to remove index ~a out of ~a\n" index (length program))
  (flush-output)

  (cond
    [(>= index (length program)) program]
    [else
     (define removed (remove-at index program))
     (cond
       [(and (equal? (length removed) (length program))
              (desired-error? removed))
        (min-program removed (+ index 1))]
       [(desired-error? removed)
        (min-program removed index)]
       [else (min-program program (+ index 1))])]))


(define (remove-random-lines program n)
  (cond
    [(<= n 0) program]
    [else
     (define index (random (length program)))
     (define new-program (remove-at index program))
     (remove-random-lines new-program (- n 1))]))

(define (min-program-random program iters)
  (cond
    [(= iters 0) program]
    [else
     (define index (random (length program)))
     (define new-program (remove-at index program))
     (if (desired-error? new-program)
         (min-program-random new-program (- iters 1))
         (min-program-random program (- iters 1)))]))

(define (min-program-greedy program num)
  (cond
    [(< num 1)
     program]
    [else
     (define prog (remove-random-lines program num))
     (if (desired-error? prog)
         (min-program-greedy prog num)
         (min-program-greedy program (* num 3/4)))]))

(define (random-and-sequential program)
  (define binary (min-program-greedy program (/ (length program) 2)))
  (define random-prog (min-program-random binary (* (length binary) RANDOM-SAMPLE-FACTOR)))
  (min-program random-prog 0))

(define (min-iterations program)
  (define programs (for/list ([i (in-range ITERATIONS)])
                     (random-and-sequential program)))
  (first (sort programs (lambda (a b) (< (length a) (length b))))))

(define (run-cmd cmd args)
  (define-values (sp out in err)
                 (apply subprocess #f #f #f cmd args))
  (printf "stdout:\n~a" (port->string out))
  (printf "stderr:\n~a" (port->string err))
  (close-input-port out)
  (close-output-port in)
  (close-input-port err)
  (subprocess-wait sp))

(define (minimize port-in port-out)
  ;; TODO how to not use absolute path here?
  (run-cmd "/Users/oflatt/.cargo/bin/cargo" (list  "build" "--release"))

  (define egglog (read-lines port-in))

  (when (not (desired-error? egglog))
    (error "Original program did not have error"))

  (define minimized (min-iterations egglog))
  (displayln "Attempting to transform rewrites...")
  (define transformed
    (parameterize ([should-transform-rewrites #t])
      (min-iterations minimized)))
  (for ([line transformed])
    (writeln (desugar line) port-out)))


(module+ main
  (command-line
   #:args (file-in file-out)
   (minimize (open-input-file file-in) (open-output-file file-out #:exists 'replace))))
