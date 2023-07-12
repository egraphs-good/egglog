#lang racket

(require racket/runtime-path)

(define (read-lines port)
  (define line (read port))
  (if (eof-object? line)
      '()
      (cons line (read-lines port))))

;; don't remove any check statements
(define (remove-at n lst)
  (define-values (head tail) (split-at lst n))
  (define line (car tail))
  (if (and (list? line)
           (or (equal? (first line) 'check)
               (equal? (first line) 'keep)))
      lst
      (append head (cdr tail))))

(define-runtime-path egglog-binary
  "../target/release/egglog")

;; timeout in seconds
(define TIMEOUT 5)
(define ITERATIONS 1)
(define RANDOM-SAMPLE-FACTOR 1)
(define MUST-NOT-STRINGS `())
(define TARGET-STRINGS `("invalid default for"))

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

  (displayln "(" egglog-in)
  (for ([line program])
    (writeln (desugar line) egglog-in))
  (displayln ")" egglog-in)
  (close-output-port egglog-in)

  (when (not (sync/timeout TIMEOUT egglog-process))
    (displayln "Timed out"))
  (subprocess-kill egglog-process #t)
  (displayln "checking output")
  (flush-output)
  (define err-str (read-string 800 err))
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
       [(equal? (length removed) (length program))
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
         (min-program-greedy program (* num 2/3)))]))

(define (random-and-sequential program)
  (define binary (min-program-greedy program (/ (length program) 2)))
  (define random-prog (min-program-random binary (* (length binary) RANDOM-SAMPLE-FACTOR)))
  (min-program random-prog 0))

(define (min-iterations program)
  (define programs (for/list ([i (in-range ITERATIONS)])
                     (random-and-sequential program)))
  (first (sort programs (lambda (a b) (< (length a) (length b))))))

(define (minimize port-in port-out)
  (define egglog (read-lines port-in))
  (pretty-print egglog)

  (when (not (desired-error? egglog))
    (error "Original program did not have error"))

  (define minimized (min-iterations egglog))
  (for ([line minimized])
    (writeln (desugar line) port-out)))


(module+ main
  (command-line
   #:args (file-in file-out)
   (minimize (open-input-file file-in) (open-output-file file-out #:exists 'replace))))
