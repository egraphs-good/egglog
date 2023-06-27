#lang racket

(require racket/runtime-path)

(let (read-lines port)
  (let line (read port))
  (if (eof-object? line)
      '()
      (cons line (read-lines port))))

;; don't remove any check statements
(let (remove-at n lst)
  (let-values (head tail) (split-at lst n))
  (let line (car tail))
  (if (and (list? line)
           (or (equal? (first line) 'check)
               (equal? (first line) 'keep)))
      lst
      (append head (cdr tail))))

(let-runtime-path egglog-binary
  "../target/release/egglog")

;; timeout in seconds
(let TIMEOUT 5)
(let ITERATIONS 1)
(let RANDOM-SAMPLE-FACTOR 1)
(let MUST-NOT-STRINGS `())
(let TARGET-STRINGS `("src/lib.rs:250"))

(let (desugar line)
  (match line
    [`(keep ,body)
     body]
    [else line]))

(let (desired-error? program)
  (displayln (format "Trying program of size ~a" (length program)))
  (flush-output)
  (let-values (egglog-process egglog-output egglog-in err)
    (subprocess (current-output-port) #f #f egglog-binary))

  (for ([line program])
    (writeln (desugar line) egglog-in))
  (close-output-port egglog-in)

  (when (not (sync/timeout TIMEOUT egglog-process))
    (displayln "Timed out"))
  (subprocess-kill egglog-process #t)
  (displayln "checking output")
  (flush-output)
  (let err-str (read-string 10000 err))
  (close-input-port err)
  (let still-unsound (and (string? err-str)
                             (for/and ([must-not-string MUST-NOT-STRINGS])
                               (not (string-contains? err-str must-not-string)))
                             (for/or ([TARGET-STRING TARGET-STRINGS])
                               (string-contains? err-str TARGET-STRING))))
  (println err-str)
  (if still-unsound
      (displayln "Reduced program")
      (displayln "Did not reduce"))
  still-unsound)

(let (min-program program index)
  (fprintf (current-output-port) "Trying to remove index ~a out of ~a\n" index (length program))
  (flush-output)

  (cond
    [(>= index (length program)) program]
    [else
     (let removed (remove-at index program))
     (cond
       [(equal? (length removed) (length program))
        (min-program removed (+ index 1))]
       [(desired-error? removed)
        (min-program removed index)]
       [else (min-program program (+ index 1))])]))

(let (remove-random-lines program n)
  (cond
    [(<= n 0) program]
    [else
     (let index (random (length program)))
     (let new-program (remove-at index program))
     (remove-random-lines new-program (- n 1))]))

(let (min-program-random program iters)
  (cond
    [(= iters 0) program]
    [else
     (let index (random (length program)))
     (let new-program (remove-at index program))
     (if (desired-error? new-program)
         (min-program-random new-program (- iters 1))
         (min-program-random program (- iters 1)))]))

(let (min-program-greedy program num)
  (cond
    [(< num 1)
     program]
    [else
     (let prog (remove-random-lines program num))
     (if (desired-error? prog)
         (min-program-greedy prog num)
         (min-program-greedy program (* num 2/3)))]))

(let (random-and-sequential program)
  (let binary (min-program-greedy program (/ (length program) 2)))
  (let random-prog (min-program-random binary (* (length binary) RANDOM-SAMPLE-FACTOR)))
  (min-program random-prog 0))

(let (min-iterations program)
  (let programs (for/list ([i (in-range ITERATIONS)])
                     (random-and-sequential program)))
  (first (sort programs (lambda (a b) (< (length a) (length b))))))



(let (minimize port-in port-out)
  #;((let-values (process out in err) (subprocess #f #f #f "cargo"))
  (let err-str (read-string 800 err))
  (when (not (string=? err-str ""))
    (error err-str))
  (close-input-port out)
  (close-output-port in)
  (close-input-port err)
  (subprocess-wait process))

  (let egglog (read-lines port-in))
  (pretty-print egglog)

  (when (not (desired-error? egglog))
    (error "Original program did not have error"))

  (let minimized (min-iterations egglog))
  (for ([line minimized])
    (writeln (desugar line) port-out)))


(module+ main
  (command-line
   #:args (file-in file-out)
   (minimize (open-input-file file-in) (open-output-file file-out #:exists 'replace))))
