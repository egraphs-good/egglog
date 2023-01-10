#lang racket

(require racket/runtime-path)

(define (read-lines port)
  (define line (read port))
  (if (eof-object? line)
      '()
      (cons line (read-lines port))))

(define (remove-at n lst)
  (define-values (head tail) (split-at lst n))
  (append head (cdr tail)))

(define-runtime-path egglog-binary
  "target/release/egg-smol")

;; timeout in seconds
(define TIMEOUT 5)
(define ITERATIONS 1)
(define RANDOM-SAMPLE-FACTOR 2)

(define (desired-error? program)
  (displayln (format "Trying program of size ~a" (length program)))
  (define-values (egglog-process egglog-output egglog-in err)
    (subprocess #f #f #f egglog-binary))
  (close-input-port egglog-output)
  (for ([line program])
    (writeln line egglog-in))
  (close-output-port egglog-in)
  (flush-output)
  (when (not (sync/timeout TIMEOUT egglog-process))
    (displayln "Timed out"))
  (subprocess-kill egglog-process #t)
  (displayln "checking output")
  (flush-output)
  (define err-str (read-string 100 err))
  (close-input-port err)
  (define still-unsound (and (string? err-str) (string-contains? err-str "Unsound")))
  (if still-unsound
      (displayln "Reduced program")
      (displayln "Did not reduce"))
  still-unsound)

(define (min-program program index)
  (fprintf (current-output-port) "Trying to remove index ~a out of ~a\n" index (length program))
  (flush-output)
  (cond
    [(>= index (length program)) program]
    [(desired-error? (remove-at index program))
     (min-program (remove-at index program) index)]
    [else (min-program program (+ index 1))]))

(define (min-program-random program iters)
  (cond
    [(= iters 0) program]
    [else
     (define index (random (length program)))
     (define new-program (remove-at index program))
     (if (desired-error? new-program)
         (min-program-random new-program (- iters 1))
         (min-program-random program (- iters 1)))]))

(define (random-and-sequential program)
  (define random-prog (min-program-random program (* (length program) RANDOM-SAMPLE-FACTOR)))
  (min-program random-prog 0))

(define (min-iterations program)
  (define programs (for/list ([i (in-range ITERATIONS)])
                     (random-and-sequential program)))
  (first (sort programs (lambda (a b) (< (length a) (length b))))))

(define (minimize port-in port-out)
  (define egglog (read-lines port-in))
  (define minimized (min-iterations egglog))
  (for ([line minimized])
    (writeln line port-out)))


(module+ main
  (command-line
   #:args (file-in file-out)
   (minimize (open-input-file file-in) (open-output-file file-out #:exists 'replace))))
