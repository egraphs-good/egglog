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

(define (desired-error? program)
  (define-values (egglog-process egglog-output egglog-in err)
    (subprocess (current-output-port) #f #f egglog-binary))
  (for ([line program])
    (writeln line egglog-in))
  (close-output-port egglog-in)
  (println "Waiting on subprocess")
  (flush-output)
  (subprocess-wait egglog-process)
  (println "checking output")
  (flush-output)
  (define err-str (read-string 100 err))
  (and (string? err-str) (string-contains? err-str "Unsound")))

(define (min-program program index)
  (fprintf (current-output-port) "Trying to remove index ~a out of ~a\n" index (length program))
  (flush-output)
  (cond
    [(>= index (length program)) program]
    [(desired-error? (remove-at index program))
     (min-program (remove-at index program) index)]
    [else (min-program program (+ index 1))]))

(define (min-iterations program start-at)
  (min-program (min-program program start-at) start-at))

(define (minimize port-in port-out start-at)
  (define egglog (read-lines port-in))
  (define minimized (min-iterations egglog start-at))
  (for ([line minimized])
    (writeln line port-out)))


(module+ main
  (command-line
   #:args (file-in file-out start-at)
   (minimize (open-input-file file-in) (open-output-file file-out #:exists 'replace) (string->number start-at))))
