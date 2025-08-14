"""
This is a simple script to generate an AC benchmark for egglog.
"""
N=13

print("""
(datatype Num (N i64) (Add Num Num))
; instead of (rewrite (Add a b) (Add b a))
(rule ((= e (Add a b))) ((set (Add b a) e)))
; instead of (rewrite (Add a (Add b c)) (Add (Add a b) c))
(rule ((= e (Add a (Add b c)))) ((set (Add (Add a b) c) e)))
""")

for i in range(N):
    print(f"(let _lit{i} (N {i}))")


var=0
print(f"(let _v{var} (Add _lit0 _lit1))")
for i in range(2, N):
    prev=var
    var += 1
    print(f"(let _v{var} (Add _lit{i} _v{prev}))")

rev=0
print(f"(let _r{rev} (Add _lit{N-1} _lit{N-2}))")
for i in range(N-2):
    prev=rev
    rev+=1
    print(f"(let _r{rev} (Add _r{prev} _lit{i}))")


print(f"(run 10000)")
print(f"(check (= _r{rev} _v{var}))")
