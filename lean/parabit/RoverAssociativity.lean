import Egglog.Examples.ParaBitInput

namespace BWLang.Certificates.Case_add_assoc_4

def problem : BWLang.Problem := {
  lhs := .app "bw" [.symbol "t", .app "+" [.app "bw" [.symbol "u", .app "+" [.app "bw" [.symbol "p", .symbol "a"], .app "bw" [.symbol "r", .symbol "b"]]], .app "bw" [.symbol "s", .symbol "c"]]]
  rhs := .app "bw" [.symbol "t", .app "+" [.app "bw" [.symbol "p", .symbol "a"], .app "bw" [.symbol "q", .app "+" [.app "bw" [.symbol "r", .symbol "b"], .app "bw" [.symbol "s", .symbol "c"]]]]]
  conditions := [.app "<" [.symbol "r", .symbol "q"], .app "<" [.symbol "s", .symbol "q"], .app "<" [.symbol "p", .symbol "u"], .app "<" [.symbol "r", .symbol "u"]]
  positiveWidths := [.symbol "t", .symbol "u", .symbol "p", .symbol "r", .symbol "s", .symbol "q"]
}

set_option maxRecDepth 100000 in
set_option maxHeartbeats 100000000 in
theorem equivalence : ∀ (ρ : String → Int), problem.Holds ρ →
    problem.lhs.eval ρ = problem.rhs.eval ρ :=
  ((problem.replay? 3 (ParaBit.ReplayHint.ground
  [0, 1, 2, 3, 5, 9, 11, 14, 16, 18, 21, 127, 149, 166, 169]
  [{ rule := 1, bindings := [(1, Egglog.Pat.app 53 []), (0, Egglog.Pat.app 50 [])] },
   { rule := 1, bindings := [(1, Egglog.Pat.app 47 []), (0, Egglog.Pat.app 44 [])] },
   { rule := 1, bindings := [(1, Egglog.Pat.app 41 []), (0, Egglog.Pat.app 38 [])] },
   { rule := 3,
     bindings := [(3, Egglog.Pat.app 50 []),
                  (2, Egglog.Pat.app 44 []),
                  (4, Egglog.Pat.app 56 []),
                  (1, Egglog.Pat.app 3 [Egglog.Pat.app 50 [], Egglog.Pat.app 53 []]),
                  (0, Egglog.Pat.app 3 [Egglog.Pat.app 44 [], Egglog.Pat.app 47 []])] },
   { rule := 3,
     bindings := [(3, Egglog.Pat.app 44 []),
                  (2, Egglog.Pat.app 38 []),
                  (4, Egglog.Pat.app 35 []),
                  (1, Egglog.Pat.app 3 [Egglog.Pat.app 44 [], Egglog.Pat.app 47 []]),
                  (0, Egglog.Pat.app 3 [Egglog.Pat.app 38 [], Egglog.Pat.app 41 []])] },
   { rule := 2,
     bindings := [(0,
                   Egglog.Pat.app
                     5
                     [Egglog.Pat.app 3 [Egglog.Pat.app 44 [], Egglog.Pat.app 47 []],
                      Egglog.Pat.app 3 [Egglog.Pat.app 50 [], Egglog.Pat.app 53 []]]),
                  (1, Egglog.Pat.app 56 [])] },
   { rule := 2,
     bindings := [(0,
                   Egglog.Pat.app
                     5
                     [Egglog.Pat.app 3 [Egglog.Pat.app 38 [], Egglog.Pat.app 41 []],
                      Egglog.Pat.app 3 [Egglog.Pat.app 44 [], Egglog.Pat.app 47 []]]),
                  (1, Egglog.Pat.app 35 [])] },
   { rule := 0,
     bindings := [(2, Egglog.Pat.app 3 [Egglog.Pat.app 50 [], Egglog.Pat.app 53 []]),
                  (1, Egglog.Pat.app 3 [Egglog.Pat.app 44 [], Egglog.Pat.app 47 []]),
                  (0, Egglog.Pat.app 3 [Egglog.Pat.app 38 [], Egglog.Pat.app 41 []])] }])).get (by decide +kernel)).down

end BWLang.Certificates.Case_add_assoc_4
