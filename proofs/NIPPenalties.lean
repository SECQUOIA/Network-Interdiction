module

public import Mathlib.Basic.Real.Basic
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.NormNum

public section

set_option autoImplicit false
set_option warningAsError true

namespace NIPPenalties

/-- Binary variables interpreted in the reals. -/
def bit (b : Bool) : ℝ := if b then 1 else 0

/-- The actual per-arc quadratization polynomial, before multiplying by β. -/
def dualTerm (a b c w : Bool) : ℝ :=
  bit w * (bit a + bit b + bit c - 1) + bit b * bit c - bit b - bit c

/-- The normalized penalty dominates the original violated-cut indicator. -/
theorem quadratization_lower (a b c w : Bool) :
    bit a * (1 - bit b) * (1 - bit c) ≤ dualTerm a b c w + 1 := by
  cases a <;> cases b <;> cases c <;> cases w <;> norm_num [dualTerm, bit]

/-- An explicit auxiliary choice attains the original cubic penalty exactly. -/
theorem quadratization_attained (a b c : Bool) :
    dualTerm a b c (!(a || b || c)) + 1 = bit a * (1 - bit b) * (1 - bit c) := by
  cases a <;> cases b <;> cases c <;> norm_num [dualTerm, bit]

/-- A violated cut has a strict improving repair when β > lam ≥ 0.
The unchanged contribution of every other arc is represented by `rest`.
For a=1,b=0,c=0 the old local contribution is lam*x; the repaired x=c=1,w=0
contribution is lam-β. This repair is independent of the capacities. -/
theorem dual_exchange (lam β x rest : ℝ) (hlam : 0 ≤ lam) (hx : 0 ≤ x)
    (hβ : lam < β) : rest + lam - β < rest + lam * x := by
  have hprod : 0 ≤ lam * x := mul_nonneg hlam hx
  linarith

/-- Strict support at budget B: every other cardinality has higher energy. -/
def strictSupport (f : ℕ → ℝ) (n B : ℕ) (lam : ℝ) : Prop :=
  ∀ k, k ≤ n → k ≠ B → f B + lam * (B : ℝ) < f k + lam * (k : ℝ)

/-- The two-sided window, written as all finite secant inequalities.
For an interior budget this is the same as max(lower secants) < lam <
min(upper secants); the universal form also handles endpoint budgets.
-/
def strictWindow (f : ℕ → ℝ) (n B : ℕ) (lam : ℝ) : Prop :=
  (∀ k, k ≤ n → B < k → (f B - f k) / ((k : ℝ) - (B : ℝ)) < lam) ∧
  (∀ k, k ≤ n → k < B → lam < (f k - f B) / ((B : ℝ) - (k : ℝ)))

/-- Exact characterization of strict optimality at cardinality B. -/
theorem strictSupport_iff_strictWindow (f : ℕ → ℝ) (n B : ℕ) (lam : ℝ) :
    strictSupport f n B lam ↔ strictWindow f n B lam := by
  constructor
  · intro hs
    constructor
    · intro k hkn hBk
      have hp : 0 < (k : ℝ) - (B : ℝ) := sub_pos.mpr (Nat.cast_lt.mpr hBk)
      apply (div_lt_iff₀ hp).2
      have h := hs k hkn (ne_of_gt hBk)
      nlinarith
    · intro k hkn hkB
      have hp : 0 < (B : ℝ) - (k : ℝ) := sub_pos.mpr (Nat.cast_lt.mpr hkB)
      apply (lt_div_iff₀ hp).2
      have h := hs k hkn (ne_of_lt hkB)
      nlinarith
  · rintro ⟨hlo, hhi⟩ k hkn hne
    rcases lt_or_gt_of_ne hne with hkB | hBk
    · have hp : 0 < (B : ℝ) - (k : ℝ) := sub_pos.mpr (Nat.cast_lt.mpr hkB)
      have h := (lt_div_iff₀ hp).1 (hhi k hkn hkB)
      nlinarith
    · have hp : 0 < (k : ℝ) - (B : ℝ) := sub_pos.mpr (Nat.cast_lt.mpr hBk)
      have h := (div_lt_iff₀ hp).1 (hlo k hkn hBk)
      nlinarith

/-- A line through (B,f B), strictly below every other sampled point.
This is the exposed lower-support formulation of the convex-envelope claim. -/
def exposedLowerPoint (f : ℕ → ℝ) (n B : ℕ) : Prop :=
  ∃ slope intercept : ℝ, f B = intercept + slope * (B : ℝ) ∧
    ∀ k, k ≤ n → k ≠ B → intercept + slope * (k : ℝ) < f k

/-- A strict linear-penalty window exists exactly at an exposed lower point.
The affine supporting line has slope -lam. -/
theorem window_exists_iff_exposed (f : ℕ → ℝ) (n B : ℕ) :
    (∃ lam, strictWindow f n B lam) ↔ exposedLowerPoint f n B := by
  constructor
  · rintro ⟨lam, hw⟩
    have hs := (strictSupport_iff_strictWindow f n B lam).2 hw
    refine ⟨-lam, f B + lam * (B : ℝ), ?_, ?_⟩
    · nlinarith
    · intro k hkn hne
      have h := hs k hkn hne
      nlinarith
  · rintro ⟨slope, intercept, hB, hline⟩
    refine ⟨-slope, (strictSupport_iff_strictWindow f n B (-slope)).1 ?_⟩
    intro k hkn hne
    have h := hline k hkn hne
    nlinarith

/-- A squared mismatch of two integer cardinalities is at least one. -/
theorem squared_budget_gap (k B : ℕ) (hne : k ≠ B) :
    1 ≤ ((k : ℝ) - (B : ℝ)) ^ 2 := by
  rcases lt_or_gt_of_ne hne with hkB | hBk
  · have h : ((k + 1 : ℕ) : ℝ) ≤ (B : ℝ) := Nat.cast_le.mpr (Nat.succ_le_of_lt hkB)
    simp only [Nat.cast_add, Nat.cast_one] at h
    nlinarith [sq_nonneg ((k : ℝ) - (B : ℝ) + 1)]
  · have h : ((B + 1 : ℕ) : ℝ) ≤ (k : ℝ) := Nat.cast_le.mpr (Nat.succ_le_of_lt hBk)
    simp only [Nat.cast_add, Nat.cast_one] at h
    nlinarith [sq_nonneg ((k : ℝ) - (B : ℝ) - 1)]

theorem dualTerm_nonnegative (a b c w : Bool) : 0 ≤ dualTerm a b c w + 1 := by
  cases a <;> cases b <;> cases c <;> cases w <;> norm_num [dualTerm, bit]

theorem dualTerm_positive_gap (a b c w : Bool) (hne : dualTerm a b c w + 1 ≠ 0) :
    1 ≤ dualTerm a b c w + 1 := by
  cases a <;> cases b <;> cases c <;> cases w <;> norm_num [dualTerm, bit] at *

/-- Simultaneous exact-penalty separation. The nonnegative objective lies in
[0,U], p is squared budget violation, q is total normalized dual/auxiliary
violation. Joint violations are allowed; either violation being at least one
suffices. Every infeasible energy then exceeds every feasible objective.
-/
theorem simultaneous_penalty_separation (U lam β feasible raw p q : ℝ)
    (hU : 0 ≤ U) (hlam : U < lam) (hβ : U < β) (hf : feasible ≤ U)
    (hr : 0 ≤ raw) (hp : 0 ≤ p) (hq : 0 ≤ q) (hbad : 1 ≤ p ∨ 1 ≤ q) :
    feasible < raw + lam * p + β * q := by
  have hlam0 : 0 ≤ lam := le_trans hU (le_of_lt hlam)
  have hβ0 : 0 ≤ β := le_trans hU (le_of_lt hβ)
  rcases hbad with hp1 | hq1
  · have hlarge : lam ≤ lam * p := by simpa using mul_le_mul_of_nonneg_left hp1 hlam0
    have hother : 0 ≤ β * q := mul_nonneg hβ0 hq
    linarith
  · have hlarge : β ≤ β * q := by simpa using mul_le_mul_of_nonneg_left hq1 hβ0
    have hother : 0 ≤ lam * p := mul_nonneg hlam0 hp
    linarith

/-- info: 'NIPPenalties.quadratization_lower' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms quadratization_lower
/-- info: 'NIPPenalties.quadratization_attained' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms quadratization_attained
/-- info: 'NIPPenalties.dual_exchange' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms dual_exchange
/-- info: 'NIPPenalties.strictSupport_iff_strictWindow' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms strictSupport_iff_strictWindow
/-- info: 'NIPPenalties.window_exists_iff_exposed' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms window_exists_iff_exposed
/-- info: 'NIPPenalties.squared_budget_gap' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms squared_budget_gap
/-- info: 'NIPPenalties.dualTerm_positive_gap' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms dualTerm_positive_gap
/-- info: 'NIPPenalties.dualTerm_nonnegative' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms dualTerm_nonnegative
/-- info: 'NIPPenalties.simultaneous_penalty_separation' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms simultaneous_penalty_separation

end NIPPenalties
