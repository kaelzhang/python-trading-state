# Problem Specification (English, Disambiguated)

## 1. Setup

We have a total amount of liquid distributed across **\(n\)** buckets.
Bucket indices are \(j \in \{0,1,\dots,n-1\}\).

- **Available volume (capacity constraint)** in each bucket:
  \[
  S = (S_0, S_1, \dots, S_{n-1}), \quad S_j \in \mathbb{R}_{\ge 0}
  \]
- **Positive weights**:
  \[
  W = (w_0, w_1, \dots, w_{n-1}), \quad w_j \in \mathbb{R}_{>0}, \quad w_0 = 1
  \]
  The input `weights: List[float]` satisfies `weights[0] == 1.0`, and each `weights[j]` represents the ratio \(w_j / w_0\).

- **Target net volume to transfer out of buckets**:
  \[
  V_{\text{target}} \in \mathbb{R}_{\ge 0}
  \]
  In code this is the parameter `take: Decimal`.

### One-shot constraint (critical)

Each bucket may be poured **at most once**. Once a bucket is poured (regardless of whether some volume is returned), it is **removed** from all subsequent allocation calculations.

Let \(U \subseteq \{0,\dots,n-1\}\) denote the set of buckets that have **not yet** been poured.

---

## 2. Pouring / Acceptance Model (Black Box)

When we *attempt* to pour an amount \(v_j\) from bucket \(j\), the factory may reject (return) part of it.

- Planned (attempted) pour from bucket \(j\): \(v_j\)
- Returned volume to bucket \(j\): \(r_j\)
- Net accepted (actually transferred out): \(a_j\)

Constraints:
\[
0 \le r_j \le v_j \le S_j
\]
\[
a_j = v_j - r_j
\]

The returned volume is produced by an external **black-box** function:
- Input: `take: Decimal` (this corresponds to \(v_j\))
- Output: `Decimal` (this corresponds to \(r_j\))

---

## 3. Remaining Demand Update

Let \(V\) be the **remaining net volume** still required to reach the target. Initially:
\[
V \leftarrow V_{\text{target}}
\]

After pouring bucket \(j\) once (attempting \(v_j\), receiving return \(r_j\)), update the remaining demand by subtracting the *net accepted* amount:
\[
V \leftarrow V - a_j = V - (v_j - r_j) = V - v_j + r_j
\]

Then remove bucket \(j\) from \(U\).

---

## 4. Allocation Goal Under Weights

At any allocation step with remaining bucket set \(U\) and remaining demand \(V\), we would like the planned pours \(\{v_j\}_{j\in U}\) to follow the weight proportions **as closely as possible**, subject to each bucket's upper bound \(S_j\).

The ideal (unconstrained) proportional allocation would be:
\[
v^{\mathrm{ideal}}_j = V\cdot \frac{w_j}{\sum_{i\in U} w_i}
\]

But we must enforce \(v_j \le S_j\). Therefore, the actual *planned* allocation for the current iteration is the solution to:
\[
\text{Find } v_j = \min(S_j, \alpha w_j)\ \text{ for } j\in U,\ \text{such that } \sum_{j\in U} v_j = V
\]
for some \(\alpha \ge 0\), if such \(\alpha\) exists.

This is a **weighted water-filling with upper bounds** problem.

---

## 5. Algorithm (Stepwise Requirements)

Given current remaining demand \(V\) and remaining buckets \(U\):

### Step 1. Full-capacity case
If:
\[
V \ge \sum_{j\in U} S_j
\]
then proceed to Step 2; otherwise go to Step 3.

### Step 2. Pour all remaining buckets (terminate)
For each \(j \in U\), perform a single pour of the **entire** bucket:
\[
v_j \leftarrow S_j
\]
Apply the black-box return to obtain \(r_j\), record net \(a_j = v_j - r_j\), update \(V\leftarrow V-v_j+r_j\), and remove \(j\) from \(U\).

After all buckets in \(U\) have been poured once, terminate (even if \(V>0\), since no buckets remain).

### Step 3. Compute proportional planned pours
Compute planned pours \(v_j\) for all \(j \in U\) that satisfy:
- \(v_j\) follows \(w_j\) proportions as closely as possible, and
- \(0 \le v_j \le S_j\),
- \(\sum_{j\in U} v_j = V\),

i.e. \(v_j = \min(S_j, \alpha w_j)\) for some \(\alpha\), if feasible.

### Step 4. Check saturation
If there exists any \(j\in U\) such that:
\[
v_j > S_j
\]
then proceed to Step 5; otherwise proceed to Step 6.

> Note: With the definition \(v_j = \min(S_j, \alpha w_j)\), the inequality \(v_j > S_j\) should not occur.
> In practice, Step 4–5 corresponds to the iterative process of identifying buckets that **would exceed** \(S_j\) under the unconstrained proportional allocation and treating them as saturated.

### Step 5. Saturate overflowing buckets and repeat
For every bucket that is determined to be saturated in the current iteration, pour it once with:
\[
v_j \leftarrow S_j
\]
Apply the black-box return, update \(V \leftarrow V - v_j + r_j\), remove those buckets from \(U\), and return to Step 1.

### Step 6. Pour planned amounts and terminate
For each remaining bucket \(j \in U\), pour exactly the computed planned amount \(v_j\) **once**, apply the black-box return, record net \(a_j=v_j-r_j\), and terminate.

---

## 6. Implementation Requirements (Python)

Implement the above behavior in Python.

### Function signature and I/O

Implement a function that:

- **Inputs**
  - `resource_volume: List[Decimal]` representing \((S_0,\dots,S_{n-1})\)
  - `weights: List[float]` representing \((w_0,\dots,w_{n-1})\), with `weights[0] == 1.0` and all entries \(>0\)
  - `take: Decimal` representing \(V_{\text{target}}\)

- **External black box**
  - A callable `return_volume(take: Decimal) -> Decimal` returning the returned volume \(r_j\) for a planned pour \(v_j\)

- **Output**
  - A list `List[Decimal]` of length \(n\) containing the **net accepted** volumes:
    \[
    (a_0, a_1, \dots, a_{n-1}) = (v_0-r_0, v_1-r_1,\dots,v_{n-1}-r_{n-1})
    \]
    For buckets never poured (because the algorithm terminated early), return \(0\) at those indices.

### Performance constraints

- The implementation should **minimize explicit looping** where possible and avoid quadratic behavior.
- Provide detailed inline comments explaining:
  - the mathematical rationale,
  - why the implementation is correct under the “pour once then exclude” rule,
  - and where the black-box calls occur.

### Final requirement: complexity analysis

Include an explicit time complexity analysis of the implementation in terms of:
- \(n\) (number of buckets), and
- \(T_{\text{blackbox}}\) (time per black-box call)

At minimum, the solution should identify the dominant term and explain why the black-box forces at least \(\Theta(k)\) calls for \(k\) poured buckets.
