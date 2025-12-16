# Weighted Volume Extraction From Multiple Buckets (with Stochastic/Black‑Box Return)

## Problem Statement

You are given \(n\) buckets containing a liquid resource. Bucket \(j\) initially contains volume

\[
S_j \in \mathbb{R}_{\ge 0}, \quad j \in \{0,1,\dots,n-1\}.
\]

You also assign a positive weight to each bucket:

\[
W_j \in \mathbb{R}_{>0}, \quad \text{with } W_0 = 1.
\]

Your goal is to extract (i.e., pour out) a **net** target volume \(V \in \mathbb{R}_{\ge 0}\) from the set of buckets, subject to:

- **Single-use constraint:** each bucket can be poured **at most once**. Once a bucket has been poured (even partially), it is removed from subsequent computation.
- **Capacity constraint:** you cannot plan to pour more than remains in that bucket, i.e. \(V_j \le S_j\) for any planned amount \(V_j\).

However, pouring is performed by a factory that may not accept all of the planned volume. Some volume may be returned to the same bucket.

---

## Pouring Model (“Pouring Procedure”)

When attempting to pour an amount \(x\) from any bucket, the factory returns a volume

\[
R(x) \in [0, x],
\]

where \(R(\cdot)\) is an **external black-box function** (unknown implementation).

- **Planned pour:** \(x\)
- **Returned volume:** \(R(x)\)
- **Net extracted volume:** \(x - R(x)\)

The black-box signature is:

- Input: `take: Decimal` (planned pour amount)
- Output: `Decimal` (returned amount)

---

## Residual Target Volume (“Remaining Volume”)

Let the remaining net volume requirement be \(V\). After pouring bucket \(j\) with planned volume \(V_j\), the remaining requirement is updated as:

\[
V \leftarrow V - (V_j - R(V_j)) = V - V_j + R(V_j).
\]

Buckets that have been poured are excluded from future steps.

---

## Algorithm Requirements

You must implement the following logic.

### Inputs

- A list of bucket volumes: \(\{S_0, S_1, \dots, S_{n-1}\}\)
- A list of weights: \(\{W_0, W_1, \dots, W_{n-1}\}\) with \(W_0=1\) and \(W_j>0\)
- A target net extraction volume: \(V\)

### Outputs

Compute and return the **actual net extracted volume** from each bucket:

\[
A_j = V_j - R(V_j), \quad j = 0,\dots,n-1,
\]

as `List[Decimal]`. If a bucket is never poured, then \(A_j = 0\).

---

## Step-by-Step Procedure

Let \(U\) be the set of buckets that have **not** yet been poured.

1. **Total capacity check**

   - If \(V \ge \sum_{j \in U} S_j\), go to Step 2.
   - Otherwise, go to Step 3.

2. **Pour all remaining buckets**

   For every \(j \in U\), perform a single pour with planned volume \(V_j := S_j\) (i.e., attempt to pour everything remaining in that bucket), using the pouring procedure. Record \(A_j\). **Stop**.

3. **Compute proportional planned pours**

   Compute planned pour amounts \(\{V_j\}_{j\in U}\) such that they follow the weight ratios **as closely as possible**, meaning:

   \[
   V_j \propto W_j \quad \text{for } j \in U,
   \]

   while also satisfying:

   \[
   0 \le V_j \le S_j \quad \forall j \in U,
   \qquad \sum_{j \in U} V_j = V.
   \]

   (Equivalently, this is a weighted allocation under upper bounds.)

4. **Check for capacity violations**

   - If \(\exists j \in U\) such that \(V_j > S_j\), go to Step 5.
   - Otherwise, go to Step 6.

5. **Saturate violating buckets, update remaining target, and repeat**

   For every bucket \(j \in U\) with \(V_j > S_j\), do:

   - Planned pour: \(V_j := S_j\) (pour the entire bucket once)
   - Apply pouring procedure to obtain returned volume \(R(V_j)\)
   - Record \(A_j = V_j - R(V_j)\)
   - Update remaining target volume:
     \[
     V \leftarrow V - V_j + R(V_j)
     \]
   - Remove \(j\) from \(U\)

   Then return to Step 1.

6. **Pour according to computed plan**

   For each \(j \in U\), pour the planned amount \(V_j\) once (using the pouring procedure), record \(A_j\), and **stop**.

---

## Implementation Requirements (Python)

Implement a function with the following signature:

- **Function name:** (free to choose)
- **Parameters:**
  - `resource_volume: List[Decimal]`  — \([S_0,\dots,S_{n-1}]\)
  - `weights: List[float]` — \([W_0,\dots,W_{n-1}]\), where `weights[0] == 1.0` and later entries are ratios relative to the first weight
  - `take: Decimal` — the initial target net extraction volume \(V\)
- **Return value:**
  - `List[Decimal]` — \([A_0,\dots,A_{n-1}]\), where \(A_j\) is the net extracted volume from bucket \(j\)

Additional constraints:

- Minimize explicit loops where possible and reduce time complexity.
- Provide detailed inline comments explaining the design and the mathematical reasoning.
- Include a final analysis of time complexity.

---

## Notes on Numerical Types

- Volumes must use `Decimal` to reduce floating-point error.
- If weights are provided as `float`, conversion strategy should be clearly justified (e.g., via `Decimal(str(w))`).
