# Geometry-Aware KV Cache Compression Math

> Reference derivation note.
>
> This document derives the surrogate geometry-aware objectives. It should not be read as evidence that direct full-metric preconditioning will work with the current V3 backend. Stage 1 showed a gap between the clean transformed-space derivation and the behavior of the actual backend used in experiments.

This note derives the proposed geometry-aware objectives for KV cache compression and shows how they reduce to ordinary Euclidean quantization after a linear transform.

The main idea is simple:

- plain quantization minimizes uniform reconstruction error, usually `||x - x_hat||_2^2`
- geometry-aware quantization minimizes a weighted quadratic form, `e^T G e`, where `e = x_hat - x`
- if `G` is positive semidefinite, we can factor it as `G = L^T L`, so the weighted error becomes ordinary MSE in transformed coordinates:

```math
e^T G e = e^T L^T L e = ||L e||_2^2
```

That lets us use a generic quantization backend, such as TurboQuant, in a task-aware geometry.

---

## 1. Setup

Consider one decoder-only transformer attention head.

For token position `i`, let:

```math
q_i = R_i W_Q h_i,\qquad
k_i = R_i W_K h_i,\qquad
v_i = W_V h_i
```

where:

- `h_i` is the hidden state
- `R_i` is the RoPE matrix
- `W_Q, W_K, W_V` are the query, key, and value projections

At decode step `t`, attention weight on cached token `i` is:

```math
a_{ti}
=
\frac{\exp(q_t^T k_i / \sqrt d)}
{\sum_{j \le t} \exp(q_t^T k_j / \sqrt d)}
```

and the attention output contributes:

```math
h_t^{\mathrm{out}}
=
h_t + \sum_{i \le t} a_{ti} W_O v_i
```

Suppose compression produces:

```math
\hat k_i = k_i + e_i^{(K)},\qquad
\hat v_i = v_i + e_i^{(V)}
```

where `e_i^{(K)}` and `e_i^{(V)}` are the key and value errors.

---

## 2. Key-Side Distortion

### 2.1 Logit perturbation

For a future query `q`, the attention logit of token `i` is:

```math
\ell_i = \frac{q^T k_i}{\sqrt d}
```

After key compression:

```math
\hat \ell_i = \frac{q^T \hat k_i}{\sqrt d}
= \frac{q^T (k_i + e_i^{(K)})}{\sqrt d}
```

so the logit perturbation is:

```math
\delta \ell_i
= \hat \ell_i - \ell_i
= \frac{q^T e_i^{(K)}}{\sqrt d}
```

### 2.2 Expected squared logit error

We want a distortion that reflects how future queries use the key. The natural objective is the expected squared logit error:

```math
D_i^{(K)}
:=
\mathbb{E}_q[(\delta \ell_i)^2]
```

Substitute the expression above:

```math
D_i^{(K)}
=
\mathbb{E}_q\left[\left(\frac{q^T e_i^{(K)}}{\sqrt d}\right)^2\right]
=
\frac{1}{d} \mathbb{E}_q\left[e_i^{(K)T} q q^T e_i^{(K)}\right]
```

Since `e_i^{(K)}` does not depend on `q`, pull it outside the expectation:

```math
D_i^{(K)}
=
\frac{1}{d}
e_i^{(K)T}
\mathbb{E}[q q^T]
e_i^{(K)}
```

Define the future-query second moment:

```math
M_q := \mathbb{E}[q q^T]
```

Then:

```math
D_i^{(K)}
=
\frac{1}{d} e_i^{(K)T} M_q e_i^{(K)}
```

This is the proposed geometry-aware key distortion.

### 2.3 Interpretation

This is a quadratic form. Its meaning is:

- errors aligned with directions that future queries probe often are expensive
- errors in directions future queries rarely use are cheap

If `M_q` has eigendecomposition:

```math
M_q = U \Lambda U^T
```

then:

```math
e^T M_q e
=
\sum_r \lambda_r (u_r^T e)^2
```

So each eigen-direction `u_r` is weighted by eigenvalue `\lambda_r`.

Large `\lambda_r` means future queries are sensitive in that direction.

---

## 3. Query Models for `M_q`

The entire key geometry is controlled by `M_q`.

### 3.1 Gaussian query model

If future queries are approximated by:

```math
q \sim \mathcal{N}(\mu_q, \Sigma_q)
```

then:

```math
M_q = \mathbb{E}[q q^T] = \Sigma_q + \mu_q \mu_q^T
```

This is the Expected Attention style formulation.

### 3.2 Empirical reference-query model

If we instead have reference queries `q_1, \dots, q_n`, then:

```math
M_q \approx \frac{1}{n} \sum_{j=1}^n q_j q_j^T
```

This is the empirical analogue used by Attention Matching.

So Expected Attention and Attention Matching induce the same mathematical object: a future-query second moment.

---

## 4. Converting Key Geometry to Ordinary MSE

Assume `M_q` is positive semidefinite. Factor it as:

```math
M_q = L_K^T L_K
```

Then:

```math
D_i^{(K)}
=
\frac{1}{d} e_i^{(K)T} L_K^T L_K e_i^{(K)}
=
\frac{1}{d} ||L_K e_i^{(K)}||_2^2
```

Now define transformed keys:

```math
\tilde k_i := L_K k_i
```

If quantization in transformed space gives:

```math
\hat{\tilde k}_i = \tilde k_i + \tilde e_i
```

then with `\hat k_i = L_K^{-1} \hat{\tilde k}_i`, we have:

```math
\tilde e_i = L_K (\hat k_i - k_i) = L_K e_i^{(K)}
```

Therefore:

```math
D_i^{(K)} = \frac{1}{d} ||\hat{\tilde k}_i - \tilde k_i||_2^2
```

So minimizing query-aware key distortion is exactly the same as minimizing Euclidean MSE in transformed coordinates.

This is the main derivation behind geometry-aware key quantization.

### 4.1 Practical note

If `M_q` is singular or noisy, use regularization:

```math
M_q^{\mathrm{reg}} = M_q + \epsilon I
```

and factor `M_q^{\mathrm{reg}}` instead.

---

## 5. Value-Side Distortion

For values, the proposal uses a first-order approximation to residual-stream error.

The residual contribution of token `i` at a future step is:

```math
\Delta h_i = a_i W_O v_i
```

After compressing the value:

```math
\hat{\Delta h}_i = a_i W_O \hat v_i
= a_i W_O (v_i + e_i^{(V)})
```

Therefore the perturbation is:

```math
\delta h_i
=
\hat{\Delta h}_i - \Delta h_i
=
a_i W_O e_i^{(V)}
```

Taking squared norm:

```math
||\delta h_i||_2^2
=
a_i^2 ||W_O e_i^{(V)}||_2^2
```

Since the true future attention weight `a_i` is unknown, replace it with an estimate `\hat a_i`, giving:

```math
D_i^{(V)}
\propto
\hat a_i^2 ||W_O e_i^{(V)}||_2^2
```

Now expand:

```math
||W_O e_i^{(V)}||_2^2
=
e_i^{(V)T} W_O^T W_O e_i^{(V)}
```

Define:

```math
G_V := W_O^T W_O
```

Then:

```math
D_i^{(V)}
\propto
\hat a_i^2 e_i^{(V)T} G_V e_i^{(V)}
```

This is the geometry-aware value distortion.

### 5.1 Value transform

Factor:

```math
G_V = L_V^T L_V
```

Then:

```math
D_i^{(V)}
\propto
\hat a_i^2 ||L_V e_i^{(V)}||_2^2
```

Define transformed values:

```math
\tilde v_i := L_V v_i
```

and quantize in this transformed space.

So value quantization is also ordinary MSE after a task-aware linear transform, with an additional token-importance weight `\hat a_i^2`.

---

## 6. Geometry vs Importance

It is useful to separate two different concepts:

### 6.1 Geometry

Geometry says which directions inside a vector matter more.

- for keys: geometry is `M_q`
- for values: geometry is `W_O^T W_O`

### 6.2 Importance

Importance says how much the token as a whole matters.

- for keys: importance can be based on expected attention `\hat a_i`
- for values: importance naturally appears as `\hat a_i^2`

So a practical distortion model usually has the form:

```math
D_i(b)
=
\alpha_i \cdot \mathrm{QuantErrorInRelevantGeometry}(b)
```

where:

- `b` is the bitwidth
- `\alpha_i` is a tokenwise importance weight

---

## 7. Rate Allocation

Suppose we allow each token to use a bitwidth from a discrete set:

```math
\mathcal{B} = \{0, 2, 3, 4, 8\}
```

where `b = 0` means prune the token.

Let `D_i(b)` be the predicted distortion if token `i` is stored at bitwidth `b`, and let `r(b)` be its rate cost.

Then the rate-distortion objective is:

```math
\min_{\{b_i\}}
\sum_i D_i(b_i) + \lambda \sum_i r(b_i)
```

This decomposes tokenwise if distortion is modeled independently:

```math
b_i^*
=
\arg\min_{b \in \mathcal{B}}
D_i(b) + \lambda r(b)
```

Interpretation:

- high-importance tokens get more bits
- low-importance tokens get fewer bits
- if `b = 0` wins, the token is pruned

This is how pruning and quantization become one unified allocation problem.

---

## 8. Geometry-Aware Key Quantization: Concrete Derivation

This section gives a fully explicit version for the simplest v1 method: key-only, token-preserving quantization.

### 8.1 Step 1: estimate query geometry

Estimate:

```math
M_q = \mathbb{E}[q q^T]
```

using either:

```math
M_q = \Sigma_q + \mu_q \mu_q^T
```

or

```math
M_q = \frac{1}{n}\sum_{j=1}^n q_j q_j^T
```

### 8.2 Step 2: transform keys

Compute a factor:

```math
M_q = L_K^T L_K
```

and transform:

```math
\tilde k_i = L_K k_i
```

### 8.3 Step 3: quantize in transformed space

Apply a quantizer `Q_b` at bitwidth `b`:

```math
\hat{\tilde k}_i^{(b)} = Q_b(\tilde k_i)
```

Map back:

```math
\hat k_i^{(b)} = L_K^{-1} \hat{\tilde k}_i^{(b)}
```

### 8.4 Step 4: distortion estimate

The key distortion at bitwidth `b` is:

```math
D_i^{(K)}(b)
=
\frac{1}{d}
(\hat k_i^{(b)} - k_i)^T M_q (\hat k_i^{(b)} - k_i)
```

Using the transform:

```math
D_i^{(K)}(b)
=
\frac{1}{d}
||\hat{\tilde k}_i^{(b)} - \tilde k_i||_2^2
```

So the quantizer only needs to minimize transformed-space MSE.

### 8.5 Step 5: add token importance

If desired, weight the distortion by expected attention:

```math
\tilde D_i^{(K)}(b)
=
\hat a_i \, D_i^{(K)}(b)
```

or another token-importance signal.

### 8.6 Step 6: choose the bitwidth

Finally:

```math
b_i^*
=
\arg\min_{b \in \mathcal B}
\tilde D_i^{(K)}(b) + \lambda r(b)
```

This is the simplest end-to-end derivation of a geometry-aware, query-aware key quantizer.

---

## 9. Attention Matching as an Induced Geometry

Attention Matching is not written as a quadratic-form method, but it implicitly induces the same structure.

Suppose for a fixed key error `e` we want to preserve logits on reference queries `q_1, \dots, q_n`. Then a natural key loss is:

```math
\sum_{j=1}^n (q_j^T e)^2
```

Rewrite it:

```math
\sum_{j=1}^n (q_j^T e)^2
=
\sum_{j=1}^n e^T q_j q_j^T e
=
e^T \left(\sum_{j=1}^n q_j q_j^T\right) e
```

So the empirical geometry matrix is:

```math
G_{\mathrm{AM}} = \sum_{j=1}^n q_j q_j^T
```

which is exactly the same type of object as `M_q`.

This is why:

- Expected Attention gives a probabilistic estimate of the geometry
- Attention Matching gives an empirical estimate of the geometry

They are mathematically aligned.

---

## 10. Why This Matters for TurboQuant

TurboQuant is a strong online quantization backend for Euclidean distortion, especially after a rotation.

The geometry-aware derivation tells us how to make it task-aware:

1. estimate the relevant metric `G`
2. factor `G = L^T L`
3. transform vectors with `L`
4. run TurboQuant in transformed space
5. invert the transform

So TurboQuant itself does not need to know anything about future queries. The future-query model only changes the geometry in which TurboQuant operates.

---

## 11. Practical Approximations

The clean derivation above is exact at the metric level, but approximations are needed in practice.

### 11.1 Approximate `M_q`

Full `d x d` second moments can be noisy or expensive. Common approximations:

- diagonal `M_q`
- low-rank plus diagonal
- block-diagonal by head substructure
- regularized full matrix

### 11.2 Approximate value geometry

The exact value-side metric uses `W_O^T W_O`. In practice:

- use the per-head block of `W_O`
- use only diagonal or low-rank approximations
- optionally absorb token importance and geometry separately

### 11.3 Nonlinearity of attention

The derivation uses:

- exact logit perturbation for keys
- first-order residual perturbation for values

But downstream attention behavior is nonlinear because of the softmax and interaction across tokens. So the quadratic objectives should be viewed as tractable local surrogates, not exact end-to-end losses.

---

## 12. Summary

The derivation can be summarized in one line:

```math
\text{relevant distortion} = e^T G e = ||L e||_2^2
```

with different choices of `G`:

- keys: `G = M_q = \mathbb{E}[q q^T]`
- values: `G = \hat a_i^2 W_O^T W_O`

This gives a principled recipe:

1. estimate which directions matter
2. turn that into a PSD metric `G`
3. factor `G = L^T L`
4. quantize in transformed coordinates
5. allocate bits by minimizing distortion plus rate

That is the mathematical core of the proposed geometry-aware KV compression approach.
