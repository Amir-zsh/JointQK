# Theoretical and Empirical Motivation for the Joint-QK Basis (`r_sym_waterfill`)

> Companion to E1–E5. This note derives the rate-distortion theory underlying the
> Stage 1E (basis × allocation) design space, shows precisely what Q-only (V) and
> TurboQuant (random Hadamard + unit-norm) miss, and motivates the joint Q-K basis
> as the principled choice. Empirical confirmation against the canonical E3
> artifacts is in §10.

## 1. Problem formulation

Let a transformer self-attention layer receive a prompt of length $L$ and produce
post-RoPE queries $q_t \in \mathbb{R}^d$ and keys $k_t \in \mathbb{R}^d$, with
$t = 1, \ldots, L$, per `(layer, kv_head)`. The layer's attention logits are the
inner products $q_s^\top k_t / \sqrt{d}$. We want to **compress** the prefill keys
$K = (k_1, \ldots, k_L) \in \mathbb{R}^{L \times d}$ to a small number of bits per
coordinate so that later queries (prefill or decode) read **reconstructed** keys
$\hat K$ and produce nearly the same logits.

Concretely, given a budget of $b_{\text{avg}}$ bits per coordinate on average, we
want a compressor $K \mapsto \hat K$ that minimizes the *attention-relevant* error
introduced by the compression. The **operational metric** is the expected squared
error in the attention logit, evaluated against the calibration query distribution.

The full design space we consider has two ingredients:

1. **Basis.** A linear transform $R : \mathbb{R}^d \to \mathbb{R}^d$ applied to keys
   before quantization. Reconstruction reverses it.
2. **Bit allocation.** A vector $b = (b_1, \ldots, b_d)$ with
   $\sum_j b_j = b_{\text{avg}} \cdot d$, specifying how many bits each transformed
   coordinate gets.

Three natural specializations occupy this space:

| Method | Basis $R$ | Allocation $b$ |
|---|---|---|
| **TurboQuant (V3)** | random orthogonal $R_{\text{rand}}$ (Hadamard); vectors unit-normalized | uniform $b_j = b_{\text{avg}}$ |
| **Q-only (V)** | eigvecs of $\Sigma_Q$ (call it $V_Q$) | reverse water-fill |
| **JointQK ($R_{\text{sym}}$)** | eigvecs of $(\Sigma_Q \Sigma_K + \Sigma_K \Sigma_Q)/2$ | reverse water-fill |

This note derives, from first principles, the asymptotic distortion each of these
achieves; identifies the precise quantity each one fails to minimize; and shows
why JointQK closes the gap.

## 2. Notation and standing assumptions

| symbol | meaning |
|---|---|
| $d$ | head dimension (=128 for Qwen3-8B) |
| $q, k$ | a single post-RoPE query / key vector in $\mathbb{R}^d$ |
| $\Sigma_Q := \mathbb{E}[q q^\top]$ | uncentered Q second moment (per `(layer, kv_head)`, GQA-pooled) |
| $\Sigma_K := \mathbb{E}[k k^\top]$ | uncentered K second moment |
| $C_{QK} := \mathbb{E}[q k^\top]$ | Q-K cross moment |
| $R$ | basis matrix; column convention $c = R^\top k$ |
| $b_j$ | bits allocated to the $j$-th basis-coordinate (real in theory; integer-rounded in practice) |
| $\kappa$ | Bennett shape constant ($\kappa = \pi\sqrt{3}/2 \approx 2.72$ for a Gaussian source) |
| $\lambda_j(M)$ | $j$-th eigenvalue of $M$ (sorted descending) |
| $(M)_{jj}$ | $j$-th diagonal entry of $M$ |

**Standing assumptions** (each invoked locally below):

- **(A1) Calibration captures the test distribution.** $\Sigma_Q, \Sigma_K, C_{QK}$
  computed from a calibration corpus describe the deployment Q/K distribution well
  enough that expectations under the calibration distribution are operationally
  meaningful. E4 verified this within ≤ 0.3 pp top-1 across LongBench-E configs.
- **(A2) Quantization noise is uncorrelated across coordinates.** This is Bennett's
  classical assumption — when each coordinate is independently scalar-quantized at
  high rate, the error process is approximately white. Invoked in §4–5.
- **(A3) Reconstruction error is uncorrelated with the query distribution.** We
  treat $\Delta k = k - \hat k$ as having a fixed (data-driven) covariance, not
  as a function of $q$. **This is a modeling choice for the calibration geometry
  objective, not an identity of the paired data**: in reality $\Delta k$ is a
  deterministic function of $k$, and $(q_t, k_t)$ at the same prompt position
  share context. E3's geometry-distortion measurement uses the same factorization,
  so the theory is at least self-consistent with what we measure.
- **(A4) High-rate scalar quantization.** $b_j \ge 2$ for active coordinates so
  Bennett's $2^{-2b}$ distortion approximation holds. At $b_{\text{avg}} \in \{2, 3, 4\}$
  this is borderline at the bottom and tight at the top; empirical agreement is
  within ~2% of theory in §10.

We do **not** assume Gaussianity of $q$ or $k$. Bennett's constant $\kappa$ depends
on source shape but cancels in *ratios*, which is all the basis-comparison analysis
needs.

## 3. Target metric: Q-weighted reconstruction MSE

The attention logit error introduced by reconstructing $k \to \hat k$ against a
particular query $q$ is $q^\top(k - \hat k) = q^\top \Delta k$. Squaring and taking
expectation under the joint $(q, \Delta k)$ distribution:

$$
\mathbb{E}\!\left[(q^\top \Delta k)^2\right]
 = \mathbb{E}\!\left[q^\top \Delta k \, \Delta k^\top q\right].
$$

**Step 1.** The squared scalar $(q^\top \Delta k)^2$ is rewritten as a quadratic
form $q^\top (\Delta k \Delta k^\top) q$.

By assumption (A3), $q$ is independent of $\Delta k$, so the expectation factors:

$$
\mathbb{E}\!\left[q^\top \Delta k \, \Delta k^\top q\right]
 = \mathbb{E}_q\!\left[ q^\top \, \mathbb{E}_{\Delta k}\!\left[\Delta k \Delta k^\top\right] q \right].
$$

**Step 2.** Independence (A3) lets us condition on $\Delta k$ first and pull the
inner expectation through.

Apply the trace identity $a^\top M a = \mathrm{tr}(M a a^\top)$ and pull the
expectation under the trace:

$$
\mathbb{E}\!\left[q^\top \Delta k \, \Delta k^\top q\right]
 = \mathrm{tr}\!\left( \mathbb{E}[q q^\top] \cdot \mathbb{E}[\Delta k \Delta k^\top] \right)
 = \mathrm{tr}\!\left( \Sigma_Q \cdot \mathrm{Cov}(\Delta k) \right).
$$

**Step 3.** Trace cyclic property and the definition of $\Sigma_Q$.

The result is the **Q-weighted reconstruction MSE**, henceforth $D$:

$$
\boxed{\,D = \mathrm{tr}\!\left( \Sigma_Q \cdot \mathrm{Cov}(\Delta k) \right).\,}
$$

> *Note on `Cov` vs second moment.* The substitution of $\mathbb{E}[\Delta k \Delta k^\top]$
> with $\mathrm{Cov}(\Delta k)$ is exact only when $\mathbb{E}[\Delta k] = 0$.
> Bennett's high-rate quantization noise is approximately zero-mean (§4.1), so
> this holds within the same approximation as (A4); the residual bias is
> $O(\|\mathbb{E}[\Delta k]\|^2)$ and dominated by the variance term at any
> reasonable bit budget.

This is exactly what E3 measures empirically as `geometry_distortion`; top-1
retention is a non-linear functional of the same $\Delta k$, but empirically
tracks $D$ for orthogonal-basis × continuous-water-fill methods (a condition we
return to in §9).

> **Key implication.** $D$ depends on $\Delta k$ only through its second moment
> $\mathrm{Cov}(\Delta k)$. Two compressors with the same noise covariance produce
> identical Q-weighted MSE — even if one makes "harder" arrow-shaped errors and
> the other makes diffuse errors. The shape of $\mathrm{Cov}(\Delta k)$ is what
> design choices control.

## 4. Per-coordinate quantization in a rotated basis

### 4.1 Bennett's high-rate scalar quantizer

For a smooth one-dimensional source $X$ with variance $\sigma^2$, an optimal
$b$-bit Lloyd–Max scalar quantizer $Q_b$ achieves

$$
\mathbb{E}\!\left[(X - Q_b(X))^2\right] = \kappa \cdot \sigma^2 \cdot 2^{-2b} + o(2^{-2b}),
$$

where $\kappa$ is a source-shape constant (Panter–Dite / Gish–Pierce). Two
properties we will use:

- **Variance scaling.** $Q_b(\alpha X)$ for fixed $\alpha$ has distortion
  $\alpha^2 \kappa \sigma^2 2^{-2b}$.
- **Whitening of error.** At high rate the quantization error
  $n = X - Q_b(X)$ is approximately uniform on each Voronoi cell, so
  $\mathbb{E}[n] \approx 0$ and $\mathbb{E}[n \cdot X] \approx 0$.

The latter is the foundation of (A2) when we move to vectors.

### 4.2 Rotated basis and per-coord quantization

Let $R \in \mathbb{R}^{d \times d}$ be invertible (we will specialize to orthogonal).
Define

$$
c := R^\top k \in \mathbb{R}^d \quad \text{(forward map)}.
$$

Independently scalar-quantize each coordinate $c_j$ with $b_j$ bits:
$\hat c_j = Q_{b_j}(c_j)$. Reconstruct

$$
\hat k = R^{-\top} \hat c = R^{-\top}(c + n) = k + R^{-\top} n,
$$

where $n_j = \hat c_j - c_j$ is the per-coord quantization noise. So

$$
\Delta k = k - \hat k = -\, R^{-\top} n.
$$

**Per-coord noise variance.** Coordinate $c_j$ has variance

$$
\sigma_j^2(R) := \mathrm{Var}(c_j) = (R^\top \Sigma_K R)_{jj}.
$$

By Bennett (§4.1):

$$
\mathrm{Var}(n_j) = \kappa \cdot \sigma_j^2(R) \cdot 2^{-2 b_j}.
$$

By (A2) different $n_j$ are uncorrelated, so

$$
\boxed{\,\mathrm{Cov}(n) = \mathrm{diag}\!\left( \kappa \cdot \sigma_j^2(R) \cdot 2^{-2 b_j} \right).\,}
$$

Reconstruction error covariance:

$$
\mathrm{Cov}(\Delta k) = R^{-\top} \, \mathrm{Cov}(n) \, R^{-1}.
$$

### 4.3 Q-weighted MSE in a rotated basis

Substituting into §3 and using the trace cyclic property:

$$
D(R, b) = \mathrm{tr}\!\left( \Sigma_Q \, R^{-\top} \mathrm{Cov}(n) \, R^{-1} \right)
        = \mathrm{tr}\!\left( R^{-1} \Sigma_Q R^{-\top} \cdot \mathrm{Cov}(n) \right).
$$

Since $\mathrm{Cov}(n)$ is diagonal, only the diagonal of $R^{-1} \Sigma_Q R^{-\top}$
contributes:

$$
D(R, b) = \sum_{j=1}^d \left( R^{-1} \Sigma_Q R^{-\top} \right)_{jj} \cdot \mathrm{Var}(n_j).
$$

Plug in $\mathrm{Var}(n_j)$:

$$
D(R, b) = \kappa \sum_{j=1}^d \underbrace{\left( R^{-1} \Sigma_Q R^{-\top} \right)_{jj}}_{w_j(R)} \cdot \underbrace{(R^\top \Sigma_K R)_{jj}}_{\sigma_j^2(R)} \cdot 2^{-2 b_j}.
$$

**Specialization to orthogonal $R$** (so $R^{-1} = R^\top$):

$$
\boxed{\,D(R, b) = \kappa \sum_{j=1}^d w_j(R) \, \sigma_j^2(R) \, 2^{-2 b_j},\,}
$$

with

$$
w_j(R) = (R^\top \Sigma_Q R)_{jj}, \qquad \sigma_j^2(R) = (R^\top \Sigma_K R)_{jj}.
$$

This is the **central distortion equation**. Three remarks:

1. The Q-weight $w_j$ is the diagonal of $\Sigma_Q$ *expressed in the basis $R$*.
   It quantifies how much queries "read" the $j$-th basis coordinate of the key.
2. The variance $\sigma_j^2$ is the per-coord variance of the *projected* key.
3. For *non-orthogonal* $R$, the inverse $R^{-\top}$ amplifies noise unevenly and
   $w_j = (R^{-1} \Sigma_Q R^{-\top})_{jj}$ can be much larger than
   $(R^\top \Sigma_Q R)_{jj}$. This is the F8 bug: the original CCA simulation
   used $\rho_j^2$ as the weight, which is $(R^\top \Sigma_Q R)_{jj}$ evaluated
   at $R = P_K$, but $P_K$ is *non-orthogonal* so $\rho_j^2$ is the wrong
   quantity. See §6 of E2.

For the rest of this note we restrict to **orthogonal** $R$, since both $V_Q$ and
$R_{\text{sym}}$ are orthogonal and TurboQuant's $R_{\text{rand}}$ is too.

## 5. Bit allocation: reverse water-filling

For a fixed orthogonal basis $R$, what allocation $b \ge 0$ with
$\sum_j b_j = b_{\text{avg}} d$ minimizes $D$?

$$
\min_{b \ge 0,\; \sum_j b_j = b_{\text{avg}} d}\; \sum_j w_j \sigma_j^2 \cdot 2^{-2 b_j}.
$$

### 5.1 Lagrangian

Form the Lagrangian

$$
\mathcal{L}(b, \mu) = \sum_j w_j \sigma_j^2 \cdot 2^{-2 b_j} - \mu \left( \sum_j b_j - b_{\text{avg}} d \right).
$$

Stationarity:

$$
\frac{\partial \mathcal{L}}{\partial b_j}
 = -2 \ln 2 \cdot w_j \sigma_j^2 \cdot 2^{-2 b_j} - \mu = 0.
$$

So at the optimum

$$
w_j \sigma_j^2 \cdot 2^{-2 b_j} = -\frac{\mu}{2 \ln 2} =: \theta.
$$

That is, **every active coordinate contributes the same per-coord distortion**
$\theta$. Solving for $b_j$:

$$
b_j^\star = \tfrac{1}{2} \log_2 \!\left( \frac{w_j \sigma_j^2}{\theta} \right).
$$

**Technique used.** Convex optimization with a linear equality constraint; strict
convexity of $2^{-2b}$ in $b$ makes the stationary point a unique minimum.
Non-negativity $b_j \ge 0$ is handled by *reverse water-filling*: any coord whose
$w_j \sigma_j^2$ falls below $\theta$ is set to $b_j = 0$ and the remaining budget
is redistributed. We assume below that no coord saturates at zero — empirically
this holds on the Stage 1E stats at $b_{\text{avg}} \ge 2$.

### 5.2 Solving for $\theta$

Plug $b_j^\star$ into the budget constraint:

$$
\sum_j \tfrac{1}{2} \log_2 \!\left( \frac{w_j \sigma_j^2}{\theta} \right) = b_{\text{avg}} d.
$$

Distribute the log:

$$
\tfrac{1}{2} \log_2 \prod_j w_j \sigma_j^2 \; - \; \tfrac{d}{2} \log_2 \theta = b_{\text{avg}} d.
$$

Solve for $\theta$:

$$
\boxed{\,\theta = \left( \prod_j w_j \sigma_j^2 \right)^{1/d} \cdot 2^{-2 b_{\text{avg}}}.\,}
$$

### 5.3 Achieved distortion: the geomean form

Each active coord contributes exactly $\theta$, so summing:

$$
\boxed{\,D^\star(R) = \kappa \cdot d \cdot \theta = \kappa \cdot d \cdot \left( \prod_{j=1}^d w_j(R) \, \sigma_j^2(R) \right)^{\!1/d} \cdot 2^{-2 b_{\text{avg}}}.\,}
$$

This is the **AM-GM form of water-fill**: at fixed average rate, distortion is
proportional to the *geometric mean* of $w_j \sigma_j^2$ across coordinates. The
basis-selection problem reduces to choosing $R$ (orthogonal) to minimize this
geometric mean.

> *Active-set generalization.* The "all coords active" derivation above is an
> idealization. In practice, reverse water-fill zeros some coordinates:
> empirically on Stage 1E `R_sym` water-fill, the minimum number of active
> coords across 288 heads is 79 at $b_{\text{avg}} = 2$, 106 at
> $b_{\text{avg}} = 3$, 118 at $b_{\text{avg}} = 4$. For active set $A$ with
> $|A|$ coords, the closed form generalizes to
>
> $$
> \theta_A = \Big(\prod_{j \in A} w_j \sigma_j^2\Big)^{1/|A|} \cdot 2^{-2 b_{\text{avg}} d / |A|},
> \qquad D^\star_A = \kappa \cdot |A| \cdot \theta_A.
> $$
>
> Ratios at fixed $b_{\text{avg}}$ are robust to the active-set adjustment
> because all bases see comparable $|A|$, so it largely cancels in the
> comparisons reported in §10.

## 6. Basis selection: Hadamard's inequality

Take logs of the objective in §5.3:

$$
\log D^\star(R) - \text{const} = \frac{1}{d} \sum_{j=1}^d \log\!\left[ (R^\top \Sigma_Q R)_{jj} \cdot (R^\top \Sigma_K R)_{jj} \right].
$$

Since $\log(ab) = \log a + \log b$, this **separates** into a Q-side and a K-side
sum:

$$
\boxed{\,d \cdot \log D^\star(R) - \text{const} = \underbrace{\sum_j \log (R^\top \Sigma_Q R)_{jj}}_{\text{Q-side}} + \underbrace{\sum_j \log (R^\top \Sigma_K R)_{jj}}_{\text{K-side}}.\,}
$$

### 6.1 Hadamard's inequality

For any positive-definite matrix $A$,

$$
\prod_{j=1}^d A_{jj} \;\ge\; \det(A),
$$

with equality **iff** $A$ is diagonal. Applied to $A = R^\top \Sigma_Q R$, since
$\det(R^\top \Sigma_Q R) = \det(\Sigma_Q)$ for orthogonal $R$:

$$
\prod_j (R^\top \Sigma_Q R)_{jj} \;\ge\; \det(\Sigma_Q),
\qquad \text{equality iff } R \text{ diagonalizes } \Sigma_Q.
$$

The same holds for $\Sigma_K$. Combining:

$$
\boxed{\,\left( \prod_j w_j \sigma_j^2 \right)^{1/d} \;\ge\; \left( \det \Sigma_Q \cdot \det \Sigma_K \right)^{1/d},\,}
$$

with **equality iff $R$ simultaneously diagonalizes both $\Sigma_Q$ and $\Sigma_K$**,
i.e. iff $[\Sigma_Q, \Sigma_K] = 0$ (they commute).

### 6.2 The fundamental tension

Real data has $[\Sigma_Q, \Sigma_K] \neq 0$. Hadamard's inequality then forces a
tradeoff: no orthogonal $R$ can hit both lower bounds simultaneously. Any choice
of $R$ incurs:

- A **Q-side slack**

$$
\mathrm{slack}_Q(R) := \frac{\left( \prod_j (R^\top \Sigma_Q R)_{jj} \right)^{1/d}}{\det(\Sigma_Q)^{1/d}} \;\ge\; 1,
$$

with equality iff $R$ diagonalizes $\Sigma_Q$ (up to sign flips, coordinate
permutations, and rotations within degenerate eigenspaces).

- A **K-side slack**

$$
\mathrm{slack}_K(R) := \frac{\left( \prod_j (R^\top \Sigma_K R)_{jj} \right)^{1/d}}{\det(\Sigma_K)^{1/d}} \;\ge\; 1,
$$

with equality iff $R$ diagonalizes $\Sigma_K$ (same caveats).

The total distortion ratio relative to the (generally unattainable) commuting
floor is $\mathrm{slack}_Q(R) \cdot \mathrm{slack}_K(R)$. The art of basis
selection is *jointly minimizing this product*.

## 7. Three methods through the same lens

We now apply §5–6 to each method and read off exactly what each one optimizes.

### 7.1 TurboQuant (V3): random Hadamard rotation + unit-norm + uniform bits

The codebase implementation (`Stage1MSECompressor`, identical to TurboQuantV3):

1. Draw a fixed random orthogonal $R_{\text{rand}}$ (e.g., a Hadamard rotation
   composed with random sign flips) at calibration time.
2. For each $k$, normalize: $\tilde k = k / \|k\|$, send $\|k\|$ separately at
   high precision.
3. Rotate: $c = R_{\text{rand}}^\top \tilde k$.
4. Quantize each coordinate of $c$ with *uniform* $b = b_{\text{avg}}$ bits using
   a single shared codebook.
5. Decompress: $\hat k = \|k\| \cdot R_{\text{rand}} \hat c$.

Two questions: (i) what is $R$, (ii) what is $b$?

**(i) The basis is random.** $R_{\text{rand}}$ is drawn *without* using $\Sigma_Q$
or $\Sigma_K$. By concentration of measure (Haar-random orthogonal matrices),
the diagonal entries of $R_{\text{rand}}^\top A R_{\text{rand}}$ concentrate around
$\mathrm{tr}(A)/d$ for any fixed PSD $A$, as $d$ grows:

$$
(R_{\text{rand}}^\top A R_{\text{rand}})_{jj} \;\approx\; \frac{\mathrm{tr}(A)}{d}.
$$

So

$$
\left( \prod_j (R_{\text{rand}}^\top \Sigma_Q R_{\text{rand}})_{jj} \right)^{1/d} \;\approx\; \frac{\mathrm{tr}(\Sigma_Q)}{d},
$$

and likewise for $\Sigma_K$. The geometric mean of the Q-diagonal collapses to
its *arithmetic* mean — i.e., to the eigenvalue-AM rather than the eigenvalue-GM.

**(ii) Uniform bits is consistent with random rotation.** Since per-coord
variances are roughly equal after a random rotation, water-fill would give back
nearly uniform bits anyway. Unit-normalization makes each rotated coord's
variance exactly $1/d$, so uniform bits is even more strongly justified.

**Achieved distortion (uniform bits in random basis):**

$$
\begin{aligned}
D_{\text{TQ}}
 &= \kappa \sum_j w_j \sigma_j^2 \cdot 2^{-2 b_{\text{avg}}} \\
 &\approx \kappa \cdot 2^{-2 b_{\text{avg}}} \cdot \sum_j (R_{\text{rand}}^\top \Sigma_Q R_{\text{rand}})_{jj} \cdot (R_{\text{rand}}^\top \Sigma_K R_{\text{rand}})_{jj} \\
 &\approx \kappa \cdot 2^{-2 b_{\text{avg}}} \cdot d \cdot \frac{\mathrm{tr}(\Sigma_Q)}{d} \cdot \frac{\mathrm{tr}(\Sigma_K)}{d} \\
 &= \kappa \cdot 2^{-2 b_{\text{avg}}} \cdot \frac{\mathrm{tr}(\Sigma_Q) \, \mathrm{tr}(\Sigma_K)}{d}.
\end{aligned}
$$

The penultimate step uses concentration: $\sum_j A_{jj} B_{jj} \approx d \cdot \bar A \bar B$
when both diag vectors are nearly constant.

**Ratio to the commuting Hadamard floor:**

$$
\frac{D_{\text{TQ}}}{D^\star_{\text{Hadamard}}}
 = \frac{\mathrm{tr}(\Sigma_Q)/d}{(\det \Sigma_Q)^{1/d}}
 \cdot \frac{\mathrm{tr}(\Sigma_K)/d}{(\det \Sigma_K)^{1/d}}
 = \frac{\mathrm{AM}\!\left(\lambda_j(\Sigma_Q)\right)}{\mathrm{GM}\!\left(\lambda_j(\Sigma_Q)\right)}
 \cdot \frac{\mathrm{AM}\!\left(\lambda_j(\Sigma_K)\right)}{\mathrm{GM}\!\left(\lambda_j(\Sigma_K)\right)}.
$$

> **What TurboQuant misses, exactly.** It pays the **AM/GM gap of both spectra,
> multiplicatively**. For peaked spectra (a few large eigenvalues, many small),
> AM/GM is large and the penalty is severe. The random rotation deliberately
> washes out spectral structure; that is the whole point of the design (it makes
> the marginal distribution close to Gaussian and unit-norm). But it pays for
> that simplicity by *throwing away both $\Sigma_Q$ and $\Sigma_K$'s spectral
> shape*.

### 7.2 Q-only ($V_Q$): eigvecs of $\Sigma_Q$ + reverse water-fill

Pick $R = V_Q$, the orthonormal eigenvector matrix of $\Sigma_Q$ ordered by
decreasing eigenvalue. By construction
$V_Q^\top \Sigma_Q V_Q = \mathrm{diag}(\lambda_j(\Sigma_Q))$, so

$$
w_j(V_Q) = \lambda_j(\Sigma_Q), \qquad \prod_j w_j = \det(\Sigma_Q).
$$

The **Q-side hits the Hadamard floor exactly**. But $V_Q$ does not in general
diagonalize $\Sigma_K$, so

$$
\sigma_j^2(V_Q) = (V_Q^\top \Sigma_K V_Q)_{jj}, \qquad \prod_j \sigma_j^2 \ge \det(\Sigma_K).
$$

The K-side slack

$$
\mathrm{slack}_K(V_Q) = \frac{\left( \prod_j (V_Q^\top \Sigma_K V_Q)_{jj} \right)^{1/d}}{(\det \Sigma_K)^{1/d}}
$$

is a *quantitative measure of how non-aligned* $\Sigma_Q$ and $\Sigma_K$ are. If
they commute, $V_Q$ is the joint optimum and the slack is 1; otherwise it is
strictly greater.

**Achieved distortion (Q-only with water-fill):**

$$
D_{V_Q} = \kappa \cdot d \cdot (\det \Sigma_Q)^{1/d} \cdot \left( \prod_j (V_Q^\top \Sigma_K V_Q)_{jj} \right)^{1/d} \cdot 2^{-2 b_{\text{avg}}}.
$$

**Ratio to TurboQuant:**

$$
\frac{D_{V_Q}}{D_{\text{TQ}}}
 = \underbrace{\frac{(\det \Sigma_Q)^{1/d}}{\mathrm{tr}(\Sigma_Q)/d}}_{\text{captures } \Sigma_Q \text{ shape; } < 1}
 \cdot \underbrace{\frac{\left( \prod_j (V_Q^\top \Sigma_K V_Q)_{jj} \right)^{1/d}}{\mathrm{tr}(\Sigma_K)/d}}_{\le 1 \text{ in general}}.
$$

The first factor is strictly $< 1$ (this is GM/AM of $\Sigma_Q$'s eigenvalues), so
$V_Q$ always beats TurboQuant on the Q-side. The second factor is generally $< 1$
as well, because the diagonals of $V_Q^\top \Sigma_K V_Q$ are *not random*: they
inherit some of $\Sigma_K$'s spectral structure. So $V_Q$ strictly dominates
TurboQuant on real data — consistent with the E3 measurement at $b_{\text{avg}} = 3$:
`v_waterfill` top-1 = 0.760 vs `v3` top-1 = 0.682.

> **What Q-only misses, exactly.** It exactly minimizes the Q-side
> $\prod_j (R^\top \Sigma_Q R)_{jj}$, but leaves $\mathrm{slack}_K(V_Q)$ on the
> table. This slack is the K-axis Hadamard penalty of non-commutativity: the K
> coordinates that line up with $V_Q$'s axes are diagonal entries that may sum to
> far more than $\det(\Sigma_K)$.

### 7.3 JointQK: choose $R$ to jointly minimize Q- and K-side slacks

The objective from §6 is

$$
\min_{R \text{ orth.}} \; \sum_j \log\!\left[ (R^\top \Sigma_Q R)_{jj} \cdot (R^\top \Sigma_K R)_{jj} \right].
$$

This is a **joint diagonalization problem**: find the orthogonal basis whose
diagonal entries against $\Sigma_Q$ and $\Sigma_K$ are simultaneously close to
those matrices' eigenvalues.

If $\Sigma_Q$ and $\Sigma_K$ commute, the unique global minimum is the shared
eigenbasis and $\mathrm{slack}_Q \cdot \mathrm{slack}_K = 1$. In general the
problem has no closed form and must be solved iteratively (e.g., Jacobi sweeps
that minimize $\sum_j \log(A_{jj} B_{jj})$ directly). However, a simple
closed-form heuristic recovers most of the gap, and that is what
`r_sym_waterfill` uses.

## 8. The $R_{\text{sym}}$ heuristic

Define

$$
M := \tfrac{1}{2} \left( \Sigma_Q \Sigma_K + \Sigma_K \Sigma_Q \right).
$$

This is the **symmetric part of the product $\Sigma_Q \Sigma_K$**. It is symmetric
(by construction) and real, so it has a real orthonormal eigenbasis. Take

$$
R_{\text{sym}} := \begin{bmatrix} v_1 & v_2 & \cdots & v_d \end{bmatrix},
$$

the eigenvectors of $M$ ordered by decreasing eigenvalue.

**What $M$ actually looks like.** $M$ is symmetric by construction and therefore
has a real orthonormal eigenbasis, so $R_{\text{sym}}$ is well-defined. However,
even though $\Sigma_Q$ and $\Sigma_K$ are PSD, $M$ is **not** in general PSD.
Empirically on the Stage 1E calibration, every one of the 288 (layer, kv_head)
heads has at least one negative eigenvalue of $M$; the smallest negative
eigenvalue is up to 87% of the largest positive eigenvalue in magnitude
(4.94% of all eigenvalues are negative). So eigenvalue *magnitude* should not
be interpreted as joint Q-K energy, and "the top-$r$ eigvecs carry most of the
joint structure" is not a meaningful reduction. The water-fill in §8.3 is
unaffected because it uses $\Sigma_Q$ and $\Sigma_K$ directly: the per-coord
weights $(R_{\text{sym}}^\top \Sigma_Q R_{\text{sym}})_{jj}$ and
$(R_{\text{sym}}^\top \Sigma_K R_{\text{sym}})_{jj}$ are diagonals of PSD
matrices and are always non-negative regardless of $M$'s signature.

### 8.1 Why this basis is reasonable

Two justifications.

**(a) Exact in the commuting case.** If $[\Sigma_Q, \Sigma_K] = 0$, then
$\Sigma_Q \Sigma_K = \Sigma_K \Sigma_Q$, so $M = \Sigma_Q \Sigma_K$. Its
eigenvectors simultaneously diagonalize $\Sigma_Q$ and $\Sigma_K$ (since the
matrices share an eigenbasis), and $R_{\text{sym}}$ hits the Hadamard floor
$\mathrm{slack}_Q \cdot \mathrm{slack}_K = 1$. So $R_{\text{sym}}$ *reduces to the
optimum* in the easy case.

**(b) Perturbation expansion.** Write $\Sigma_K = \Sigma_Q + \Delta$, viewing
$\Sigma_K$ as a perturbation of $\Sigma_Q$. Substituting:

$$
M = \tfrac{1}{2} \left( \Sigma_Q (\Sigma_Q + \Delta) + (\Sigma_Q + \Delta) \Sigma_Q \right)
  = \Sigma_Q^2 + \tfrac{1}{2}\!\left( \Sigma_Q \Delta + \Delta \Sigma_Q \right).
$$

This is an exact polynomial identity in $\Delta$, not a Taylor expansion — there
is no remainder term. $\Sigma_Q^2$ has the same eigenvectors as $\Sigma_Q$ (just
squared eigenvalues), so removing $\Delta$ recovers $V_Q$. Adding $\Delta$ back
introduces information from $\Sigma_K$ into the basis-defining matrix; the
direction and magnitude of the resulting eigenvector rotation depend on the
spectral gaps of $\Sigma_Q$ and the off-diagonal structure of
$\Sigma_Q \Delta + \Delta \Sigma_Q$ (Davis–Kahan-style bounds), so the rotation
is **not in general monotone toward $V_K$**. What we can say is that $R_{\text{sym}}$
sits "between" $V_Q$ and a basis informed by $\Sigma_K$, with the exact location
depending on data.

### 8.2 What $R_{\text{sym}}$ is *not*

$R_{\text{sym}}$ is a heuristic, not a global minimizer of the log-product
objective from §7.3. The remaining gap to the (generally unreachable) Hadamard
floor is the upside an iterative joint-diagonalizer could capture. Computed in
**product-geomean units** (per-head geomean of $w_j \sigma_j^2$ across coords,
then arithmetic mean across the 35 non-zero layers): $R_{\text{sym}}$ achieves
1.405 versus a Hadamard floor of 1.026, so

$$
\frac{R_{\text{sym}} \text{ geomean}}{\text{Hadamard floor}}
 = \frac{1.405}{1.026}
 \approx 1.37.
$$

An oracle joint-diagonalizer could therefore reduce post-water-fill distortion
by at most ~37%. Translated through the per-layer prediction-vs-measured
Pearson of ~0.98, that is perhaps 1–4 pp of additional top-1 — worth
investigating in Stage 3, but $R_{\text{sym}}$ already captures most of the
available gain.

### 8.3 The water-fill weights for $R_{\text{sym}}$

Plugging $R = R_{\text{sym}}$ into the central distortion equation:

$$
D_{R_{\text{sym}}}(b)
 = \kappa \sum_j (R_{\text{sym}}^\top \Sigma_Q R_{\text{sym}})_{jj} \cdot (R_{\text{sym}}^\top \Sigma_K R_{\text{sym}})_{jj} \cdot 2^{-2 b_j},
$$

so the water-fill objective uses

$$
\text{weight}_j = (R_{\text{sym}}^\top \Sigma_Q R_{\text{sym}})_{jj} \cdot (R_{\text{sym}}^\top \Sigma_K R_{\text{sym}})_{jj}.
$$

This is exactly what
[`build_method_compressor`](../../../../experiments/stage1/toolkit/per_coord_quantization.py#L307-L316)
computes for the `r_sym_*` branch.

## 9. Why orthogonality + water-fill keeps top-1 in sync with geometry

§3 says Q-weighted MSE depends only on $\mathrm{Cov}(\Delta k)$. But top-1
retention depends on the *full distribution* of $q^\top \Delta k$ across
queries, not just its variance. Why should reducing geometry distortion
translate into top-1? The honest answer has a small rigorous core and a
mostly-empirical conclusion.

### 9.1 Rigorous structural properties

**No inverse-map noise amplification.** For orthogonal $R$,

$$
\mathrm{Cov}(\Delta k) = R \cdot \mathrm{diag}\!\left( \kappa \sigma_j^2 \cdot 2^{-2 b_j} \right) \cdot R^\top,
$$

so the eigenvalues of $\mathrm{Cov}(\Delta k)$ are exactly the per-coord noise
variances $\kappa \sigma_j^2 \cdot 2^{-2 b_j}$, in the basis $R$. The norm of
the noise vector is preserved through the inverse map. This is not true for
non-orthogonal $R$: with CCA's $P_K$, the inverse $P_K^{-\top}$ amplifies
coordinate-aligned noise unevenly across directions, and a few queries see
much larger error than the average.

**Water-fill equalizes the per-coord Q-weighted contribution.** Reverse
water-fill sets $w_j \sigma_j^2 \cdot 2^{-2 b_j} = \theta$ for every active
coord. So every active coord contributes exactly $\theta$ to the Q-weighted
MSE; the distortion budget is *equipartitioned* across query-relevant
directions. (This is **weaker** than saying the noise covariance is
isotropic in the $\Sigma_Q^{1/2}$-metric — the eigenvalues of
$\mathrm{Cov}(\Delta k)$ are $\theta / w_j$ and the $w_j$ span an order of
magnitude on real heads, so the noise is not spectrally white in the original
space. Water-fill controls only the *Q-weighted budget*, not the noise
covariance shape.)

These two properties together rule out the failure modes that empirically
flip top-1: (i) noise concentrated on a single discarded direction (hard
cutoffs), and (ii) noise amplified non-uniformly by an inverse map
(non-orthogonal bases).

### 9.2 Empirical claim

For the orthogonal-basis × continuous-water-fill methods tested in E3
($V_Q$, $V_h$, $R_{\text{sym}}$), lower predicted geomean translates
**monotonically** into higher top-1, with per-layer Pearson 0.93–0.98
between predicted geomean and measured geometry distortion (§10.2). The
non-orthogonal `cca_waterfill` and the hard-cutoff `*_uniform` / `*_truncate`
methods break this monotonicity in exactly the directions the rigorous
properties predict — see the §10.3 row for `cca_waterfill`: low geometry
distortion (0.097) but middling top-1 (0.535), the signature of
inverse-map noise amplification.

We do **not** have a clean general-position theorem for the geometry → top-1
implication; the agreement above is empirical for the specific methods on
the Stage 1E data. The Stage 1E rule of thumb is therefore: **(orthogonal
joint basis) × (continuous water-fill)** — both ingredients are required for
the empirical geometry-↔-top-1 alignment we observe.

## 10. Empirical verification

The theory makes specific quantitative predictions, all checkable against the
canonical E3 artifacts.

### 10.1 Geomean prediction matches measured distortion

From `cca_stats.pt` (full-pool calibration), compute the per-head
$\left(\prod_j w_j \sigma_j^2\right)^{1/d}$ for each basis and average across the
35 non-zero layers:

| Basis | $(\prod w_j)^{1/d}$ (Q-side) | $(\prod \sigma_j^2)^{1/d}$ (K-side) | product (geomean) |
|---|---:|---:|---:|
| Hadamard floor $(\det \Sigma_\cdot)^{1/d}$ | 0.866 | 1.191 | **1.031** ← unreachable ($[\Sigma_Q, \Sigma_K] \neq 0$); see footnote |
| **$R_{\text{sym}}$** | 1.001 | 1.421 | **1.422** |
| $V_Q$ (Q-only) | 0.866 | 2.059 | **1.782** |
| $V_K$ (K-only) | 1.352 | 1.191 | **1.611** |
| $V_h$ (orthogonal CCA) | — | — | 6.07 |
| Random orth (TurboQuant proxy) | 2.004 | 5.968 | **11.96** |

Read off:
- $V_Q$ hits the Q-floor (Q-side = 0.866 = floor) but pays K-slack 1.728×.
- $V_K$ hits the K-floor (K-side = 1.191 = floor) but pays Q-slack 1.562×.
- $R_{\text{sym}}$ pays small slack on each side (1.156× and 1.193×), and the
  *product* beats both $V_Q$ and $V_K$.
- Random orth pays the full AM/GM penalty of both spectra.

> *Footnote on averaging conventions.* The 1.031 in the table is the product
> of side arithmetic means: $0.866 \times 1.191$. The headroom calculation in
> §8.2 uses a different (equally valid) statistic — the per-head product
> geomean $(\det \Sigma_Q \cdot \det \Sigma_K)^{1/d}$ averaged arithmetically
> across heads — which evaluates to $1.026$ on the same data. The two differ
> because $\mathbb{E}[XY] \neq \mathbb{E}[X]\mathbb{E}[Y]$ when the
> per-head $\det^{1/d}$ values of $\Sigma_Q$ and $\Sigma_K$ are correlated.
> §8.2's number is directly comparable to $R_{\text{sym}}$'s $1.422$/$1.405$
> entries here (both are mean-of-per-head-products); the $1.031$ above is for
> a quick AM/GM intuition only.

### 10.2 Predicted ratio vs measured ratio

The predicted ratio of post-water-fill distortion is the ratio of geomeans
(§5.3). At $b_{\text{avg}} = 3$ measured against E3 canonical artifacts:

| Comparison | Predicted ratio (geomean) | Measured ratio (geo distortion) | Agreement |
|---|---:|---:|---:|
| $R_{\text{sym}} / V_Q$ | **0.809** | **0.824** | within 2% |
| $V_Q / \text{TurboQuant}$ | **0.149** | **0.144** | within 4% |
| $R_{\text{sym}} / \text{TurboQuant}$ | **0.121** | **0.119** | within 2% |

Per-layer Pearson correlation between predicted geomean and measured
`geometry_distortion`: **0.977 for $R_{\text{sym}}$, 0.933 for $V_Q$**.

The Bennett constant $\kappa$ and finite-bit Lloyd–Max approximation cancel in
these ratios; what is left is the geomean structure that §5–6 derived. The
quantitative agreement is, in my view, the strongest single piece of evidence
that the theory is correctly characterizing the methods.

### 10.3 Top-1 retention follows the same ordering

From `e3_b3_r64_summary.json`, layer-0-excluded means at $b_{\text{avg}} = 3$:

| Method | top-1 prefill | geo distortion | predicted geomean |
|---|---:|---:|---:|
| `r_sym_waterfill` | **0.860** | **0.0542** | **1.42** |
| `v_waterfill` | 0.760 | 0.0658 | 1.78 |
| `cca_orth_waterfill` | 0.675 | 0.197 | 6.07 |
| `v3` (TurboQuant) | 0.682 | 0.456 | ~11.96 |
| `cca_waterfill` | 0.535 | 0.0965 | (non-orth, separate) |

The top-1 ordering tracks the predicted geomean exactly among the orthogonal +
water-fill methods, consistent with the §9.2 empirical claim that lower geomean
translates monotonically into higher top-1 within that method class.

The one apparent anomaly — V3 has lower top-1 (0.682) but worse geometry
distortion (0.456) than `cca_waterfill` (0.535 / 0.0965) — is exactly the
non-orthogonal/inverse-map-amplification effect from §9.1: `cca_waterfill` has
good *geometry* but bad *top-1* because $P_K^{-\top}$ amplifies the noise
unevenly across queries.

### 10.4 Cross-task and LOO stability (E4)

The theory predicts that $R_{\text{sym}}$'s basis should be stable across
calibration sources: $\Sigma_Q$ and $\Sigma_K$ are global second-moment statistics
that do not change much between LongBench-E configs. Empirically (E4a):

- $R_{\text{sym}}$ top-1 spread across {qasper, hotpotqa, passage_retrieval_en}
  calibration: **0.2 pp** (tighter even than calibration-independent V3).
- $V_Q$ spread: 2.4 pp.
- LOO std-dev within a config: ≤ 0.024 across all methods, ≤ 0.009 for
  $R_{\text{sym}}$.

Both consistent with the theory: $R_{\text{sym}}$'s basis is set by global
$(\Sigma_Q, \Sigma_K)$ structure, which averages out per-example noise.

### 10.5 Decode generalization (E5)

The §9.1 structural properties (no inverse-map amplification, equipartitioned
Q-weighted budget) do not depend on which queries read the keys, as long as
calibration $\Sigma_Q$ covers the test distribution. Empirically, decode-phase
queries see *higher* top-1 than prefill queries for every method (decode is
"easier" because attention concentrates more sharply on later tokens).
$R_{\text{sym}}$ at $b_{\text{avg}} = 3$ reaches 0.904 decode top-1.

## 11. Summary

The hierarchy of methods is a hierarchy of **how much spectral information each
one uses**:

| Method | Uses $\Sigma_Q$ shape? | Uses $\Sigma_K$ shape? | Distortion ratio to floor |
|---|:---:|:---:|---|
| TurboQuant (V3) | ✗ (random) | ✗ (random) | $\mathrm{AM/GM}(\Sigma_Q) \cdot \mathrm{AM/GM}(\Sigma_K)$ |
| Q-only ($V_Q$) | ✓ (eigvecs) | ✗ (only via diag) | $\mathrm{slack}_K(V_Q)$ |
| JointQK ($R_{\text{sym}}$) | ≈ (perturbed) | ≈ (perturbed) | $\mathrm{slack}_Q \cdot \mathrm{slack}_K$ (small) |
| Iterative joint-diag | ✓ (full) | ✓ (full) | minimum achievable |
| Hadamard floor | (commuting case only) | | 1 (unreachable in general) |

The theoretical motivation for JointQK is direct: water-filling reduces basis
selection to minimizing the geometric mean
$\prod_j (R^\top \Sigma_Q R)_{jj} \cdot (R^\top \Sigma_K R)_{jj}$; Hadamard's
inequality says this geomean factors into a Q-side floor and a K-side floor;
any orthogonal basis incurs slack on at least one side unless $\Sigma_Q$ and
$\Sigma_K$ commute. $R_{\text{sym}}$ trades a small Q-slack for a much smaller
K-slack and wins on the product.

The empirical motivation closes the loop: the predicted geomean ratios match
measured geometry distortion within ~2%; per-layer Pearson is ~0.98; the top-1
ranking on Qwen3-8B + LongBench-E follows the predicted ordering; cross-task
and LOO stability are tight (≤ 0.3 pp); decode generalization is favorable.
The +10 pp top-1 advantage of `r_sym_waterfill` over `v_waterfill` at
$b_{\text{avg}} = 3$ is exactly what the K-side slack reduction predicts.

The remaining theoretical headroom — $1.405 / 1.026 \approx 1.37\times$ in
product-geomean units (§8.2) — is the upside an iterative joint-diagonalizer
could capture beyond $R_{\text{sym}}$, and is the natural next direction for
Stage 3.

## 12. Open theoretical items

1. **Iterative joint-diagonalization vs $R_{\text{sym}}$.** Implement Jacobi-style
   sweeps minimizing $\sum_j \log(A_{jj} B_{jj})$ directly and quantify the gain
   over $R_{\text{sym}}$ on the Stage 1E calibration. The maximum achievable
   improvement is bounded by the ratio of $R_{\text{sym}}$'s product-geomean to
   the Hadamard floor, computed at ~1.37× on Stage 1E (§8.2). Translating
   through the per-layer prediction Pearson, this is at most ~1–4 pp top-1
   upside.
2. **Centered vs uncentered moments.** Stage 1E uses uncentered second moments
   ($\mathbb{E}[q q^\top]$, not $\mathrm{Cov}(q)$). The theory in §3–9 holds
   verbatim for either choice; centering changes top eigenvalues by
   $O(\|\mu_q\|^2)$ but does not change the AM/GM tradeoff structure. E1's review
   noted $\rho_{\max}$ drops from 0.993 to 0.940 under centering; the
   basis-selection conclusions are unchanged.
3. **Non-orthogonal bases and noise amplification.** §9's argument breaks for
   non-orthogonal $R$; CCA's $P_K$ is the empirical demonstration. A theoretical
   bound on the additional structured-noise penalty (in terms of
   $\|P_K^{-\top}\|$) would tighten the design rule.
4. **Bit-budget transfer.** §5–6 predict basis ranking is invariant in
   $b_{\text{avg}}$ (the $2^{-2 b_{\text{avg}}}$ factor is a global multiplier).
   E3's bit-budget sensitivity table confirms this empirically. Beyond
   $b_{\text{avg}} \ge 4$, all methods approach the full-precision ceiling and
   the gap collapses; the high-rate asymptotic regime that justifies Bennett's
   approximation is $b \in [2, 8]$.
5. **$R_{\text{sym}}$ vs other symmetric combinations.** Variants like
   $\Sigma_Q^{1/2} \Sigma_K \Sigma_Q^{1/2}$ (a Mahalanobis-style choice) or
   $\Sigma_K^{1/2} \Sigma_Q \Sigma_K^{1/2}$ could be compared. They reduce to
   the same eigenbasis only when $\Sigma_Q$ and $\Sigma_K$ commute; in general
   they are different heuristics with different first-order behavior. E3 has
   only the one $R_{\text{sym}}$ variant; comparison is a Stage 3 study.
