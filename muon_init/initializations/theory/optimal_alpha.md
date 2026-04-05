# Optimal Scale Factor for Scaled Orthogonal Initialization Under Muon

## 1. Setup and Notation

Consider a depth-$L$ feedforward network with weight matrices $W_\ell \in \mathbb{R}^{m_\ell \times n_\ell}$ for $\ell = 1, \dots, L$, pointwise activation functions $\phi_\ell$, and the Muon optimizer which orthogonalizes every update via Newton-Schulz (NS) iteration.

We initialize each weight as:
$$W_\ell = \alpha_\ell \cdot U_\ell V_\ell^\top$$
where $U_\ell, V_\ell$ are drawn from the Haar measure on the Stiefel manifold (i.e., random partial isometries), and $\alpha_\ell > 0$ is a scalar scale factor. All singular values of $W_\ell$ are exactly $\alpha_\ell$.

We begin with the uniform case $\alpha_\ell = \alpha$ for all layers, then treat the per-layer case.

**Goal.** Find $\alpha^*$ that simultaneously satisfies:
1. Signal propagation (dynamical isometry) through the network at initialization.
2. Zero polar error (Muon's NS iteration has nothing to correct).
3. Spectral norm ball membership (weight decay equilibrium is immediate).
4. Good conditioning of the first Muon step.

---

## 2. Polar Error Is Zero by Construction

**Claim.** For any $\alpha > 0$, the polar factor of $W = \alpha \cdot Q$ where $Q$ is a partial isometry is exactly $Q$ itself.

**Proof.** The polar decomposition of $W$ is $W = P \cdot H$ where $P$ is a partial isometry and $H = (W^\top W)^{1/2}$. We have $W^\top W = \alpha^2 Q^\top Q = \alpha^2 I_{n}$ (assuming $m \geq n$; the $m < n$ case is symmetric). Thus $H = \alpha I_n$ and $P = W H^{-1} = \alpha Q \cdot \alpha^{-1} I_n = Q$. $\square$

**Consequence for Muon.** The Newton-Schulz iteration approximates the polar factor. If the input is already a scaled isometry, NS converges in 0 effective iterations — the iterates stay at $Q$ from the first step. This means:
- No approximation error at initialization.
- No wasted NS work on step 0.
- The first Muon update operates on a perfectly conditioned starting point.

This holds for **any** $\alpha > 0$. Polar error does not constrain $\alpha$.

---

## 3. Signal Propagation Analysis

This is the primary constraint on $\alpha$. We analyze how signals propagate through the network at initialization.

### 3.1 Linear Network (No Activation)

For a depth-$L$ linear network $f(x) = W_L W_{L-1} \cdots W_1 x$, the end-to-end Jacobian is:
$$J = W_L W_{L-1} \cdots W_1$$

With $W_\ell = \alpha Q_\ell$ where each $Q_\ell$ is a partial isometry:
$$J = \alpha^L \cdot Q_L Q_{L-1} \cdots Q_1$$

The product $Q_L \cdots Q_1$ is itself a partial isometry (products of partial isometries with compatible dimensions remain partial isometries). Therefore **all singular values of $J$ are exactly $\alpha^L$**.

**Dynamical isometry** requires $\sigma_i(J) \approx 1$ for all $i$. This gives:
$$\alpha^L = 1 \implies \boxed{\alpha_{\text{linear}} = 1}$$

For $\alpha > 1$: exponential signal explosion ($\alpha^L \to \infty$).
For $\alpha < 1$: exponential signal decay ($\alpha^L \to 0$).
For $\alpha = 1$: perfect isometry at any depth.

### 3.2 ReLU Networks

ReLU($z$) = max(0, $z$) zeroes out roughly half the pre-activation entries. For a single layer with pre-activation $z = W_\ell h_{\ell-1}$:

$$h_\ell = \text{ReLU}(W_\ell h_{\ell-1}) = D_\ell W_\ell h_{\ell-1}$$

where $D_\ell = \text{diag}(\mathbf{1}[z_i > 0])$ is the (random) diagonal mask with roughly half the entries equal to 1.

**Variance propagation.** For a single neuron:
$$\text{Var}(h_\ell^{(i)}) = \frac{1}{2} \sum_j w_{\ell,ij}^2 \cdot \text{Var}(h_{\ell-1}^{(j)})$$

The factor $1/2$ comes from ReLU zeroing out negative pre-activations (for symmetric input distributions). With all singular values of $W_\ell$ equal to $\alpha$:
$$\|W_\ell\|_F^2 = n_\ell \cdot \alpha^2 \quad (\text{for } m_\ell \times n_\ell \text{ matrix with } \min(m,n) = n_\ell)$$

More precisely, each row of $W_\ell$ has squared norm $\alpha^2 \cdot (n_\ell / m_\ell)$ when $m_\ell \geq n_\ell$, or $\alpha^2$ when $m_\ell \leq n_\ell$. For the square case ($m = n$) each row has squared norm $\alpha^2$.

The variance propagation per layer is:
$$\text{Var}(h_\ell) = \frac{\alpha^2}{2} \cdot \text{Var}(h_{\ell-1})$$

The per-layer gain factor is $g = \alpha^2 / 2$. Over $L$ layers:
$$\text{Var}(h_L) = \left(\frac{\alpha^2}{2}\right)^L \cdot \text{Var}(h_0)$$

**Variance preservation** ($g = 1$) requires:
$$\frac{\alpha^2}{2} = 1 \implies \boxed{\alpha_{\text{ReLU}} = \sqrt{2} \approx 1.414}$$

This is exactly the Kaiming condition, but now applied to a matrix with **uniform** singular values rather than a random matrix whose singular values follow the Marchenko-Pastur distribution. The crucial difference: Kaiming init satisfies this condition *in expectation over the random matrix*, while scaled orthogonal satisfies it *exactly for every realization*. There is zero variance in the per-layer gain.

**Jacobian analysis.** The Jacobian of layer $\ell$ is $J_\ell = D_\ell W_\ell$. Since $D_\ell$ zeros out rows, the singular values of $J_\ell$ are either $\alpha$ (for active neurons) or 0 (for dead neurons). The non-zero singular values are **all equal to $\alpha$**.

The end-to-end Jacobian $J = \prod_\ell D_\ell W_\ell$ has non-zero singular values that are exactly $\alpha^L$ (along surviving signal paths). This means:
- **No singular value spread** among active paths — in stark contrast to Kaiming/Xavier where the Marchenko-Pastur spread causes exponentially growing condition numbers with depth.
- The condition number of $J$ restricted to the active subspace is exactly 1.

### 3.3 GELU Networks

GELU($z$) = $z \cdot \Phi(z)$ where $\Phi$ is the standard normal CDF. For pre-activations drawn from $\mathcal{N}(0, \sigma^2)$:

$$\mathbb{E}[\text{GELU}(z)^2] = c_{\text{GELU}} \cdot \sigma^2$$

The constant $c_{\text{GELU}}$ can be computed analytically. Since GELU$(z) \approx z$ for $|z| \gg 0$ and GELU$(z) \approx 0$ for $z \ll 0$:

$$c_{\text{GELU}} = \mathbb{E}\left[z^2 \Phi(z)^2\right] / \mathbb{E}[z^2] \approx 0.425$$

(Numerical evaluation gives $c_{\text{GELU}} \approx 0.4252$ for standard normal inputs.)

The variance preservation condition becomes:
$$\alpha^2 \cdot c_{\text{GELU}} = 1 \implies \boxed{\alpha_{\text{GELU}} = 1/\sqrt{c_{\text{GELU}}} \approx 1.534}$$

### 3.4 General Activation Function

For any pointwise activation $\phi$ and zero-mean Gaussian pre-activations with variance $\sigma^2$, define the **activation gain**:
$$c_\phi = \frac{\mathbb{E}[\phi(z)^2]}{\mathbb{E}[z^2]}, \quad z \sim \mathcal{N}(0, 1)$$

| Activation | $c_\phi$ | $\alpha^* = 1/\sqrt{c_\phi}$ |
|---|---|---|
| Identity (linear) | 1.0 | 1.0 |
| ReLU | 0.5 | $\sqrt{2} \approx 1.414$ |
| GELU | $\approx 0.4252$ | $\approx 1.534$ |
| SiLU / Swish | $\approx 0.3024$ | $\approx 1.818$ |
| Tanh | $\approx 0.3926$ | $\approx 1.596$ |
| Leaky ReLU($a$) | $(1+a^2)/2$ | $\sqrt{2/(1+a^2)}$ |

The general formula is:
$$\boxed{\alpha^*_{\text{signal}} = \frac{1}{\sqrt{c_\phi}}}$$

### 3.5 Sensitivity to Misspecification

How bad is it to use the wrong $\alpha$? The end-to-end gain after $L$ layers is:
$$G(L) = (\alpha^2 c_\phi)^L$$

The signal propagation error (log-deviation from isometry) grows linearly in depth:
$$\log G(L) = L \cdot \log(\alpha^2 c_\phi)$$

For the 8-layer MLP with ReLU:
| $\alpha$ | $\alpha^2 / 2$ | $G(8)$ | Signal status |
|---|---|---|---|
| 0.5 | 0.125 | $5.96 \times 10^{-8}$ | Catastrophic decay |
| 0.75 | 0.281 | $1.50 \times 10^{-5}$ | Severe decay |
| 1.0 | 0.5 | 0.0039 | Strong decay |
| $\sqrt{2}$ | 1.0 | 1.0 | **Perfect preservation** |
| 1.5 | 1.125 | 2.57 | Mild growth |
| 2.0 | 2.0 | 256 | Severe explosion |
| 2.5 | 3.125 | $2.33 \times 10^4$ | Catastrophic explosion |

**Key insight.** At $\alpha = 1$ (pure orthogonal), the 8-layer ReLU MLP attenuates signals by $256\times$. This is why `nn.init.orthogonal_` underperforms Kaiming on deep ReLU networks despite being "clean" spectrally. The fix is trivial: multiply by $\sqrt{2}$.

---

## 4. Spectral Norm Ball Constraint

Muon's weight decay creates an implicit spectral norm constraint. The AdamW-style weight decay update is:
$$W \leftarrow (1 - \eta \lambda) W - \eta \cdot \text{NS}(\text{momentum})$$

At equilibrium, weight decay shrinks all singular values by $(1 - \eta\lambda)$ per step while the orthogonalized update adds $\eta$ to the spectral norm. The equilibrium spectral norm is approximately:
$$\|W\|_{\text{op}} \approx \frac{1}{\lambda}$$

(This is the "spectral norm ball" $\mathcal{B}_{1/\lambda}$ discussed in the Muon literature.)

**Phase 1** is the transient period where weight matrices are outside this ball and weight decay is actively shrinking them. If $\alpha > 1/\lambda$, Phase 1 has nonzero duration — the network starts outside the equilibrium manifold.

**Constraint on $\alpha$:**
$$\alpha \leq \frac{1}{\lambda}$$

For the MLP config: $\lambda_{\text{muon}} = 0$ (no weight decay on Muon params), so this constraint is vacuous. For configs with Muon weight decay:

| Weight decay $\lambda$ | Max $\alpha$ | Compatible with $\alpha_{\text{ReLU}} = \sqrt{2}$? |
|---|---|---|
| 0 | $\infty$ | Yes |
| 0.01 | 100 | Yes |
| 0.1 | 10 | Yes |
| 1.0 | 1.0 | **No** — must use $\alpha = 1$ |

In practice, Muon weight decay is typically small ($\lambda \leq 0.01$) or zero, so this constraint is almost never binding. When it is binding ($\lambda \geq 1/\alpha^*_{\text{signal}}$), the network is so heavily regularized that initialization matters less.

**If the constraint binds:** set $\alpha = 1/\lambda$ and accept the signal propagation mismatch. The Phase 1 duration is then 0, which may be worth the sub-optimal signal propagation.

---

## 5. First Muon Step Conditioning

Beyond signal propagation, we want the first Muon step to be well-conditioned. Let's analyze what happens.

### 5.1 Gradient Structure at Initialization

At step 0, the gradient $\nabla_{W_\ell} \mathcal{L}$ has some singular value spectrum $\{\sigma_i^{(G)}\}$. Muon computes:
1. Momentum buffer $M_0 = G$ (first step, buffer is zero-initialized).
2. With Nesterov: update = $G \cdot (1-\beta) + G \cdot \beta = G$ (simplifies since buf was zero).
3. Newton-Schulz orthogonalization: $\hat{U} = \text{NS}(G) \approx \text{polar}(G)$.
4. Weight update: $W_1 = W_0 - \eta \hat{U}$.

The effective update is:
$$W_1 = \alpha Q - \eta \cdot \text{polar}(G)$$

Both $Q$ and $\text{polar}(G)$ are partial isometries. The question is how they interact.

### 5.2 Why Uniform Singular Values Help

Under Kaiming initialization, $W_0$ has singular values following the Marchenko-Pastur distribution:
$$\sigma_i(W_0) \in [\sigma_{\min}, \sigma_{\max}], \quad \frac{\sigma_{\max}}{\sigma_{\min}} = \frac{(\sqrt{r}+1)}{(\sqrt{r}-1)} \quad \text{where } r = m/n$$

For the 8-layer MLP with $256 \times 256$ weights ($r = 1$), the condition number diverges. In practice, for finite matrices, $\kappa(W_0) \approx O(\sqrt{n})$.

This means different singular directions get vastly different effective learning rates on the first step. Directions corresponding to large singular values of $W_0$ undergo small relative changes, while those corresponding to small singular values undergo large relative changes.

With scaled orthogonal init, $\kappa(W_0) = 1$ exactly. **Every direction gets the same effective learning rate on the first step.** This is the ideal starting condition for any optimizer, and especially for Muon which orthogonalizes updates anyway.

### 5.3 Gradient Effective Rank

The gradient at initialization $G = \nabla_{W_\ell} \mathcal{L}$ depends on the loss landscape at the initial point. For cross-entropy loss on CIFAR-10 with random predictions:

$$G_\ell = \delta_\ell h_{\ell-1}^\top$$

where $\delta_\ell$ is the backpropagated error and $h_{\ell-1}$ is the layer input. With scaled orthogonal init:
- $h_{\ell-1}$ has near-uniform variance across dimensions (due to isometric signal propagation at optimal $\alpha$).
- $\delta_\ell$ also has near-uniform variance (due to isometric gradient propagation).

This means $G_\ell$ has higher effective rank than under Kaiming init, where the Marchenko-Pastur spread in $W$ creates non-uniform activation statistics. A higher effective rank gradient means Muon's orthogonalization step extracts more useful information per update.

---

## 6. Interaction with Normalization Layers

### 6.1 BatchNorm

BatchNorm normalizes pre-activations to zero mean and unit variance before applying learnable scale $\gamma$ and shift $\beta$. This means:

$$\text{BN}(W_\ell h_{\ell-1}) = \gamma_\ell \cdot \frac{W_\ell h_{\ell-1} - \mu}{\sigma} + \beta_\ell$$

The normalization **absorbs the scale $\alpha$** — for signal propagation purposes, any $\alpha$ gives the same forward pass after the first batch. However:
1. The gradient computation *does* depend on $\alpha$ through the BN backward pass.
2. The polar error is still zero regardless of $\alpha$.
3. The spectral norm ball constraint still applies.

**Prediction:** With BatchNorm, $\alpha$ has weak effect on training dynamics. The polar error = 0 advantage persists, but the signal propagation advantage is largely negated. $\alpha = 1$ is simplest and sufficient.

### 6.2 LayerNorm

LayerNorm normalizes across the feature dimension:
$$\text{LN}(z) = \gamma \cdot \frac{z - \mu_z}{\sigma_z} + \beta$$

Like BatchNorm, this absorbs the scale. However, LayerNorm's backward pass propagates gradients differently:

$$\frac{\partial \text{LN}}{\partial z} = \frac{\gamma}{\sigma_z} \left(I - \frac{1}{d}\mathbf{1}\mathbf{1}^\top - \frac{\hat{z}\hat{z}^\top}{d}\right)$$

The scale $\alpha$ affects $\sigma_z$ at init (before the first step), which affects the gradient Jacobian. But after one step, the weights have been updated and the original $\alpha$ is irrelevant.

**Prediction:** With LayerNorm, $\alpha$ matters only for the first few steps. The polar error = 0 advantage persists throughout training. $\alpha = 1$ is again sufficient.

### 6.3 No Normalization (Our Primary Target)

Without normalization layers, $\alpha$ directly controls signal propagation at all depths, and the derived $\alpha^* = 1/\sqrt{c_\phi}$ is critically important. This is why the 8-layer MLP without BatchNorm is the primary experiment — it has maximal initialization sensitivity.

---

## 7. The Unified Formula

Combining all constraints, the optimal $\alpha$ is:

$$\boxed{\alpha^* = \min\left(\frac{1}{\sqrt{c_\phi}},\; \frac{1}{\lambda}\right)}$$

where:
- $c_\phi$ is the activation gain (Section 3.4)
- $\lambda$ is the Muon weight decay (Section 4)

In the common case $\lambda = 0$ (no Muon weight decay), this simplifies to:
$$\alpha^* = \frac{1}{\sqrt{c_\phi}}$$

**Concrete values for the experiment configs:**

| Config | Activation | BatchNorm/LayerNorm | $\lambda_{\text{muon}}$ | $\alpha^*$ | Notes |
|---|---|---|---|---|---|
| `mlp_cifar10` | ReLU | None | 0 | $\sqrt{2} \approx 1.414$ | Most init-sensitive |
| `resnet_cifar10` | ReLU | BatchNorm | 0 | 1.0 (BN absorbs scale) | Weak $\alpha$ dependence |
| `vit_cifar10` | GELU | LayerNorm | 0 | 1.0 (LN absorbs scale) | Weak $\alpha$ dependence |
| `nanogpt_owt` | GELU | LayerNorm | 0 | 1.0 (LN absorbs scale) | Weak $\alpha$ dependence |

---

## 8. Per-Layer $\alpha_\ell$ Analysis

### 8.1 When Uniform $\alpha$ Suffices

For networks where all hidden layers have the same activation and no normalization, uniform $\alpha = 1/\sqrt{c_\phi}$ is optimal. The signal propagation gain is exactly 1 at every layer.

### 8.2 When Per-Layer Scaling Helps

Per-layer $\alpha_\ell$ is needed when:

1. **Mixed activations.** If layer $\ell$ uses activation $\phi_\ell$, set $\alpha_\ell = 1/\sqrt{c_{\phi_\ell}}$.

2. **Rectangular weight matrices with varying aspect ratios.** For $W_\ell \in \mathbb{R}^{m_\ell \times n_\ell}$ with $m_\ell \neq n_\ell$, the signal propagation through $W_\ell$ depends on the aspect ratio. Specifically, for a semi-orthogonal $W_\ell = \alpha_\ell U_\ell V_\ell^\top$:
   - If $m_\ell > n_\ell$: $W_\ell^\top W_\ell = \alpha_\ell^2 I_{n_\ell}$, so the output variance is $\alpha_\ell^2 \cdot \text{Var(input)}$.
   - If $m_\ell < n_\ell$: $W_\ell W_\ell^\top = \alpha_\ell^2 I_{m_\ell}$, same variance gain.
   
   The per-neuron gain is $\alpha_\ell^2 \cdot \min(m_\ell, n_\ell) / m_\ell$ (each output neuron receives contributions from $\min(m_\ell, n_\ell)$ singular directions spread across $m_\ell$ output dimensions).
   
   For the common case $m_\ell = n_\ell$ (square), this simplifies to $\alpha_\ell^2$, recovering the uniform formula.

3. **First and last layers.** The first layer ($W_1 \in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{input}}}$) and last layer ($W_L \in \mathbb{R}^{d_{\text{output}} \times d_{\text{hidden}}}$) often have very different dimensions. The MLP has:
   - $W_1 \in \mathbb{R}^{256 \times 3072}$: wide input projection
   - $W_2, \dots, W_7 \in \mathbb{R}^{256 \times 256}$: square hidden layers
   - $W_8 \in \mathbb{R}^{10 \times 256}$: narrow output projection
   
   For the input layer: $\min(256, 3072) = 256$, so the effective output per neuron sees 256 non-zero singular values, and the gain is $\alpha^2 \cdot 256/256 = \alpha^2$. Same as square.
   
   For the output layer: it's the final projection; its initialization matters less for intermediate signal propagation.

### 8.3 Geometric Decay/Growth

If a single $\alpha$ doesn't work (unlikely for the MLP), try:
$$\alpha_\ell = \alpha_0 \cdot \gamma^{\ell-1}$$

For variance preservation with ReLU:
$$\frac{(\alpha_0 \gamma^{\ell-1})^2}{2} = 1 \quad \forall \ell$$

This requires $\gamma = 1$, reducing to the uniform case. Geometric scaling only makes sense when you intentionally want signal growth in early layers and decay in later layers (or vice versa), which can arise from architectural asymmetries but not from the activation function alone.

**Recommendation:** Start with uniform $\alpha$. Per-layer scaling is only needed if experiments reveal layer-dependent sensitivity.

---

## 9. Comparison with Existing Initialization Schemes

### 9.1 Why Scaled Orthogonal Dominates Under Muon

| Property | Kaiming Normal | Xavier Normal | Orthogonal ($\alpha=1$) | Scaled Orth ($\alpha^*$) |
|---|---|---|---|---|
| Polar error at init | High (MP spread) | High (MP spread) | 0 | **0** |
| Condition number | $O(\sqrt{n})$ | $O(\sqrt{n})$ | 1 | **1** |
| Signal preservation | Yes (in expectation) | No (wrong scale for ReLU) | No ($\alpha=1$ too small for ReLU) | **Yes (exact)** |
| Variance of per-layer gain | $O(1/n)$ | $O(1/n)$ | 0 | **0** |
| NS iterations needed at step 0 | 5 | 5 | 0 | **0** |
| Phase 1 duration | >0 steps | >0 steps | 0 steps | **0 steps** |

### 9.2 The Marchenko-Pastur Problem

Kaiming and Xavier initialization draw i.i.d. entries, producing singular value spectra that follow the Marchenko-Pastur (MP) distribution. For an $m \times n$ matrix with i.i.d. entries of variance $\sigma^2/n$:

$$\rho(\sigma) = \frac{\sqrt{(\sigma_+^2 - \sigma^2)(\sigma^2 - \sigma_-^2)}}{2\pi \sigma^2 \cdot (m/n)} , \quad \sigma_\pm = 1 \pm \sqrt{n/m}$$

This spread means:
1. **Muon's NS iteration must do real work.** The NS iteration maps singular values $\sigma_i \mapsto 1$, but convergence is slower when the input has high condition number. With 5 NS steps and MP-distributed singular values, the approximation error is nonzero.
2. **Signal propagation is stochastic.** The per-layer gain is $\alpha^2 c_\phi$ only in expectation; individual realizations fluctuate by $O(1/\sqrt{n})$.
3. **Condition number compounds with depth.** For a depth-$L$ network, the end-to-end Jacobian condition number is roughly $\prod_\ell \kappa(W_\ell) = O(n^{L/2})$.

Scaled orthogonal eliminates all three problems by construction.

---

## 10. Warmup Sensitivity Analysis

### 10.1 Why Warmup Exists

Learning rate warmup is a hack to compensate for initialization mismatch. At initialization:
1. Gradients may be poorly conditioned (high condition number).
2. The loss landscape may have sharp curvature in some directions.
3. Muon's NS iteration may not converge well for ill-conditioned momentum.

Warmup gradually increases the learning rate, giving the network time to "settle" before taking large steps. This is necessary when initialization creates a poor starting point.

### 10.2 Why Scaled Orthogonal Should Eliminate Warmup

With scaled orthogonal initialization at $\alpha^*$:
1. The gradient condition number is minimal (all singular values of $W$ are equal).
2. The loss landscape is smooth along the Stiefel manifold (which is where Muon moves).
3. Muon's NS iteration is exact at step 0.

**Prediction:** The warmup sensitivity curve (final loss vs. warmup steps) should be **flat** for scaled orthogonal init. Specifically:
- $\text{FinalLoss}(\text{warmup}=0) \approx \text{FinalLoss}(\text{warmup}=200) \approx \text{FinalLoss}(\text{warmup}=1000)$

For Kaiming init, we expect:
- $\text{FinalLoss}(\text{warmup}=0) \gg \text{FinalLoss}(\text{warmup}=200)$

If this prediction holds, it validates the core thesis: **initialization mismatch is the root cause of warmup sensitivity under Muon, and scaled orthogonal initialization eliminates it.**

### 10.3 What "Flat" Means Quantitatively

Define the **warmup sensitivity score** as the coefficient of variation of final loss across warmup schedules:
$$S = \frac{\text{std}(\text{FinalLoss}(w) : w \in \{0, 100, 200, 500, 1000\})}{\text{mean}(\text{FinalLoss}(w) : w \in \{0, 100, 200, 500, 1000\})}$$

- $S \approx 0$: warmup-insensitive (desired).
- $S > 0.1$: significant warmup sensitivity (Kaiming expected range).

---

## 11. Predictions Summary

### Experiment 1: MLP Sweep

| $\alpha$ | Predicted warmup=0 | Predicted best warmup | Warmup sensitivity $S$ |
|---|---|---|---|
| 0.5 | Diverges or very high loss | High | Very high |
| 0.75 | High loss (signal decay) | 500+ | High |
| 1.0 | Moderate loss | 200 | Moderate |
| $\sqrt{2}$ | **Near-optimal** | 0 | **Near-zero** |
| 1.5 | Good (slight signal growth) | 0-100 | Low |
| 2.0 | Unstable (signal explosion) | 500+ | High |
| 2.5 | Diverges | N/A | N/A |

**Predicted ranking at warmup=0:**
$$\alpha = \sqrt{2} > \alpha = 1.5 > \alpha = 1.0 > \alpha = 0.75 \gg \alpha = 2.0 \gg \alpha = 0.5 \approx \alpha = 2.5$$

**Predicted ranking at best warmup:**
$$\alpha = \sqrt{2} \geq \alpha = 1.5 \geq \alpha = 1.0 \approx \text{Kaiming}$$

### Experiment 2: Normalized Architectures

| Architecture | Predicted best $\alpha$ | Predicted warmup sensitivity |
|---|---|---|
| ResNet (BatchNorm) | Any $\alpha \in [0.75, 2.0]$ works | Low for all $\alpha$ (BN absorbs) |
| ViT (LayerNorm) | Any $\alpha \in [0.75, 2.0]$ works | Low for all $\alpha$ (LN absorbs) |
| NanoGPT (LayerNorm) | Any $\alpha \in [0.75, 2.0]$ works | Low for all $\alpha$ (LN absorbs) |

**The valuable finding:** If normalization layers negate $\alpha$'s effect on signal propagation but the polar error = 0 advantage persists, we should see a small but consistent improvement from scaled orthogonal (any $\alpha$) over Kaiming/Xavier even in normalized architectures. This improvement would be due purely to the NS convergence advantage.

---

## 12. Summary of Key Results

1. **The optimal scale factor is $\alpha^* = 1/\sqrt{c_\phi}$**, determined solely by the activation function's second-moment gain. For ReLU, $\alpha^* = \sqrt{2}$. For GELU, $\alpha^* \approx 1.534$.

2. **Polar error is zero for any $\alpha$**. The scale factor does not affect Muon's ability to orthogonalize — it affects signal propagation only.

3. **The spectral norm ball constraint $\alpha \leq 1/\lambda$ is almost never binding** in practice because Muon weight decay is typically small or zero.

4. **Normalization layers absorb $\alpha$'s signal propagation effect** but not the polar error advantage. In normalized architectures, $\alpha = 1$ suffices.

5. **Scaled orthogonal should eliminate warmup sensitivity** at $\alpha^*$ because it simultaneously achieves zero polar error, perfect signal propagation, and unit condition number.

6. **The advantage over standard orthogonal init ($\alpha = 1$) in ReLU networks is exactly the Kaiming correction factor $\sqrt{2}$** — but applied to a spectrally uniform matrix rather than an i.i.d. random matrix.

---

## Appendix A: Computing $c_\phi$ Numerically

For any activation $\phi$:

```python
import torch

def compute_activation_gain(phi, num_samples=1_000_000):
    """Compute c_phi = E[phi(z)^2] / E[z^2] for z ~ N(0,1)."""
    z = torch.randn(num_samples)
    return (phi(z) ** 2).mean().item()

# Examples:
c_relu = compute_activation_gain(torch.relu)           # ~0.5
c_gelu = compute_activation_gain(torch.nn.functional.gelu)  # ~0.4252
c_silu = compute_activation_gain(torch.nn.functional.silu)   # ~0.3024
```

## Appendix B: Haar-Distributed Orthogonal Matrices via QR

PyTorch's `nn.init.orthogonal_` uses QR decomposition of a random Gaussian matrix, which produces Haar-distributed orthogonal matrices (up to a sign correction that doesn't matter for our purposes since we only care about singular values, which are all $\alpha$).

```python
# nn.init.orthogonal_(W) is equivalent to:
# A = torch.randn_like(W)
# Q, R = torch.linalg.qr(A)
# W.copy_(Q[:, :n])  (for m >= n)
```

This is the correct sampling procedure. No additional steps needed.

## Appendix C: Newton-Schulz Convergence Guarantee

The NS iteration used in Muon is the quintic variant:
$$X_{k+1} = a X_k + (b X_k X_k^\top + c X_k X_k^\top X_k X_k^\top) X_k$$

with $(a, b, c) = (3.4445, -4.7750, 2.0315)$.

For convergence, the input must be normalized so that $\|X_0\|_{\text{op}} \leq 1$. The code does:
```python
X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
```

This Frobenius normalization ensures $\|X\|_F = 1$, which implies $\|X\|_{\text{op}} \leq 1$. For a scaled orthogonal matrix $\alpha Q$, after Frobenius normalization:
$$X_0 = \frac{Q}{\|Q\|_F} = \frac{Q}{\sqrt{\min(m,n)}}$$

All singular values of $X_0$ are $1/\sqrt{\min(m,n)}$, which is well within the convergence basin. The NS iteration then maps all singular values toward 1. After 5 iterations on such a well-conditioned input, the approximation error is negligible ($< 10^{-6}$).

For comparison, under Kaiming init, the Frobenius-normalized singular values span a wide range (Marchenko-Pastur), and the NS iteration must work harder to map them all to 1. The error after 5 iterations is larger, especially for the extreme singular values.
