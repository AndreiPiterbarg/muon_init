# Theory Task: Optimal Scaled Orthogonal Initialization for Muon

## Core Idea

A semi-orthogonal matrix is $W = \alpha \cdot UV^\top$ where $U, V$ are random orthogonal (Haar-distributed) and $\alpha > 0$ is a scalar scale factor. All singular values equal $\alpha$. This is the simplest possible initialization that has zero polar error — Muon's NS iteration has nothing to do on step 0 because the matrix is already a scaled isometry.

The only free parameter is $\alpha$. Find the optimal $\alpha$ as a function of layer dimensions $(m, n)$, depth $L$, weight decay $\lambda$, and activation function.

## Why This Is the Right Starting Point

1. **Polar error = 0 by construction.** $\text{polar}(\alpha UV^\top) = UV^\top$ for any $\alpha > 0$. The NS iteration converges in 0 steps.

2. **Spectral norm ball is trivially satisfiable.** Just set $\alpha \leq 1/\lambda$. Phase 1 duration = 0.

3. **Signal propagation reduces to a scalar equation.** For a depth-$L$ linear network with semi-orthogonal weights at scale $\alpha$, the end-to-end Jacobian has all singular values equal to $\alpha^L$. Dynamical isometry ($\alpha^L \approx 1$) gives $\alpha = 1$ for any depth. With ReLU (kills half the variance), the effective per-layer gain is $\alpha \cdot \sqrt{1/2}$, so $\alpha = \sqrt{2}$ recovers Kaiming-scale signal propagation.

4. **One scalar to sweep.** No distribution shape, no complicated spectral engineering. Just $\alpha$.

## What to Do

### Experiment 1: Sweep $\alpha$ on the MLP (no normalization, highest init sensitivity)

Using `mlp_cifar10.yaml` (8-layer MLP, no BatchNorm, no skip connections):

- Init: $W_\ell = \alpha \cdot U_\ell V_\ell^\top$ for each layer, with $U_\ell, V_\ell$ drawn from Haar measure
- Sweep $\alpha \in \{0.5, 0.75, 1.0, \sqrt{2}, 1.5, 2.0, 2.5\}$
- For each $\alpha$, sweep warmup $\in \{0, 100, 200, 500, 1000\}$
- 5 seeds each
- Compare against Kaiming/Xavier/Orthogonal baselines (already have warmup sweep infrastructure)

**Primary question**: Is there an $\alpha$ where warmup = 0 matches or beats Kaiming with warmup = 200?

### Experiment 2: Validate on architectures with normalization

Run the winner from Experiment 1 on `resnet_cifar10.yaml` (BatchNorm), `vit_cifar10.yaml` (LayerNorm), and `nanogpt_owt.yaml` (LayerNorm). Check whether normalization layers make $\alpha$ irrelevant (likely partially, which is itself a useful finding).

### Experiment 3: Per-layer $\alpha_\ell$ (only if uniform $\alpha$ fails)

If no single $\alpha$ works across all layers, try:
- $\alpha_\ell = \alpha_0 \cdot \gamma^\ell$ (geometric decay/growth with depth)
- Sweep $(\alpha_0, \gamma)$ on a small grid

This adds one parameter but captures the intuition that deeper layers may need different scaling.

## Metrics to Track

All of these are already implemented:

| What to measure | Function | Expected result for scaled orthogonal |
|---|---|---|
| Polar error at init | `compute_polar_error()` | 0 for all $\alpha$ |
| Spectral norm ball | `check_spectral_norm_ball()` | Inside iff $\alpha \leq 1/\lambda$ |
| Phase 1 duration | `Phase1Tracker` | 0 if $\alpha \leq 1/\lambda$ |
| Jacobian singular values | `compute_jacobian_spectrum()` | All equal to $\alpha^L \cdot c_{\text{act}}^L$ |
| Warmup sensitivity | `WarmupSensitivityAnalyzer` | Should be flat (warmup-insensitive) at optimal $\alpha$ |
| Final loss/accuracy | training loop | Must match or beat baselines |
| First-step gradient condition number | `analyze_first_muon_step()` | Should be lower than Kaiming |

## Predictions

1. $\alpha = 1$ (pure orthogonal) will be close to optimal for networks with normalization layers.
2. $\alpha = \sqrt{2}$ will be close to optimal for ReLU networks without normalization (Kaiming-scale correction).
3. Any $\alpha$ in a reasonable range will beat Kaiming/Xavier at warmup = 0, because polar error = 0 and the spectral structure is uniform.
4. The warmup sensitivity curve (final loss vs. warmup steps) will be nearly flat for the scaled orthogonal init, while Kaiming will show steep degradation as warmup → 0.

## Implementation

One function in `initializations/implementations/`:

```python
def scaled_orthogonal(model, alpha=1.0):
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.orthogonal_(module.weight)
            module.weight.data.mul_(alpha)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
```

That's it. The research question is picking $\alpha$, not writing code.
