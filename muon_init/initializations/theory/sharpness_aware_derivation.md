# Sharpness-Aware Initialization: Derivation of alpha*_EoS

## Objective

Derive the initialization scale alpha such that the top Hessian eigenvalue at
initialization satisfies the Edge of Stability condition:

    eta * lambda_max(alpha) < 2

where eta is the Muon learning rate. This ensures the full learning rate is
immediately stable, potentially eliminating the need for warmup.

## Background

### Edge of Stability (Cohen et al., 2021)

For gradient descent with learning rate eta on a loss L(w):
- If eta * lambda_max > 2, the gradient step overshoots and training is unstable
- Training self-regulates to the "edge of stability" where eta * lambda_max ~ 2

### Warmup as Sharpness Reduction (Kalra & Barkeshli, NeurIPS 2024)

Warmup's primary benefit is keeping eta_effective < 2/lambda_max during early
training when sharpness is high. If initialization controls lambda_max(0), we
can set alpha to make the full learning rate immediately stable.

## Analytical Derivation for Unnormalized MLPs

### Setup

Consider an L-layer MLP with scaled orthogonal initialization:

    W_l = alpha * Q_l,  Q_l ~ Haar(O(n))

and activation function phi with gain c_phi = E[phi(z)^2] for z ~ N(0,1).

### Gauss-Newton Approximation

For cross-entropy loss, the Hessian admits the Gauss-Newton decomposition:

    H = J^T diag(p(1-p)) J + (second-order term)

where J = d(logits)/d(params) is the Jacobian. At initialization with random
weights, the second-order term is small and:

    lambda_max(H) ~ lambda_max(J^T J) = sigma_max(J)^2

### Signal Propagation Through Layers

For W_l = alpha * Q_l with activation phi:

    h_l = phi(W_l h_{l-1}) = phi(alpha * Q_l h_{l-1})

The Jacobian of h_l w.r.t. h_{l-1} has spectral norm:

    sigma_max(dh_l/dh_{l-1}) = alpha * sigma_max(diag(phi'(z_l)) * Q_l)

For ReLU, phi'(z) in {0, 1}, and roughly half the neurons are active:

    sigma_max(dh_l/dh_{l-1}) ~ alpha * sqrt(c_phi_eff)

where c_phi_eff accounts for the activation pattern. The end-to-end Jacobian
spectral norm is:

    sigma_max(J) ~ prod_l alpha * sqrt(c_phi_eff) = (alpha * sqrt(c_phi_eff))^L

Therefore:

    lambda_max ~ C_data * (alpha^2 * c_phi_eff)^L

where C_data depends on the data distribution and loss function.

### Solving for alpha*

Setting lambda_max = 2/eta:

    C_data * (alpha*^2 * c_phi_eff)^L = 2/eta

    alpha* = (2 / (eta * C_data))^(1/(2L)) / sqrt(c_phi_eff)

### Estimating C_data

C_data can be estimated from a single lambda_max measurement at a known alpha_ref:

    C_data = lambda_max(alpha_ref) / (alpha_ref^2 * c_phi_eff)^L

## Empirical Results

### Deep MLP (8 layers, 256 hidden, ReLU)

| Method | alpha*_EoS | Notes |
|--------|-----------|-------|
| Grid search | 1.34 | Largest alpha with eta*lambda_max < 2 |
| Power law fit (R^2=0.975) | 1.395 | lambda_max = 3.39 * alpha^10.17 |
| Analytical formula | 1.298 | Using C_data from alpha=1.0 reference |

The fitted exponent k=10.17 is between L=8 and 2L=16. This discrepancy arises
because the simple (alpha^2 * c_phi)^L model assumes:
- All singular values of each layer's Jacobian are equal (they're not — ReLU
  creates a sparse activation pattern that is random, not uniform)
- The end-to-end Jacobian spectral norm equals the product of per-layer norms
  (it doesn't — singular vectors are not aligned across layers)

The effective per-layer exponent is k/L = 10.17/8 = 1.27, meaning each layer
contributes alpha^1.27 to the spectral norm (vs the predicted alpha^2).

### ViT-Tiny (6 layers, 192 dim, LayerNorm)

| Method | alpha*_EoS | Notes |
|--------|-----------|-------|
| Grid search | ~0.12 | Only alpha=0.1 tested stable |
| Exponential fit (R^2=0.90) | 0.510 | lambda_max = 44.2 * exp(1.6 * alpha) |
| Analytical formula | 0.924 | WRONG — 90% mismatch on exponent |

**The analytical formula fails for ViT because LayerNorm breaks the
multiplicative spectral amplification.** The fitted exponent k=1.22 instead of
the predicted 2L=12. Each LayerNorm layer resets the activation scale, preventing
the exponential growth that drives the MLP's super-exponential lambda_max(alpha).

### ResNet-18 (BatchNorm + skip connections)

The EoS framework does not apply: all top eigenvalues at initialization are
negative (the loss surface is a saddle point). BatchNorm creates negative
curvature directions that dominate. The sharpness-aware initialization approach
is not applicable to BatchNorm architectures.

## Architecture Taxonomy

| Architecture Feature | Effect on lambda_max(alpha) |
|-|-|
| No normalization | Super-exponential: lambda_max ~ alpha^(~1.3L) |
| LayerNorm | Near-linear: lambda_max ~ C * alpha^(~1.2) |
| BatchNorm | Indefinite Hessian: lambda_max < 0 at init |
| Skip connections | Reduces effective depth (residual dampens amplification) |
| Attention | Amplifies curvature (quadratic interaction in QK^T) |

## Practical Formula

### For unnormalized networks (MLP without BN/LN):

    alpha*_EoS = (2 / (eta * C_data))^(1/(2L)) / sqrt(c_phi)

where C_data is measured by computing lambda_max at alpha=1.0.

### For normalized networks (Transformer with LN):

No closed-form formula. Use the empirical procedure:
1. Measure lambda_max at 5-10 alpha values
2. Fit lambda_max(alpha) = a * exp(b * alpha)  or  C * alpha^k
3. Solve for alpha where lambda_max = 2/eta

### For BatchNorm networks:

Not applicable. BatchNorm handles sharpness independently of initialization scale.
