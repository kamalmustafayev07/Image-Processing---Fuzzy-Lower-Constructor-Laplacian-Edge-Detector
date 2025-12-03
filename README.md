# Image-Processing---Fuzzy-Lower-Constructor-Laplacian-Edge-Detector

# Fuzzy Lower-Constructor + Laplacian Edge Detector  
**Highly noise-robust second-order edge detection via fuzzy ordered weighted aggregation**

A research-grade implementation of a novel edge detection pipeline that dramatically outperforms the classical discrete Laplacian (and Laplacian-of-Gaussian) on noisy real-world images.

The key contribution: **replace raw pixel values with a fuzzy lower-constructor estimate before applying the Laplacian**. This single preprocessing step suppresses impulse and Gaussian noise extremely effectively while perfectly preserving step edges — making reliable zero-crossing detection possible again.

## Repository Contents

| Notebook | Purpose |
|----------|--------|
| `lc_laplacian_operator_simplified.ipynb` | Core pipeline (fixed decreasing weights), quantitative evaluation (Precision/Recall/F1), direct head-to-head comparison with pure Laplacian |
| `lc_laplacian_operator_with_tnorms.ipynb` | Fully generalized version supporting arbitrary **t-norms** and **t-conorms**, exhaustive visual results on multiple noisy datasets |
| `morphological_operators.ipynb` | Baseline comparison against classical morphological gradients (external, internal, Beucher, opening/closing, combined) |

## Method Summary

For each pixel and a local `(2k+1)×(2k+1)` window (typically 7×7 or 9×9):

1. Sort the `N = (2k+1)²` intensity values:  
   `x₁ ≤ x₂ ≤ … ≤ x_N`

2. Apply **fuzzy lower constructor** (OWA with monotonically decreasing weights `wₖ`):  
   `L(p) = Σ_{k=1}^{N} wₖ · xₖ`  
   or, in the generalized version:  
   `L_T(p) = T( w₁ ⊙ x₁, w₂ ⊙ x₂, … )`

3. Compute the standard discrete Laplacian on the denoised image `L(image)`

4. Detect zero-crossings → binary edge map (optional hysteresis thresholding)

Result: **clean, high-precision edges** even under severe noise where pure Laplacian completely collapses.

## Quantitative Example (`planes.jpg`)

| Method                         | Precision | Recall | F1-score |
|--------------------------------|-----------|--------|----------|
| Lower Constructor + Laplacian  | **0.934** | 0.623  | **0.748** |
| Pure Laplacian                 | 0.765     | 0.020  | 0.038    |

Morphological gradients also fail on the same data (see `morphological_operators.ipynb`).

## Supported T-norms / T-conorms (`lc_laplacian_operator_with_tnorms.ipynb`)

| Family       | T-norm                     | T-conorm                    | Parameter |
|--------------|----------------------------|-----------------------------|-----------|
| Minimum      | `min(a,b)`                 | `max(a,b)`                  | –         |
| Product      | `a*b`                      | `a+b–a*b`                   | –         |
| Łukasiewicz  | `max(a+b–1,0)`             | `min(a+b,1)`                | –         |
| Drastic      | drastic product            | drastic sum                 | –         |
| Hamacher     | parameterized              | parameterized               | γ ≥ 0     |
| Frank        | parameterized              | parameterized               | s > 0     |
| Yager        | parameterized              | parameterized               | p > 0     |

Empirically best combination on noisy images:  
**Lower constructor** → `T = minimum`  
**Upper constructor (when used)** → `S = Łukasiewicz`

## Dependencies

```bash
numpy
opencv-python
matplotlib
scikit-learn
scipy
jupyter
```

## Quick Usage

```python
import cv2
from utils.fuzzy_laplacian import fuzzy_edge_detection_pipeline_optimized

img = cv2.imread('images/noisy_image.jpg', cv2.IMREAD_GRAYSCALE)

edges = fuzzy_edge_detection_pipeline_optimized(
    image=img,
    window_size=9,           # 7–9 recommended
    T1="min",                # lower constructor t-norm
    T2="luk",                # upper constructor t-conorm (optional)
    zero_crossing_threshold=25
)

cv2.imwrite('edges.png', edges)
```

## Performance

- Fully vectorized using `scipy.ndimage.generic_filter`
- ~0.3–0.9 s per megapixel (window 9×9) on a modern CPU
- Memory-efficient (processes one neighborhood at a time)

**When classical second-order detectors fail due to noise — the fuzzy lower constructor makes them work again.**
