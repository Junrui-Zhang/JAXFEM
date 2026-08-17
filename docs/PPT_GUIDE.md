# PPT Conversion Guide — JAX EBE-PCG Project

> How to convert [TECH_REPORT.md](TECH_REPORT.md) into a 12-slide presentation.
> Each slide specifies: **title**, **text content**, **visual element**, and **speaker notes**.

---

## Slide 1: Title Slide

**Title**: GPU-Parallel Finite Element Solver Based on JAX EBE-PCG

**Subtitle**: Feasibility Verification on Simply-Supported BEAM4 Beam

**Text**: Your Name · Supervisor · Date · Institution Logo

**Visual**: Export a 3D rendering of the deformed beam via `post.py --save screenshot.png` (or use the HTML export opened in browser, screenshot it).

**Speaker note**: "This project investigates GPU acceleration for finite element analysis using JAX's vmap vectorization. We implement an Element-by-Element PCG solver and verify it on a simply-supported beam."

---

## Slide 2: Problem & Motivation

**Title**: Why GPU-Parallel FEM?

**Text**:
- Bridge buffeting analysis requires solving **K u = F** thousands of times
- Direct solver: O(n³) — prohibitive for fine meshes
- Serial CG: O(n²) per solve — too slow for real-time
- **Goal**: GPU-parallel iterative solver with O(n) per-iteration cost

**Visual**: [figures/architecture.pdf](figures/architecture.pdf)

**Speaker note**: "Long-span bridges under wind loads need time-domain analysis with thousands of time steps. At each step, a large linear system must be solved. Traditional methods can't keep up."

---

## Slide 3: What Is EBE-PCG?

**Title**: Element-by-Element Formulation

**Text**:
- Classical CG needs global stiffness matrix K → O(n²) memory
- EBE: Decompose ALL CG operations to element level
- Key concept: **Real vector v^e** (self only) vs. **Fake vector v^(e)** (with neighbor contributions)
- Inner product: (r,r) = Σ_e (r^e)^T r^(e)
- No global matrix assembly needed

**Visual**: [figures/ebe_principle.pdf](figures/ebe_principle.pdf)

**Speaker note**: "The breakthrough of EBE is that every CG operation — inner products, matrix-vector products — can be decomposed as sums over elements. This means we never need to build the global stiffness matrix. Each element only knows itself and its immediate neighbors."

---

## Slide 4: BEAM4 Element

**Title**: BEAM4 3D Beam Element

**Text**:
- 2 nodes × 6 DOF/node = **12×12 stiffness matrix**
- DOF: UX, UY, UZ, ROTX, ROTY, ROTZ
- Stiffness components: axial + torsion + XY bending + XZ bending
- Section properties computed via **sectionproperties** (warping analysis)

**Visual**: [figures/beam4_element.pdf](figures/beam4_element.pdf)

**Speaker note**: "We use BEAM4 — the ANSYS 3D beam element. Each node has 6 degrees of freedom. The 12×12 stiffness matrix combines axial, torsional, and bending stiffness. Cross-section properties are computed using the sectionproperties library with full warping analysis."

---

## Slide 5: JAX Implementation

**Title**: From Serial Loops to GPU vmap

**Text (left column)**:
```
# NumPy: Serial loop
for e in range(n_elem):
    Kp[e] = K_e[e] @ p_fake[e]
```
→ O(n_elem) × Python overhead

**Text (right column)**:
```
# JAX: vmap parallel
jax.vmap(lambda K, p: K @ p)(K_e, p_fake)
```
→ All elements in one GPU kernel

**Key optimization**: Replaced dense adjacency matrix `(n×n) @ (n×6)` with O(n) index gather: `v_pad[neighbor_idx, :]`

**Visual**: [figures/benchmark_per_iter.pdf](figures/benchmark_per_iter.pdf)

**Speaker note**: "The key JAX feature is vmap — it automatically vectorizes a function over a batch dimension. We apply it to element stiffness operations, so all elements compute simultaneously on GPU. We also replaced the O(n²) dense adjacency multiplication with O(n) index gathering."

---

## Slide 6: Challenge 1 — Accuracy Collapse

**Title**: Why Large Models Fail: The Convergence Trap

**Text**:
- When n_elem = 200, max_iter=500 → **96.7% error**!
- CG converges "suddenly" after prolonged stagnation
- Low-frequency bending mode needs many Krylov iterations to capture
- **Iteration count scales as ~O(n^1.7) with diagonal preconditioner**

**Visual**: [figures/convergence.pdf](figures/convergence.pdf) + Table:

| n_elem | Required Iter | max_iter=500 |
|--------|:------------:|:------------:|
| 10 | 11 | ✅ |
| 100 | 301 | ✅ |
| 200 | 1119 | ❌ 96.7% err |
| 500 | 5765 | ❌ |

**Speaker note**: "The most confusing bug: small models work perfectly, but large models give nonsense. The root cause is that CG needs many iterations to capture the global bending mode. For 200 elements, iterations 1-500 show almost zero progress. The solution abruptly appears around iteration 600. The fix: adaptive max_iter based on problem size."

---

## Slide 7: Challenge 2 — O(n²) Bottleneck

**Title**: Hidden O(n²) in the Adjacency Operation

**Text**:
- Original `real_to_fake` used `(n×n) @ (n×6)` dense matrix multiply
- For beam chain: each element has ≤2 neighbors → 99.9% sparsity
- **Fix**: Replace dense matrix with index array + gather
- Result: per-iteration time constant at **~0.13ms regardless of n_elem**

**Visual**: [figures/benchmark_per_iter.pdf](figures/benchmark_per_iter.pdf)

| n_elem | Before | After | Speedup |
|--------|-------:|------:|--------:|
| 200 | 0.73ms | 0.34ms | 2.1× |
| 500 | 0.89ms | 0.13ms | **6.8×** |

**Speaker note**: "The dense adjacency operation was a silent performance killer. For a 500-element beam chain, we were computing 250,000 multiplications when only 500 were needed. Switching to index gathering eliminated this entirely."

---

## Slide 8: Challenge 3 — Cross-Section Accuracy

**Title**: sectionproperties Torsion Constant Accuracy

**Text**:
- Default mesh was too coarse for warping analysis
- J error: 6.7% vs. Saint-Venant exact solution
- **Fix**: Mesh size from `min(b,h)/20` → `min(b,h)/100`
- J error reduced to **0.3%**

| Source | J (m⁴) | Error |
|--------|--------|:-----:|
| Saint-Venant | 4.698e-04 | — |
| ANSYS | 4.7e-04 | ~0% |
| sp (mesh=0.01) | 5.014e-04 | 6.7% |
| sp (mesh=0.002) | 4.712e-04 | 0.3% |

**Speaker note**: "For geometric properties like area and moment of inertia, the mesh is irrelevant — they're exact from geometry. But the torsion constant J depends on the warping function, which needs a fine mesh to resolve accurately."

---

## Slide 9: Challenge 4 — Visualization

**Title**: Making 3D Visualization PPT-Ready

**Text**:
- **Ghost mesh only on right subplot** → moved `show_shade` into each subplot loop
- **Beam section proportions wrong** → camera set to side view (Y-axis) to show Z-height
- **Colorbar precision lost in HTML** → data scaled to mm/MPa, explicit `clim=[min,max]`
- Output: self-contained **interactive HTML** (vtk.js) insertable into PPT via Web Viewer

**Visual**: Use `post.py --save-html beam.html` output, screenshot for slide

**Speaker note**: "Interactive 3D visualizations help communicate results. The HTML export preserves rotation and zoom. For PPT, either embed via Office Web Viewer, or take a high-resolution screenshot with `window_size=(2400,1800)`."

---

## Slide 10: Benchmark Results

**Title**: Performance: JAX vs. NumPy vs. Direct

**Text**:
- **Small models (≤100 elem)**: GPU overhead dominates, JAX slower
- **Cross-over at ~200 elem**: JAX overtakes NumPy EBE
- **500 elem**: JAX is **23× faster** than NumPy EBE
- **Limiting factor**: Iteration count, not per-iteration speed

**Visual**: [figures/benchmark_time_cost.pdf](figures/benchmark_time_cost.pdf) + [figures/scaling.pdf](figures/scaling.pdf)

| n_elem | Direct | NP EBE | JAX EBE | JAX/NP |
|--------|-------:|-------:|--------:|-------:|
| 100 | 10ms | 190ms | 316ms | 0.6× |
| 200 | 39ms | 1198ms | 347ms | 3.5× |
| 500 | 369ms | 17026ms | 740ms | **23×** |

**Speaker note**: "The benchmark shows a clear story: GPU wins at scale. For small problems, GPU launch overhead makes JAX slower. But at 500 elements, the 23× speedup is dramatic. And the per-iteration cost is constant — the remaining growth is from the iteration count, which we can address with better preconditioners."

---

## Slide 11: Future Work

**Title**: Beyond Diagonal Preconditioning

**Text**:
- **Current bottleneck**: Iteration count grows as O(n^1.7)
- **Root cause**: Diagonal preconditioner cannot handle condition number κ ~ O(n⁴)
- **Solution**: Multigrid preconditioner (MGCG) → ~O(1) iterations
  - Coarse grid correction: P @ K_c⁻¹ @ P^T
  - Expected: 6624 → ~30 iterations for n_elem=500
- **Extensions**: Shell elements (SHELL181), nonlinear analysis, dynamic time-stepping

**Visual**: Simple diagram of two-level multigrid V-cycle:

```
Fine grid   — smoothing —\    /— correction — smoothing
                          V-cycle
Coarse grid ———— K_c^{-1} ————
```

**Speaker note**: "The ultimate solution to the iteration count problem is multigrid. By solving a coarse version of the problem and interpolating the correction back, we can reduce iterations from thousands to tens — independent of problem size. This is our next milestone."

---

## Slide 12: Summary & Key Takeaways

**Title**: Summary

**Text**:
1. ✅ **EBE-PCG on GPU** is feasible and accurate (verified against analytical solution)
2. ✅ **vmap parallelization** achieves constant per-iteration cost (~0.13ms) regardless of mesh size
3. ✅ **23× speedup** over NumPy at 500 elements
4. ⚠ **Diagonal PC limits convergence** — iteration count grows as O(n^1.7)
5. 🔮 **Next step**: Multigrid PCG for O(1) iterations

**Key numbers**:
- BEAM4: 12×12 stiffness, 6 DOF/node
- J: 4.712×10⁻⁴ m⁴ (sectionproperties, 0.3% accuracy)
- Deflection: -2.2046 mm = PL³/(48EI)
- CG at 500 elem: 5765 iterations, 740 ms

**Visual**: Export `post.py --save-html beam.html` for the deformed beam, screenshot for slide background.

**Speaker note**: "To summarize: we have a working GPU-parallel FE solver. It's correct, it's significantly faster than serial at scale, and the architecture is clean. The main limitation — iteration count growth — has a clear path to resolution through multigrid methods."

---

## Appendix: Image Generation Commands

All figures can be regenerated:

```bash
# Regenerate all report figures
python generate_figures.py

# Generate 3D beam screenshot for slides
python post.py --save slide_beam.png --window-size 2400 1800
python post.py --mises --save slide_mises.png --window-size 2400 1800

# Generate interactive HTML (open in browser, screenshot for PPT)
python post.py --save-html interactive_beam.html
python post.py --mises --save-html interactive_mises.html
```

---

## Appendix: PPT Design Tips

1. **Consistent color palette**: Use the npg colors from plots (`#E64B35` red, `#4DBBD5` blue, `#00A087` green)
2. **Font**: Use sans-serif (Arial/Calibri) for titles, serif for equations
3. **Equation rendering**: Use LaTeX in PowerPoint (`Insert → Equation`) for formulas
4. **Slide aspect ratio**: 16:9 widescreen
5. **Minimal text**: Each slide ≤ 5 bullet points, ≤ 2 lines each
6. **Image quality**: Export PDF figures at dpi=200+, or use `--window-size 2400 1800` for PNG screenshots
7. **Interactive demos**: For defense, open the HTML exports in browser alongside the slides
8. **Backup slides**: Add 2-3 appendix slides with detailed formulas and data tables
