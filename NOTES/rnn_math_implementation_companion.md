# RNN Math Doc — Implementation Companion

Companion to `rnn_math_from_first_principles.md`. The math doc stays clean for standalone reading; this file cross-references each part to the actual C code in `RNN_in_C/`. Read side by side.

---

## File map

```
RNN_in_C/
├── main.c                    training loop, chunk building
├── gradient_check.c          separate binary — finite-difference verification
├── data_ops/
│   ├── data.h                Corpus struct, one_hot
│   └── data.c
└── rnn/
    ├── rnn.h                 RNN, RNNGradients, RNNCache; function declarations
    └── rnn.c                 rnn_step, rnn_forward, rnn_backward,
                              gradients_clip, rnn_update, rnn_sample,
                              plus softmax and static helpers

lib/matrix/                   shared matrix library (used by both binaries)
```

---

## Notation: math doc ↔ our code

The math doc uses generic `D` (input dim), `H` (hidden dim), `C` (output classes). For a character-level LM, `D = C = V` (vocabulary size).

| Math doc                | Our code                            | Meaning                                          |
|-------------------------|-------------------------------------|--------------------------------------------------|
| `D`                     | `rnn->V`                            | input dim = vocab size                           |
| `H`                     | `rnn->H`                            | hidden dim (32 in our config)                    |
| `C`                     | `rnn->V`                            | output classes = vocab size                      |
| `T`                     | `T` (macro in rnn.h, = 25)          | timesteps per BPTT chunk                         |
| `x_t`                   | `cache->x_cache[t]`                 | one-hot input at time t                          |
| `h_t` (math)            | `cache->h_cache[t]`                 | hidden state (0-indexed to T)                    |
| `a_t`                   | (not cached, computed inline)       | preactivation                                    |
| `h_t = tanh(a_t)`       | `cache->h_cache[t+1]`               | new hidden state (see indexing note below)       |
| `z_t`                   | (not cached, computed inline)       | output logits                                    |
| `ŷ_t` / `p_t`           | `cache->p_cache[t]`                 | softmax output                                   |
| `y_t`                   | `one_hot(chunk[t+1], V)`            | target one-hot, built on the fly in backward     |
| `L_t`                   | `-log(p_cache[t]->values[target][0])` | per-step cross-entropy loss                    |
| `W_xh, W_hh, b_h, W_hy, b_y` | `rnn->Wxh, ->Whh, ->bh, ->Why, ->by` | parameter matrices/vectors                |
| `dW_xh, dW_hh, db_h, dW_hy, db_y` | `grads->dWxh, ->dWhh, ->dbh, ->dWhy, ->dby` | gradient accumulators           |
| `δ_{z,t}`               | `dzt` (local in `rnn_backward`)     | ∂L/∂z_t                                          |
| `dh_t`                  | `dh_t` (local)                      | ∂L/∂h_t (full gradient into hidden state)        |
| `δ_t` / `da_t`          | `da_t` (local)                      | ∂L/∂a_t (gradient at preactivation)              |
| `W_hh^T · δ_{t+1}`      | `dh_next` (loop-carried)            | future gradient, already premultiplied by W_hh^T |

### Indexing convention

Math doc section 104 uses:

```
h[0] = 0
h[t+1] = tanh(W_xh x[t] + W_hh h[t] + b_h)
```

Our code follows this exactly. So `cache->h_cache` has T+1 slots (0 to T inclusive). The parameter matrices read at the natural indices for that convention.

The consequence for backward — the single most common BPTT bug source:

- **Gradient at step t w.r.t. `W_hh`**:
  - Math: `dW_hh += δ_t · h_{t-1}^T`
  - Code: `dWhh += da_t · h_cache[t]^T` (transpose of *previous* h)
- **Gradient at step t w.r.t. `W_hy`**:
  - Math: `dW_hy += δ_{z,t} · h_t^T`
  - Code: `dWhy += dzt · h_cache[t+1]^T` (transpose of *current* new h)

Our code uses `h_prev_T` and `h_next_T` as variable names to make the distinction unmistakable.

---

## Cross-reference by section

### Section 0 — The map

**0.1 What we ultimately need** — the five gradient boxes correspond exactly to the fields of `RNNGradients` in `rnn.h`.

**0.2 The forward dependency** — one iteration of the equation chain is `rnn_step`. The chain over `t = 0..T-1` is `rnn_forward`.

**0.3 Why the RNN dependency is different** — `h_t` has two downstream paths. In code, the two contributions to `dh_t` are the two summands of one `add`:

```c
Matrix* dh_t = add(WhyT_dot_dzt, dh_next);   // current output + future recurrent
```

**0.4 Backward dependency** — the `δ_{T+1} = 0` boundary condition is set in code as:

```c
Matrix* dh_next = matrix_create(rnn->H, 1);
matrix_init(dh_next, 0.0);
// then enter the loop
```

**0.5 Full dependency map** — the ASCII diagram in the math doc corresponds to the code flow in `rnn_step` (forward branch) and the six steps inside `rnn_backward`'s T-loop (backward branch).

**0.6 Prerequisite table** — each row maps to specific accumulations in `rnn_backward`:

| Math need         | Prerequisite            | Code                                                           |
|-------------------|-------------------------|----------------------------------------------------------------|
| `∂L/∂W_hy`        | `δz_t` and `h[t+1]`     | `dot(dzt, transpose(h_cache[t+1]))` → `matrix_add_inplace`     |
| `∂L/∂b_y`         | `δz_t`                  | `matrix_add_inplace(grads->dby, dzt)`                          |
| `∂L/∂h_t`         | current + future paths  | `add(dot(WhyT, dzt), dh_next)`                                 |
| `∂L/∂a_t`         | `dh_t` and tanh'        | `hadamard(dh_t, apply(one_minus_hsq, h_cache[t+1]))`           |
| `∂L/∂W_xh`        | `da_t` and `x[t]`       | `dot(da_t, transpose(x_cache[t]))` → `matrix_add_inplace`      |
| `∂L/∂W_hh`        | `da_t` and `h[t]`       | `dot(da_t, transpose(h_cache[t]))` → `matrix_add_inplace`      |
| `∂L/∂b_h`         | `da_t`                  | `matrix_add_inplace(grads->dbh, da_t)`                         |

---

### Sections 1-2 — Notation

Our tiny model:
- H = 32 (hidden units)
- V = 51 (unique characters in our English corpus)
- T = 25 (chunk length)

Setup functions in `rnn.c`:
- `rnn_create(H, V)` — allocates the 5 parameter matrices with correct shapes (per section 2), randomizes weights (small values), zeros biases.
- `gradients_create(H, V)` — allocates 5 gradient matrices matching parameter shapes, zeros them.
- `cache_create()` — allocates struct, sets all slots to NULL.

---

### Section 104 — Minimal conceptual C model

The section our code most directly implements. Line-for-line mapping.

#### Forward

Math doc:
```
h[0] = 0
for t:
    h[t+1] = tanh(Wxh*x[t] + Whh*h[t] + bh)
    z[t]   = Why*h[t+1] + by
    p[t]   = softmax(z[t])
```

Our code in `rnn_step`:
```c
Matrix* input = one_hot(input_index, rnn->V);      // x[t]
cache->x_cache[t] = input;

Matrix* xh   = dot(rnn->Wxh, input);
Matrix* hh   = dot(rnn->Whh, cache->h_cache[t]);   // uses h[t]
Matrix* sum1 = add(xh, hh);
Matrix* a    = add(sum1, rnn->bh);                 // a_t
matrix_free(xh); matrix_free(hh); matrix_free(sum1);

Matrix* h = apply(tanh, a);                        // h[t+1]
matrix_free(a);
cache->h_cache[t + 1] = h;

Matrix* Wh = dot(rnn->Why, h);
Matrix* z  = add(Wh, rnn->by);                     // z[t]
matrix_free(Wh);

Matrix* p = softmax(z);                            // p[t]
matrix_free(z);
cache->p_cache[t] = p;
```

Called from `rnn_forward`:
```c
for (int t = 0; t < T; t++) {
    rnn_step(rnn, cache, chunk[t], t);
    int target = chunk[t + 1];
    double p_target = cache->p_cache[t]->values[target][0];
    loss += -log(p_target);
}
```

#### Backward

Math doc:
```
dh_next = 0
for t backward:
    dz = p[t] - y[t]
    dh = Whyᵀ*dz + dh_next
    da = dh * (1 - h[t+1]^2)

    dWhy += dz * h[t+1]ᵀ
    dby  += dz
    dWxh += da * x[t]ᵀ
    dWhh += da * h[t]ᵀ
    dbh  += da

    dh_next = Whhᵀ * da
```

Our code in `rnn_backward` (structure only; see rnn.c for full ownership discipline):
```c
Matrix* dh_next = matrix_create(rnn->H, 1);
matrix_init(dh_next, 0.0);

for (int t = T - 1; t >= 0; t--) {
    // dz = p[t] - y[t]
    Matrix* y_t = one_hot(chunk[t + 1], rnn->V);
    Matrix* dzt = subtract(cache->p_cache[t], y_t);
    matrix_free(y_t);

    // dh = Whyᵀ · dz + dh_next
    Matrix* WhyT       = transpose(rnn->Why);
    Matrix* WhyT_dzt   = dot(WhyT, dzt);
    Matrix* dh_t       = add(WhyT_dzt, dh_next);
    /* free intermediates */

    // da = dh ⊙ (1 - h[t+1]²)
    /* build (1 - h²) via apply/subtract, then hadamard */
    Matrix* da_t = hadamard(dh_t, tanh_deriv);
    /* free intermediates */

    // Accumulate five contributions
    // dWhy += dz · h[t+1]ᵀ
    // dby  += dz
    // dWxh += da · x[t]ᵀ
    // dWhh += da · h[t]ᵀ     ← NOTE: h[t], not h[t+1]
    // dbh  += da

    // dh_next = Whhᵀ · da
    Matrix* WhhT         = transpose(rnn->Whh);
    Matrix* new_dh_next  = dot(WhhT, da_t);
    matrix_free(WhhT);
    matrix_free(dh_next);
    dh_next = new_dh_next;

    /* free iteration-local matrices */
}
matrix_free(dh_next);
```

Structurally: one dh_next set-up, T iterations of six operations (three intermediates + five accumulations + one dh_next update), one dh_next teardown.

---

### Section 105 — Reconstruction sheet

Direct mapping of the boxed equations to code:

| Math (boxed)                          | Code                                                |
|---------------------------------------|-----------------------------------------------------|
| `δ_{T+1} = 0`                         | `matrix_init(dh_next, 0.0)` before loop            |
| `δ_{z,t} = ŷ_t - y_t`                 | `subtract(cache->p_cache[t], y_t)`                 |
| `dh_t = W_hy^T δ_{z,t} + W_hh^T δ_{t+1}` | `add(dot(transpose(Why), dzt), dh_next)`        |
| `δ_t = dh_t ⊙ (1 - h_t^2)`            | `hadamard(dh_t, 1 - h_cache[t+1]²)`               |
| `dW_hy += δ_{z,t} h_t^T`              | `matrix_add_inplace(dWhy, dot(dzt, h_next_T))`    |
| `db_y += δ_{z,t}`                     | `matrix_add_inplace(dby, dzt)`                    |
| `dW_xh += δ_t x_t^T`                  | `matrix_add_inplace(dWxh, dot(da_t, x_T))`        |
| `dW_hh += δ_t h_{t-1}^T`              | `matrix_add_inplace(dWhh, dot(da_t, h_prev_T))`   |
| `db_h += δ_t`                         | `matrix_add_inplace(dbh, da_t)`                   |
| `θ ← θ - η dθ`                        | `rnn_update(rnn, grads, LR)` after loop           |

`W_hh^T δ_{t+1}` becomes `dh_next` in the code — it's pre-multiplied at the end of the previous iteration, so this iteration just reads it as-is.

---

### Sections 98-101 — Gradient checking

Implemented in `gradient_check.c` as a separate binary (own `main`, own `Makefile` target `gradcheck`, does not touch the training program).

Structure:
- `run_forward(rnn, cache, chunk)` — resets cache, sets `h[0] = 0`, runs `rnn_forward`.
- `numerical_grad(rnn, cache, chunk, param, i, j)` — perturbs one entry by ±ε, runs two forwards, returns central finite difference:
  ```
  (L(θ + ε at [i][j]) - L(θ - ε at [i][j])) / (2ε)
  ```
- `check_param(name, param, grad, ...)` — samples 5 entries from a parameter matrix, compares analytical (from `rnn_backward`) to numerical, reports relative error.

Recommended `ε = 1e-5`. Relative error thresholds:
- `< 1e-5` → OK (machine precision)
- `< 1e-3` → SUSPECT (investigate)
- larger → WRONG (bug)

Our gradient check passed with rel_err ~`1e-10` to `1e-12` across all sampled entries. That's the machine-precision agreement Section 101 promises.

---

### Section 102 — Common mistakes

Each of these came up during code review:

| Mistake                                       | How we caught / avoided it                                         |
|-----------------------------------------------|--------------------------------------------------------------------|
| 1. `h_t` vs `h_{t-1}` in `dW_hh`              | Variable names `h_prev_T` (for h[t]) vs `h_next_T` (for h[t+1]).   |
| 2. Forgetting the future gradient             | `dh_next` initialized to zero, added into `dh_t` every timestep.   |
| 3. `dh_next = da` instead of `W_hh^T · da`    | Transpose applied at end of every iteration, before rebinding.     |
| 4. Not accumulating shared parameter gradients | `matrix_add_inplace` accumulates; `gradients_zero` wipes between chunks. |
| 5. Updating parameters before finishing BPTT  | `rnn_update` is a separate function, called after `rnn_backward`.  |
| 6. Unstable softmax                            | Our `softmax` subtracts the max logit before exponentiating.       |

---

### Section 103 — Why no autodiff

Every op has an derivative we coded by hand:

| Op                     | Backward                                                           |
|------------------------|--------------------------------------------------------------------|
| `add`, `subtract`      | Gradient flows through unchanged (identity).                       |
| `dot(A, B)` w.r.t. A   | `dot(∂L/∂C, B^T)`                                                  |
| `dot(A, B)` w.r.t. B   | `dot(A^T, ∂L/∂C)`                                                  |
| `apply(tanh, a)`       | `hadamard(∂L/∂h, 1 - h²)` (using `h`, not `a`)                    |
| softmax + cross-entropy | `p - y` (elegant fusion of the two backwards into one line)       |

---

## Beyond the math doc: char-LM specialization

Things not covered in the math doc but critical for our implementation.

### One-hot inputs

The math doc's `x_t ∈ ℝ^D` in our case is a one-hot column vector. Every `dot(Wxh, x_t)` could theoretically be replaced with "select the `input_index`-th column of Wxh" — O(H) instead of O(HV). Our code uses the generic matmul; this is a future optimization.

### Sparse cross-entropy

For a one-hot target, `-Σ y_k log(p_k)` collapses to `-log(p[target])`. Our `rnn_forward` uses this shortcut:

```c
double p_target = cache->p_cache[t]->values[target][0];
if (p_target < 1e-12) p_target = 1e-12;      // guard against log(0)
total_loss += -log(p_target);
```

The gradient `δz = p - y` still touches all V entries because `y - one_hot` has one `-1` entry and V-1 zeros. Both loss and gradient forms are correct; loss is cheaper, gradient is dense.

### Truncated BPTT with hidden state carry

Real corpora are too long to backprop through end-to-end. We chunk into T=25-character sequences. Between chunks, we *carry* `h[T]` forward as the next chunk's `h[0]`, but the gradient does not flow across chunk boundaries.

Implemented in `main.c`:

```c
for (chunk_num) {
    // ... forward, backward, clip, update ...

    // Carry h across the boundary; discard everything else.
    Matrix* cache_carry = matrix_copy(cache->h_cache[T]);
    cache_reset(cache);
    cache->h_cache[0] = cache_carry;
}
```

Order matters: `matrix_copy` before `cache_reset`, because `cache_reset` frees `h_cache[T]`.

At the start of each new epoch, `h[0]` is reset to zero (fresh state, no carry across epoch boundaries).

### Gradient clipping

Vanilla RNNs are notorious for exploding gradients. Our `gradients_clip(grads, max_norm=5.0)`:

```c
void gradients_clip(RNNGradients* grads, double max_norm) {
    clip_one(grads->dWxh, max_norm);   // per-matrix L2 norm clipping
    clip_one(grads->dWhh, max_norm);
    clip_one(grads->dbh,  max_norm);
    clip_one(grads->dWhy, max_norm);
    clip_one(grads->dby,  max_norm);
}
```

Where `clip_one` computes the L2 norm of the matrix; if it exceeds `max_norm`, scale every entry by `max_norm / norm`. Direction preserved, magnitude capped.

Called between `rnn_backward` and `rnn_update` in `main.c`.

### Sampling

Not covered in the math doc. Our `rnn_sample(rnn, corpus, seed_index, N)`:

```
h = zero H×1 vector
current = seed_index
print index_to_char[current]

for i in 0..N-1:
    x = one_hot(current, V)
    <forward one step: a, h_new, z, p>   (same as rnn_step, no cache)
    current = sample_from(p)              (categorical sampling)
    print index_to_char[current]
    (free old h, h = h_new)

free final h
```

`sample_from(p)` uses inverse-CDF sampling:
1. Generate uniform `r` in `[0, 1)`.
2. Walk cumulative sum through `p`.
3. Return first index where cumulative sum exceeds `r`.

This gives proportional probability: class `k` gets sampled with probability `p[k]`.

Seeding: `srand(time(NULL))` once at start of `main()` for varied output across runs.

---

## Data flow: one chunk of training

```
main.c:
    build chunk[T+1] from corpus text starting at chunk_num * T
    gradients_zero(grads)

rnn_forward(rnn, cache, chunk):
    for t in 0..T-1:
        rnn_step:  x_t → a_t → h[t+1] → z_t → p[t]   (cached)
        loss += -log(p[t][chunk[t+1]])
    return loss

rnn_backward(rnn, cache, chunk, grads):
    dh_next = 0
    for t in T-1 down to 0:
        δz_t = p[t] - one_hot(chunk[t+1])
        dh_t = W_hy^T · δz_t + dh_next
        δa_t = dh_t ⊙ (1 - h[t+1]²)
        dWhy += δz_t · h[t+1]^T
        dby  += δz_t
        dWxh += δa_t · x[t]^T
        dWhh += δa_t · h[t]^T
        dbh  += δa_t
        dh_next = W_hh^T · δa_t

gradients_clip(grads, 5.0):
    scale each gradient matrix if its L2 norm > 5.0

rnn_update(rnn, grads, LR):
    for each of 5 params: param -= LR * grad

carry:
    h_carry = matrix_copy(cache->h_cache[T])
    cache_reset(cache)
    cache->h_cache[0] = h_carry
```

---

## Training results

On a 22.8 KB English corpus (~51 unique characters), H=32, T=25, LR=0.01, gradient clipping max_norm=5.0:

- Initial loss (uniform softmax): ~98 per chunk.
- After 25 epochs: 30-50 range.
- Sample output at loss ~40: recognizable letter frequencies, real short words (`the`, `and`, `it`, `when`), plausible punctuation, gibberish at sentence level.

Matches Karpathy's char-rnn output profile for a comparable model size. Vanilla RNN with H=32 has an inherent ceiling — real coherent output requires LSTMs, larger H, or much more data.
