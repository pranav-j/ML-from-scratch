# C Learnings

Notes on general C concepts, patterns, and traps encountered while implementing the RNN in C. Non-RNN-specific. Reference material for future C-heavy projects.

---

## 1. Pointers and heap memory

### The two-zone rule

C has two places variables live:

- **Stack**: local variables declared without `malloc`. Auto-cleaned when the function returns.
- **Heap**: memory returned by `malloc`, `calloc`, etc. Lives until you explicitly `free` it.

When you write `Matrix* xh = dot(...)`:

- The pointer `xh` (8 bytes on a 64-bit machine) sits on the stack.
- The Matrix it points at (struct + values grid, hundreds of bytes) sits on the heap.

Function return reclaims the stack pointer. The heap allocation persists — orphaned — unless you called `free`.

### Every returned pointer is owned by you

Any function that returns a `T*` is handing you a heap allocation. You now own it. Two options:

1. Keep a lasting handle to it (store in a cache, output parameter, or return further up).
2. Free it before losing scope.

Do neither → leak.

### Reassigning a pointer never frees what it used to point at

The most common C memory bug. This pattern:

```c
Matrix* h = matrix_create(H, 1);   // h → Matrix A on heap
h = matrix_create(H, 1);           // h → Matrix B on heap. Matrix A is now orphaned.
```

Matrix A leaked. To be leak-free:

```c
matrix_free(h);                    // return Matrix A to heap
h = matrix_create(H, 1);           // now bind h to Matrix B
```

Every leak in the RNN was some variant of this pattern. Rule: **before reassigning a pointer, free what it currently points at.**

### Every operation returns a fresh matrix

If matrix ops in your library all return `Matrix*`, chaining them leaks the intermediates:

```c
Matrix* a = add(dot(Wxh, x), dot(Whh, h));   // leaks BOTH dot results
```

The two `dot` results have no name and no `free`. Correct:

```c
Matrix* xh   = dot(Wxh, x);
Matrix* hh   = dot(Whh, h);
Matrix* a    = add(xh, hh);
matrix_free(xh);
matrix_free(hh);
```

Ugly. But correct. The ugliness is why in-place ops exist — for accumulations, they avoid the alloc-add-free dance.

### `free(NULL)` should be safe

Convention: `free`-style functions should guard against NULL:

```c
void matrix_free(Matrix* m) {
    if (!m) return;
    // ...
}
```

This lets you write cleaner code without conditional guards at every call site. Standard `stdlib.h`'s `free(NULL)` is safe by the C standard — follow the pattern in your own code.

### Container ownership rule

When something (like a cache) holds pointers to heap allocations, decide once who owns them.

Our RNN convention: **the cache owns everything in its slots.** Anything assigned to a cache slot must NOT be freed by the assigning function — the cache is what keeps it alive. Freeing happens in one place only: `cache_reset` or `cache_free`.

Miss this rule and you get either double-frees (crash) or use-after-frees (silent corruption).

---

## 2. Types and typing

### No operator overloading

C doesn't. `+` on pointers does pointer arithmetic. `Matrix* a + Matrix* b` is not matrix addition — it's a nonsense address computation. Use `add(a, b)`.

Same for `-`, `*`, `/`, `%`, `<`, `==`, etc. Every one is defined on primitive types. Never on your custom types.

### Format specifiers must match types

`printf("%d", loss)` with `loss` as `double` is undefined behavior. What prints is whatever bits of the double `printf` reads as an int. Value bears no relation to the actual number.

Common ones:
- `%d` → `int`
- `%ld` → `long`
- `%f` → `double` (or `float`; float is promoted)
- `%lu` → `unsigned long`
- `%p` → pointer (prints as hex address)
- `%c` → `char` (as character)
- `%s` → null-terminated string
- `%zu` → `size_t`

`-Wformat` catches most format mismatches. `-Wall` includes it. Turn on `-Werror` and format bugs won't compile.

### Character literals are integers

`'a'` is the integer 97. It is NOT a special "character type" — it's a value of type `int` (in C). Same for `'0'` (48), `' '` (32), `'\n'` (10). These are ASCII values.

Consequences:
- `printf("%d\n", 'a')` prints 97
- `arr['a']` indexes slot 97 of `arr`
- `c == 'A'` is really `c == 65`

The `%c` and `%d` in `printf` don't change the bits stored — they change how they're displayed.

### `'a'` vs `"a"` — quote style determines type

- `'a'` is a `char`. Value: 97.
- `"a"` is a `char[2]`: `{'a', '\0'}`. It's a **string** — a pointer, not a value.

Not interchangeable. `putchar("a")` doesn't compile. `strcmp(str, 'x')` doesn't compile. Single quotes = character. Double quotes = string.

### Signed `char` vs `unsigned char`

On macOS and most Linux setups, plain `char` is **signed**. That means a byte with value 200 (0xC8) — perhaps an accented character in UTF-8 — is stored as the integer -56.

Consequences:

```c
char c = 0xC8;
arr[c];   // arr[-56] — walks 56 slots BEFORE arr[0]. Undefined behavior.
```

**Rule: whenever using a `char` as an array index, cast to `unsigned char` first.**

```c
unsigned char uc = (unsigned char)c;
arr[uc];   // arr[200] — safe
```

`-Wchar-subscripts` catches this. With `-Werror`, it becomes a hard error. Worth enabling.

Pure-ASCII input avoids the trap. Any byte ≥ 128 (accented characters, curly quotes, emoji, arbitrary binary data) hits it.

### Reserved keywords

Can't use as variable names: `char`, `int`, `float`, `double`, `struct`, `if`, `for`, `return`, `void`, etc. Rename to `ch`, `c`, `character`, etc.

---

## 3. Arrays

### Array decay at function boundaries

In a function parameter list, `int arr[]` and `int arr[26]` and `int* arr` are all identical to the compiler. The `[26]` is discarded. Array size cannot be enforced at the function boundary.

Outside function parameters, they're different:
- `int arr[26]` → reserves 26 ints (104 bytes) contiguously in one memory chunk
- `int* arr` → declares one pointer (8 bytes) that must be assigned before use

When you pass an array to a function, C automatically converts it to a pointer to its first element — "array decay." Once decayed, the size is lost.

Consequence: **you cannot get an array's length inside a function.** Pass it as a separate parameter, or make it a compile-time constant.

### Bounds checking is your job

C does not check. `arr[100]` on a 10-element array walks 90 elements past the end and reads whatever bytes live there. No crash. Just undefined behavior.

Tools that help:
- **ASan** (AddressSanitizer) — catches out-of-bounds at runtime with clear line numbers.
- **UBSan** (UndefinedBehaviorSanitizer) — catches related integer overflows, misaligned reads, etc.

Enable for every dev build. Silent bugs become `AddressSanitizer: stack-buffer-overflow at rnn.c:146`.

### Off-by-one

C loops usually go `for (int i = 0; i < N; i++)`. Two off-by-one traps:

- `i <= N` when you mean `i < N` — writes past the end.
- `i < N - 1` when you mean `i < N` — misses the last element.

Array of length T+1 has valid indices 0..T (both inclusive) — that's T+1 slots. `arr[T]` is valid; `arr[T+1]` is not.

---

## 4. Structs

### `->` vs `.`

- `.` — access field through a struct value or reference.
- `->` — access field through a struct pointer. Equivalent to `(*ptr).field`.

```c
Matrix m;                  // struct value on stack
m.rows;                    // use .

Matrix* mp = &m;           // pointer to struct
mp->rows;                  // use ->
(*mp).rows;                // same thing, uglier
```

Rule of thumb: 90% of real C code uses `->` because heap-allocated structs are always accessed through pointers.

### Reading nested access left-to-right

Expressions like:

```c
cache->p_cache[t]->values[target][0]
```

Read left to right, tracking the type at each step:

```
cache                                        →  RNNCache*
cache->p_cache                               →  Matrix*[T]
cache->p_cache[t]                            →  Matrix*
cache->p_cache[t]->values                    →  double**
cache->p_cache[t]->values[target]            →  double*
cache->p_cache[t]->values[target][0]         →  double
```

Each operator peels one layer. Once you can do this in your head, complex expressions become mechanical.

### Struct-inline arrays

An array field inside a struct sits inline with the other fields:

```c
typedef struct {
    Matrix* h_cache[T + 1];   // 26 pointer slots, inline in the struct
    int meta;
} RNNCache;
```

`malloc(sizeof(RNNCache))` allocates the struct including all 26 pointer slots. But it doesn't allocate any Matrix — slots contain garbage until you assign to them.

Compare:

```c
Matrix** h_cache;              // just a pointer, 8 bytes, points nowhere
```

The latter requires a separate `malloc((T+1) * sizeof(Matrix*))` to become usable.

### Array size in a struct must be a compile-time constant

```c
typedef struct {
    Matrix* h_cache[T + 1];    // T must be a compile-time constant
} RNNCache;
```

Won't work if `T` is a runtime variable. Options:

- `#define T 25` (preprocessor macro — untyped, global)
- `enum { T = 25 };` (typed integer constant, respects scope)
- Use a pointer field (`Matrix** h_cache`) and allocate at runtime.

Prefer `enum` for local project constants: typed, scoped, doesn't pollute the preprocessor namespace.

### Container pattern: create / free / reset

Any struct that holds pointers to heap allocations typically needs three functions:

```c
Cache* cache_create(void);    // alloc struct, init slots to NULL
void   cache_free(Cache* c);  // free contents + struct
void   cache_reset(Cache* c); // free contents, leave struct alive
```

- `cache_create` runs once per session.
- `cache_reset` runs between uses (wipe old contents so slots are ready for new).
- `cache_free` runs at shutdown.

Initializing slots to NULL in `create` matters: `reset` then does `if (slot) matrix_free(slot);` and correctly skips empty slots. Without NULL init, `reset` would try to free garbage.

---

## 5. File I/O

### Always check `fopen`

```c
FILE* f = fopen(name, "rb");
if (!f) {
    fprintf(stderr, "fopen failed: %s\n", name);
    return NULL;
}
```

`fopen` returns NULL on failure (missing file, permissions, etc.). Failing to check leads to segfaults with no context later.

### `"rb"` vs `"r"`

- `"rb"` = binary mode.
- `"r"` = text mode.

On Unix they're identical. On Windows, text mode silently converts `\r\n` → `\n` on read. For byte-level modeling (character LM, checksums, image files, anything binary), always use `"rb"`.

### File size via `fseek`/`ftell`

Standard idiom:

```c
fseek(f, 0, SEEK_END);   // jump to end
long size = ftell(f);    // cursor position = file size
rewind(f);               // jump back to start
```

`SEEK_END`, `SEEK_SET`, `SEEK_CUR` — reference points for the offset:
- `SEEK_SET` → offset from start
- `SEEK_CUR` → offset from current position
- `SEEK_END` → offset from end

`fseek(f, 0, SEEK_END)` = "end + 0" = "the end."

`SEEK_*` constants live in `<stdio.h>`.

### `fread` returns the count actually read

```c
size_t got = fread(buf, 1, size, f);
```

`got` may be less than `size` (short read, EOF). Store the return value. Use it as the authoritative "how many bytes did I read." Don't use `strlen` on the buffer — it stops at the first null byte, which may exist mid-file.

---

## 6. Header files and includes

### Header guards

Every `.h` file needs one:

```c
#ifndef FOO_H
#define FOO_H
// ... contents ...
#endif
```

Prevents duplicate typedef errors when the header is included from multiple `.c` files (or transitively).

### Don't rely on transitive includes

If your `.c` uses `malloc`, `#include <stdlib.h>`. If it uses `NULL`, ditto. Even if another included header pulls it in — that's a fragile dependency that breaks between platforms.

Rule: include the header for every standard symbol you use.

Lookups (macOS/Linux): `man malloc`, `man fseek`, etc. The SYNOPSIS at the top lists required includes.

### `static` functions are file-local

```c
static double square(double x) { return x * x; }
```

`static` on a function means "not exported to the linker; not visible from other `.c` files." Use for helpers only the current file needs. Doesn't need a declaration in the header.

Not `static` = external linkage = must be declared in a header if used from elsewhere.

### No nested function definitions

C doesn't allow defining a function inside another function:

```c
int main(void) {
    void helper(int x) { ... }   // Not standard C. Won't compile in Clang.
}
```

GCC has an extension; Clang doesn't. Move helper functions to file scope, mark `static` if internal.

---

## 7. Numerical gotchas

### `log(0) = -inf`

Common trap in cross-entropy: if a probability underflows to exactly zero, `-log(0) = +inf`, and the loss becomes garbage. Guard:

```c
double p = ...;
if (p < 1e-12) p = 1e-12;
loss += -log(p);
```

Same idea for divisions where a denominator might underflow.

### Softmax numerical stability

Naive softmax overflows for logits above ~700:

```c
exp(z_i) / sum(exp(z_j))    // if any z_j > 700, this is inf/inf = NaN
```

Standard fix: subtract the max logit before exponentiating. Mathematically identical (softmax is shift-invariant), numerically stable:

```c
double m = max(z);
exp(z_i - m) / sum(exp(z_j - m))
```

### Float precision for finite differences

Doubles have ~15 significant digits. For sums of many small terms (like gradients across timesteps), precision is fine. For subtracting nearly-equal numbers (like numerical gradients: `(L(θ+ε) - L(θ-ε))`), precision matters — choose ε carefully:

- Too small (say 1e-15) → loss of precision from subtraction.
- Too large (say 1e-1) → approximation error dominates.
- Sweet spot: `ε ≈ 1e-5`.

---

## 8. Compiler warnings save time

### Turn on `-Wall -Wextra -Werror`

- `-Wall` — enable most useful warnings.
- `-Wextra` — enable more.
- `-Werror` — treat warnings as errors. Prevents shipping code with warnings.

The compiler catches:
- Format string type mismatches (`%d` vs `%f`).
- Signed-char array subscripts.
- Uninitialized variables.
- `=+` vs `+=` typos (unary plus assign vs compound assign).
- Missing return values.
- Shadowed variables.
- Unused variables.
- Comparison type mismatches.

Every warning ignored in this project was a real bug. `-Werror` forces the discipline.

### Read the actual error message

Clang error output is precise:

```
error: array subscript is of type 'char' [-Wchar-subscripts]
    int current_index = corpus->char_to_index[seed_char];
                                             ^~~~~~~~~~
```

Line number. Column. Underline pointing at the offending token. Warning flag name. Message is usually a direct clue. Read it before flailing.

---

## 9. Sanitizers

### AddressSanitizer (ASan)

Catches heap and stack memory bugs at runtime:
- Use-after-free
- Heap buffer overflow
- Stack buffer overflow
- Double-free
- Invalid memory accesses

Flags:

```makefile
debug: CFLAGS += -O0 -g -fsanitize=address,undefined
debug: LDLIBS += -fsanitize=address,undefined
```

**Both compile and link flags are required.** Compile-time instruments the code. Link-time adds the runtime library. Missing the link flag gives you a wall of `___asan_init` unresolved-symbol errors.

Sample output:

```
==1234==ERROR: AddressSanitizer: stack-buffer-overflow ... at rnn.c:146
'chunk' (line 23) <== Memory access at offset 148 overflows this variable
```

Line number, variable name, exact overflow location. Turns "mysterious crash" into "here's the bug."

### UndefinedBehaviorSanitizer (UBSan)

Same flag: `-fsanitize=undefined` (combined into `-fsanitize=address,undefined`).

Catches:
- Signed integer overflow
- Divide by zero
- Misaligned pointer dereferences
- Some UB categories not caught by ASan

### Use during development, disable for release

Sanitized builds are 2-3x slower. Fine for development, not fine for training runs. Standard pattern: `make debug` for development, plain `make` for production.

### Object files can't mix

`.o` files compiled with sanitizers reference sanitizer runtime symbols. Linking them against a non-sanitized target fails. Always `make clean` when switching between sanitized and non-sanitized builds.

---

## 10. Makefile discipline

### Target-specific variables

```makefile
debug: CFLAGS += -O0 -g -fsanitize=address,undefined
debug: LDLIBS += -fsanitize=address,undefined
debug: clean $(TARGET)
```

When building the `debug` target, `CFLAGS` gets extra flags appended. Regular `make` doesn't see them. Same source, different build.

### `.PHONY` for command targets

```makefile
.PHONY: all clean debug
```

Tells make these are not files. Prevents confusion if a file named `clean` appears in the directory.

### Circular target/binary name clash

```makefile
gradcheck: gradcheck                    # circular! make drops the dep silently
    $(CC) ... -o $@
```

If a target uses its own name (or a variable resolving to it) as a dependency, `make` reports "circular dependency dropped" and does the wrong thing.

Give binaries and convenience targets different names — `gradcheck` (binary) vs `gc-debug` (convenience target).

### Object files aren't cross-platform

`.o` compiled on Linux/x86 won't link on macOS/ARM. If you sync a project across machines, `make clean` first. Otherwise you get inscrutable linker errors.

---

## 11. Debugging discipline

### Fix the whole review before recompiling

Fast workflow: read the whole review, fix all items in the source, then run the compiler once.

Slow workflow: fix one thing, recompile, fix the next thing, recompile.

Every "recompile-fix" cycle costs time. Batch the fixes.

### Type-check every expression as you write

Reflex: whenever you write `expr[i]` or `expr.field`, ask "what is the type of `expr`?" This catches:
- `cache->p_cache = y` — missing `[t]` (`p_cache` is `Matrix*[T]`, not `Matrix*`)
- `if (cache->h_cache)` — missing `[t]` (`h_cache` is an array, never NULL)
- `arr[bigptr]` — bigptr isn't an integer

Slowing down for two seconds per line eliminates a whole class of bugs.

### Similarly-named variables are a smell

Two variables both named `h_cacheT` in the same scope — one for `h[t+1]^T`, one for `h[t]^T`. C won't allow it, and even if it did, you'd forget which is which.

Rename to `h_next_T` and `h_prev_T` — names call out *meaning*, not derivation. This is one of the single most impactful habits for keeping BPTT-style code readable.

### The compiler already found half the bugs

Every "why doesn't this work" moment in this project had a compiler warning or error pointing at it. Reading the error message before flailing is faster than any other debugging technique.

---

## 12. Design patterns that emerged

### Separate binary for verification

Gradient checking is a *separate program* with its own `main()`, compiled into its own binary. Not called from the main training program. Not part of production code.

Why: different setup (tiny model, fixed seed, no training loop), different purpose (verify math, not train), different lifetime (runs once, never again after passing). Bloating the training binary with mode flags would be worse.

Pattern: create per-purpose binaries that share a common library.

### Two forward paths, one implementation

`rnn_step` is used by both training (via `rnn_forward`) and inference (via `rnn_sample`). The step is: `(x_t, h_prev) → (h_new, p_t)`. Training caches everything; inference discards everything except the last `h`.

Separating "one step" from "many steps in a loop" made both use cases trivial to write. Similar factoring works for lots of numerical code.

### In-place accumulators vs pure functions

Every op in the shared library returns a fresh matrix (pure functional style). Great for readability, terrible for accumulation loops that run millions of times.

Adding `matrix_add_inplace` (mutates its first argument, returns void) collapsed 6-line accumulations to 3 lines and cut allocations by 5x per timestep.

Rule: pure functions for one-shot work, in-place ops for hot loops.

---

Not exhaustive. But these patterns and traps came up repeatedly. Building the reflexes above is what separates "I understand C" from "I can ship a numerics-heavy C project without leaks or crashes."
