# Recurrent Neural Network Mathematics From First Principles

## A self-contained derivation of a vanilla RNN with \(D\)-dimensional input, \(H\)-dimensional hidden state, and \(C\)-class softmax output

This document is organized around one question:

> **We ultimately need the gradients of the RNN's weights and biases. What derivative do we need first, why do we need it, and where does it get plugged next?**

The RNN is:

\[
x_t \rightarrow h_t \rightarrow \hat y_t
\]

at every time step \(t\), with the hidden state carried forward through time:

\[
h_{t-1}\rightarrow h_t.
\]

We use:

- a linear hidden preactivation,
- `tanh` in the hidden state,
- a linear output layer,
- softmax in the output,
- cross-entropy loss,
- gradient descent,
- backpropagation through time (BPTT).

The main derivation assumes a **many-to-many** RNN: every time step has a target and therefore contributes to the sequence loss. A many-to-one variant is covered later.

---

# 0. THE MAP — READ THIS FIRST

The single most important thing in this document is the dependency structure.

## 0.1 What we ultimately need

The trainable parameters are:

\[
W_{xh},\quad W_{hh},\quad b_h,\quad W_{hy},\quad b_y.
\]

To train them, we need:

\[
\boxed{
\frac{\partial L}{\partial W_{xh}},\quad
\frac{\partial L}{\partial W_{hh}},\quad
\frac{\partial L}{\partial b_h},\quad
\frac{\partial L}{\partial W_{hy}},\quad
\frac{\partial L}{\partial b_y}
}
\tag{0.1}
\]

These five gradients are the actual destination of the whole derivation.

---

## 0.2 The forward dependency

At time step \(t\), the RNN computes:

\[
\boxed{
x_t,\ h_{t-1}
\rightarrow
a_t
\rightarrow
h_t
\rightarrow
z_t
\rightarrow
\hat y_t
\rightarrow
L_t
}
\tag{0.2}
\]

where:

\[
a_t=W_{xh}x_t+W_{hh}h_{t-1}+b_h
\tag{0.3}
\]

\[
h_t=\tanh(a_t)
\tag{0.4}
\]

\[
z_t=W_{hy}h_t+b_y
\tag{0.5}
\]

\[
\hat y_t=\operatorname{softmax}(z_t)
\tag{0.6}
\]

\[
L_t=-\sum_{k=1}^{C}y_{t,k}\log\hat y_{t,k}.
\tag{0.7}
\]

The total sequence loss is:

\[
\boxed{
L=\sum_{t=1}^{T}L_t
}
\tag{0.8}
\]

or, if you prefer a mean loss,

\[
L=\frac1T\sum_{t=1}^{T}L_t.
\tag{0.9}
\]

The mean only introduces a factor \(1/T\) into the gradients.

---

## 0.3 Why the RNN dependency is different

In a feed-forward network, a hidden unit might affect one downstream computation.

In an RNN, \(h_t\) has **two jobs**:

1. it contributes to the output at time \(t\),
2. it becomes part of the input to the next recurrent step.

So:

\[
h_t
\rightarrow
z_t
\rightarrow
L_t
\]

and simultaneously:

\[
h_t
\rightarrow
a_{t+1}
\rightarrow
h_{t+1}
\rightarrow
h_{t+2}
\rightarrow\cdots
\]

Therefore, when we calculate the total gradient with respect to \(h_t\), the contributions from the current output path and the future recurrent path must be **added**.

This is the central idea of BPTT.

---

## 0.4 Backward dependency

Backpropagation reverses the arrows.

At a given time step:

\[
\boxed{
L_t
\rightarrow
\hat y_t
\rightarrow
z_t
\rightarrow
h_t
\rightarrow
a_t
}
\tag{0.10}
\]

But \(h_t\) also receives gradient from the future:

\[
L_{t+1:T}
\rightarrow
h_t.
\]

The key recurrence is therefore:

\[
\boxed{
\frac{\partial L}{\partial h_t}
=
\underbrace{
\frac{\partial L_t}{\partial h_t}
}_{\text{current output}}
+
\underbrace{
\frac{\partial L_{t+1:T}}{\partial h_t}
}_{\text{future outputs}}
}
\tag{0.11}
\]

Once that quantity is known, we cross `tanh` to get:

\[
\frac{\partial L}{\partial a_t}.
\]

That derivative then directly gives the hidden parameter gradients.

---

## 0.5 The full dependency map

```text
                            FORWARD

                   ┌─────────────────────┐
x_t ──────────────►│ Wxh x_t             │
                   │                     │
h_{t-1} ─────────►│ Whh h_{t-1}          │
                   │                     │
                   │       + b_h         │
                   └──────────┬──────────┘
                              │
                              ▼
                             a_t
                              │
                            tanh
                              │
                              ▼
                             h_t
                           /     \
                          /       \
                         ▼         ▼
                   Why h_t       h_t → h_{t+1}
                      │
                      ▼
                     z_t
                      │
                   softmax
                      │
                      ▼
                    ŷ_t
                      │
                cross-entropy
                      │
                      ▼
                     L_t
```

Backward:

```text
L_t
│
▼
ŷ_t - y_t
│
▼
δz_t = ∂L_t/∂z_t
│
├─────────────────────► dWhy
│
├─────────────────────► dby
│
▼
Whyᵀ δz_t
│
├──────────────────────┐
│                      │
│                      ▼
│               Whhᵀ δ_{t+1}
│                      │
└──────────┬───────────┘
           ▼
          dh_t = ∂L/∂h_t
           │
           ▼
     ⊙ (1 - h_t²)
           │
           ▼
      δ_t = ∂L/∂a_t
           │
      ┌────┼───────────┐
      ▼    ▼           ▼
    dWxh dWhh         dbh
      │    │
      └────┴──────► carry δ_t backward in time
```

If you lose track, come back to this section.

---

## 0.6 The prerequisite table

| What we ultimately need | What must be known first | Why that prerequisite is needed |
|---|---|---|
| \(\partial L/\partial W_{hy}\) | \(\partial L/\partial z_t\) and \(h_t\) | \(W_{hy}\) directly creates \(z_t\) |
| \(\partial L/\partial b_y\) | \(\partial L/\partial z_t\) | \(b_y\) directly creates \(z_t\) |
| \(\partial L/\partial h_t\) | current output gradient + future recurrent gradient | \(h_t\) has two downstream paths |
| \(\partial L/\partial a_t\) | \(\partial L/\partial h_t\) and tanh derivative | \(h_t=\tanh(a_t)\) |
| \(\partial L/\partial W_{xh}\) | \(\partial L/\partial a_t\) and \(x_t\) | \(W_{xh}\) directly creates \(a_t\) |
| \(\partial L/\partial W_{hh}\) | \(\partial L/\partial a_t\) and \(h_{t-1}\) | \(W_{hh}\) directly creates \(a_t\) |
| \(\partial L/\partial b_h\) | \(\partial L/\partial a_t\) | \(b_h\) directly creates \(a_t\) |
| total parameter gradients | per-time-step contributions | the same parameters are reused at every time step |

The actual derivation order is therefore:

\[
\boxed{
L_t
\rightarrow
\frac{\partial L_t}{\partial z_t}
}
\]

then:

\[
\boxed{
\frac{\partial L_t}{\partial z_t}
\rightarrow
\left\{
\frac{\partial L}{\partial W_{hy}},
\frac{\partial L}{\partial b_y},
\frac{\partial L_t}{\partial h_t}
\right\}
}
\]

then:

\[
\boxed{
\text{current } \frac{\partial L}{\partial h_t}
+
\text{future recurrent contribution}
\rightarrow
\frac{\partial L}{\partial a_t}
}
\]

and finally:

\[
\boxed{
\frac{\partial L}{\partial a_t}
\rightarrow
\left\{
\frac{\partial L}{\partial W_{xh}},
\frac{\partial L}{\partial W_{hh}},
\frac{\partial L}{\partial b_h}
\right\}.
}
\]

---

# 1. What problem does an RNN solve?

A standard feed-forward network processes an input:

\[
x\rightarrow \hat y.
\]

An RNN processes a sequence:

\[
x_1,x_2,\ldots,x_T.
\]

The important difference is that the prediction at time \(t\) may need information from earlier inputs.

We therefore introduce a hidden state:

\[
h_t.
\]

The hidden state summarizes information from the past that the network has learned to preserve.

The recurrence is:

\[
\boxed{
h_t=f(x_t,h_{t-1})
}
\tag{1.1}
\]

For a vanilla RNN, \(f\) will be:

\[
\tanh(W_{xh}x_t+W_{hh}h_{t-1}+b_h).
\]

Thus:

\[
\boxed{
h_t
=
\tanh(
W_{xh}x_t
+
W_{hh}h_{t-1}
+
b_h
)
}
\tag{1.2}
\]

This single equation is the defining equation of the vanilla RNN.

---

# 2. Notation and dimensions

We use column vectors.

## 2.1 Sequence input

At time \(t\):

\[
\boxed{
x_t\in\mathbb R^D
}
\tag{2.1}
\]

There are \(T\) such vectors:

\[
x_1,\ldots,x_T.
\]

---

## 2.2 Hidden state

The RNN has \(H\) hidden neurons:

\[
\boxed{
h_t\in\mathbb R^H
}
\tag{2.2}
\]

The initial hidden state is:

\[
\boxed{
h_0\in\mathbb R^H.
}
\tag{2.3}
\]

For the first implementation, choose:

\[
\boxed{
h_0=0.
}
\tag{2.4}
\]

---

## 2.3 Input-to-hidden parameters

The input \(x_t\) has \(D\) components and must produce an \(H\)-dimensional contribution.

Therefore:

\[
\boxed{
W_{xh}\in\mathbb R^{H\times D}
}
\tag{2.5}
\]

and:

\[
\boxed{
b_h\in\mathbb R^H.
}
\tag{2.6}
\]

The matrix multiplication has the shape:

\[
(H\times D)(D\times1)
=
(H\times1).
\]

---

## 2.4 Hidden-to-hidden parameters

The previous hidden state has \(H\) components and the next hidden state also has \(H\) components.

Therefore:

\[
\boxed{
W_{hh}\in\mathbb R^{H\times H}.
}
\tag{2.7}
\]

This is the recurrent matrix.

It is the parameter reused from one time step to the next.

---

## 2.5 Hidden-to-output parameters

Suppose there are \(C\) output classes.

Then:

\[
\boxed{
W_{hy}\in\mathbb R^{C\times H}
}
\tag{2.8}
\]

and:

\[
\boxed{
b_y\in\mathbb R^C.
}
\tag{2.9}
\]

The logits satisfy:

\[
z_t\in\mathbb R^C.
\]

The predicted probability vector satisfies:

\[
\hat y_t\in\mathbb R^C.
\]

The target is:

\[
y_t\in\mathbb R^C.
\]

---

## 2.6 Summary of dimensions

| Quantity | Shape |
|---|---|
| \(x_t\) | \(D\times1\) |
| \(h_t\) | \(H\times1\) |
| \(a_t\) | \(H\times1\) |
| \(W_{xh}\) | \(H\times D\) |
| \(W_{hh}\) | \(H\times H\) |
| \(b_h\) | \(H\times1\) |
| \(z_t\) | \(C\times1\) |
| \(W_{hy}\) | \(C\times H\) |
| \(b_y\) | \(C\times1\) |
| \(\hat y_t\) | \(C\times1\) |
| \(y_t\) | \(C\times1\) |

---

# 3. The forward pass

We now calculate everything in the order the network needs it.

At each time step \(t\), there are five conceptual stages:

\[
x_t,h_{t-1}
\rightarrow
a_t
\rightarrow
h_t
\rightarrow
z_t
\rightarrow
\hat y_t.
\]

---

## 3.1 Hidden preactivation

The recurrent affine transformation is:

\[
\boxed{
a_t
=
W_{xh}x_t
+
W_{hh}h_{t-1}
+
b_h
}
\tag{3.1}
\]

Why do we calculate \(a_t\)?

Because it is the quantity to which we apply the nonlinear activation.

It also becomes the quantity whose gradient directly gives all hidden-layer parameter gradients.

Dimensions:

\[
(H\times D)(D\times1)
+
(H\times H)(H\times1)
+
(H\times1)
=
H\times1.
\]

---

## 3.2 Scalar form of the hidden preactivation

For hidden neuron \(i\):

\[
\boxed{
a_{t,i}
=
\sum_{j=1}^{D}
W_{xh,ij}x_{t,j}
+
\sum_{k=1}^{H}
W_{hh,ik}h_{t-1,k}
+
b_{h,i}
}
\tag{3.2}
\]

This equation is worth keeping visible because the parameter derivatives will come directly from it.

For example:

\[
\frac{\partial a_{t,i}}{\partial W_{xh,ij}}
=
x_{t,j}
\]

and:

\[
\frac{\partial a_{t,i}}{\partial W_{hh,ik}}
=
h_{t-1,k}.
\]

---

## 3.3 Hidden activation

Apply `tanh` elementwise:

\[
\boxed{
h_t=\tanh(a_t)
}
\tag{3.3}
\]

At scalar level:

\[
\boxed{
h_{t,i}=\tanh(a_{t,i})
}
\tag{3.4}
\]

Why do we need the nonlinearity?

Without it, repeated affine transformations would collapse into a single linear/affine transformation and the network would not have the same nonlinear representational power.

The hidden state \(h_t\) has two downstream uses:

\[
h_t\rightarrow z_t
\]

and:

\[
h_t\rightarrow a_{t+1}.
\]

This fork is what later creates the two gradient contributions.

---

## 3.4 Output logits

The hidden state is mapped into output-logit space:

\[
\boxed{
z_t=W_{hy}h_t+b_y.
}
\tag{3.5}
\]

At scalar level:

\[
\boxed{
z_{t,k}
=
\sum_{j=1}^{H}
W_{hy,kj}h_{t,j}
+
b_{y,k}.
}
\tag{3.6}
\]

Why do we need \(z_t\)?

Because the output probability distribution is obtained by applying softmax to the logits.

---

## 3.5 Softmax

For output class \(k\):

\[
\boxed{
\hat y_{t,k}
=
\frac{e^{z_{t,k}}}
{\sum_{r=1}^{C}e^{z_{t,r}}}
}
\tag{3.7}
\]

or:

\[
\boxed{
\hat y_t=\operatorname{softmax}(z_t).
}
\tag{3.8}
\]

The probabilities satisfy:

\[
0<\hat y_{t,k}<1
\]

and:

\[
\boxed{
\sum_{k=1}^{C}\hat y_{t,k}=1.
}
\tag{3.9}
\]

---

## 3.6 Cross-entropy loss at one time step

The loss for time step \(t\) is:

\[
\boxed{
L_t
=
-\sum_{k=1}^{C}
y_{t,k}\log\hat y_{t,k}.
}
\tag{3.10}
\]

For a one-hot target, if the correct class is \(c\):

\[
\boxed{
L_t=-\log\hat y_{t,c}.
}
\tag{3.11}
\]

The total loss is:

\[
\boxed{
L=\sum_{t=1}^{T}L_t.
}
\tag{3.12}
\]

So the complete forward chain is:

\[
\boxed{
(x_t,h_{t-1})
\rightarrow
a_t
\rightarrow
h_t
\rightarrow
z_t
\rightarrow
\hat y_t
\rightarrow
L_t.
}
\tag{3.13}
\]

---

# 4. Unrolling the RNN through time

The recurrence is easier to reason about if we explicitly write several time steps.

Suppose:

\[
h_0=0.
\]

Then:

\[
a_1=W_{xh}x_1+W_{hh}h_0+b_h
\tag{4.1}
\]

\[
h_1=\tanh(a_1)
\tag{4.2}
\]

Then:

\[
a_2=W_{xh}x_2+W_{hh}h_1+b_h
\tag{4.3}
\]

\[
h_2=\tanh(a_2)
\tag{4.4}
\]

Then:

\[
a_3=W_{xh}x_3+W_{hh}h_2+b_h
\tag{4.5}
\]

\[
h_3=\tanh(a_3).
\tag{4.6}
\]

So the graph is:

```text
x1 ──► h1 ──► h2 ──► h3 ──► ...
       ▲       ▲       ▲
       │       │       │
      x?      x2      x3
```

More explicitly:

```text
x1 ──► a1 ──► h1 ──► a2 ──► h2 ──► a3 ──► h3
             ▲          ▲          ▲
             │          │          │
             x2         x3         x4
```

The same \(W_{xh}\), \(W_{hh}\), and \(b_h\) occur at every step.

This is parameter sharing through time.

---

# 5. Why \(h_t\) represents memory

Expand the recurrence:

\[
h_t
=
\tanh(
W_{xh}x_t
+
W_{hh}h_{t-1}
+
b_h).
\]

But:

\[
h_{t-1}
=
\tanh(
W_{xh}x_{t-1}
+
W_{hh}h_{t-2}
+
b_h).
\]

Therefore:

\[
h_t
\]

depends on:

\[
x_t,x_{t-1},x_{t-2},\ldots
\]

through repeated application of the recurrence.

The network does not explicitly store every previous input.

Instead, the past is compressed into the current state:

\[
\boxed{
h_t=\text{learned representation of the relevant past and current input}.
}
\tag{5.1}
\]

---

# 6. What backpropagation is trying to do

Gradient descent updates a parameter \(\theta\) according to:

\[
\boxed{
\theta_{\text{new}}
=
\theta
-
\eta
\frac{\partial L}{\partial\theta}
}
\tag{6.1}
\]

where \(\eta\) is the learning rate.

Therefore we need:

\[
\frac{\partial L}{\partial W_{xh}},
\quad
\frac{\partial L}{\partial W_{hh}},
\quad
\frac{\partial L}{\partial b_h},
\quad
\frac{\partial L}{\partial W_{hy}},
\quad
\frac{\partial L}{\partial b_y}.
\]

Backpropagation is the efficient application of the chain rule needed to obtain these quantities.

The complication is that each parameter is reused at every time step.

---

# 7. The chain rule is the entire engine

For scalar variables:

\[
L=L(u),\qquad u=u(v),
\]

then:

\[
\boxed{
\frac{\partial L}{\partial v}
=
\frac{\partial L}{\partial u}
\frac{\partial u}{\partial v}.
}
\tag{7.1}
\]

The interpretation is:

> The sensitivity of the loss to \(v\) equals the sensitivity to the downstream quantity \(u\), multiplied by how sensitive \(u\) is to \(v\).

When a variable affects the loss through multiple paths, the path contributions add.

That second point is crucial for \(h_t\):

\[
h_t
\rightarrow
L_t
\]

and:

\[
h_t
\rightarrow
h_{t+1}
\rightarrow
L_{t+1:T}.
\]

Therefore:

\[
\frac{\partial L}{\partial h_t}
=
\text{current contribution}
+
\text{future contribution}.
\]

---

# 8. FIRST DESTINATION: \(\partial L_t/\partial z_t\)

The output logits are immediately followed by softmax and cross-entropy:

\[
z_t\rightarrow\hat y_t\rightarrow L_t.
\]

The first useful backward quantity is therefore:

\[
\boxed{
\frac{\partial L_t}{\partial z_t}.
}
\tag{8.1}
\]

Why?

Because \(z_t\) is directly produced by:

\[
W_{hy},b_y,
\]

and therefore once \(\partial L_t/\partial z_t\) is known, we can immediately calculate:

\[
\frac{\partial L_t}{\partial W_{hy}}
\]

and:

\[
\frac{\partial L_t}{\partial b_y}.
\]

The same quantity also lets us continue backward into \(h_t\).

---

# 9. Derivative of cross-entropy with respect to \(\hat y_t\)

At a fixed time step, write:

\[
L_t=-\sum_{k=1}^{C}y_{t,k}\log\hat y_{t,k}.
\]

For output component \(l\):

\[
\boxed{
\frac{\partial L_t}{\partial\hat y_{t,l}}
=
-\frac{y_{t,l}}{\hat y_{t,l}}.
}
\tag{9.1}
\]

This is **not yet** the derivative we want.

We still have to cross the softmax.

---

# 10. Softmax derivative — why there are two cases

Write:

\[
\hat y_{t,k}
=
\frac{e^{z_{t,k}}}{S_t},
\tag{10.1}
\]

where:

\[
\boxed{
S_t=\sum_j e^{z_{t,j}}.
}
\tag{10.2}
\]

We want:

\[
\boxed{
\frac{\partial\hat y_{t,k}}
{\partial z_{t,l}}.
}
\tag{10.3}
\]

There are two cases:

\[
k=l
\]

and:

\[
k\neq l.
\]

The cases differ because \(z_{t,l}\) may appear in the numerator as well as in the denominator.

---

# 11. A small but important derivative: \(\partial S_t/\partial z_{t,l}\)

Recall:

\[
S_t=\sum_j e^{z_{t,j}}.
\]

Differentiating with respect to \(z_{t,l}\), every term except \(e^{z_{t,l}}\) is constant:

\[
\boxed{
\frac{\partial S_t}{\partial z_{t,l}}
=
e^{z_{t,l}}.
}
\tag{11.1}
\]

This derivative will be used in both softmax cases.

---

# 12. Kronecker delta

The Kronecker delta is:

\[
\boxed{
\delta_{ij}
=
\begin{cases}
1,&i=j,\\
0,&i\neq j.
\end{cases}
}
\tag{12.1}
\]

It is useful for representing statements such as:

\[
\frac{\partial z_{t,j}}{\partial z_{t,l}}
=
\delta_{jl}.
\]

It is simply a compact way to encode whether two indices are equal.

---

# 13. Softmax derivative: case \(k=l\)

Consider:

\[
\hat y_{t,k}
=
\frac{e^{z_{t,k}}}{S_t}.
\]

Differentiate with respect to \(z_{t,k}\).

Using the quotient rule:

\[
\frac{\partial\hat y_{t,k}}{\partial z_{t,k}}
=
\frac{
S_t e^{z_{t,k}}
-
e^{z_{t,k}}e^{z_{t,k}}
}{
S_t^2
}.
\tag{13.1}
\]

So:

\[
=
\frac{e^{z_{t,k}}}{S_t}
\left(
1-\frac{e^{z_{t,k}}}{S_t}
\right).
\tag{13.2}
\]

Since:

\[
\hat y_{t,k}
=
\frac{e^{z_{t,k}}}{S_t},
\]

we obtain:

\[
\boxed{
\frac{\partial\hat y_{t,k}}
{\partial z_{t,k}}
=
\hat y_{t,k}(1-\hat y_{t,k}).
}
\tag{13.3}
\]

---

# 14. Softmax derivative: case \(k\neq l\)

Now consider:

\[
\frac{\partial\hat y_{t,k}}
{\partial z_{t,l}},
\qquad k\neq l.
\]

The numerator:

\[
e^{z_{t,k}}
\]

does not depend on \(z_{t,l}\), so its derivative is zero.

The denominator does depend on \(z_{t,l}\):

\[
\frac{\partial S_t}{\partial z_{t,l}}
=
e^{z_{t,l}}.
\]

Therefore:

\[
\frac{\partial\hat y_{t,k}}
{\partial z_{t,l}}
=
-\frac{
e^{z_{t,k}}e^{z_{t,l}}
}{
S_t^2
}.
\tag{14.1}
\]

Recognize the softmax probabilities:

\[
\frac{e^{z_{t,k}}}{S_t}
=
\hat y_{t,k}
\]

and:

\[
\frac{e^{z_{t,l}}}{S_t}
=
\hat y_{t,l}.
\]

Therefore:

\[
\boxed{
\frac{\partial\hat y_{t,k}}
{\partial z_{t,l}}
=
-\hat y_{t,k}\hat y_{t,l},
\qquad k\neq l.
}
\tag{14.2}
\]

The negative sign is important.

Increasing one logit raises its own probability while reducing the probability mass available to the other classes.

---

# 15. The softmax Jacobian

The softmax Jacobian is the matrix:

\[
J_{kl}
=
\frac{\partial\hat y_{t,k}}
{\partial z_{t,l}}.
\]

Therefore:

\[
\boxed{
J_{kl}
=
\begin{cases}
\hat y_{t,k}(1-\hat y_{t,k}),&k=l,\\[4pt]
-\hat y_{t,k}\hat y_{t,l},&k\neq l.
\end{cases}
}
\tag{15.1}
\]

In compact matrix form:

\[
\boxed{
J
=
\operatorname{diag}(\hat y_t)
-
\hat y_t\hat y_t^T.
}
\tag{15.2}
\]

For a direct implementation of softmax + cross-entropy, we do **not** need to explicitly build this Jacobian. The chain-rule product simplifies.

---

# 16. NOW combine cross-entropy and softmax

We want:

\[
\boxed{
\frac{\partial L_t}{\partial z_{t,l}}.
}
\]

Because changing \(z_{t,l}\) affects every softmax output:

\[
\boxed{
\frac{\partial L_t}{\partial z_{t,l}}
=
\sum_k
\frac{\partial L_t}{\partial\hat y_{t,k}}
\frac{\partial\hat y_{t,k}}{\partial z_{t,l}}.
}
\tag{16.1}
\]

Substitute:

\[
\frac{\partial L_t}{\partial\hat y_{t,k}}
=
-\frac{y_{t,k}}{\hat y_{t,k}}.
\]

So:

\[
\frac{\partial L_t}{\partial z_{t,l}}
=
\sum_k
\left(
-\frac{y_{t,k}}{\hat y_{t,k}}
\right)
\frac{\partial\hat y_{t,k}}{\partial z_{t,l}}.
\tag{16.2}
\]

Split into the \(k=l\) term and \(k\neq l\) terms:

\[
\frac{\partial L_t}{\partial z_{t,l}}
=
\left(
-\frac{y_{t,l}}{\hat y_{t,l}}
\right)
\hat y_{t,l}(1-\hat y_{t,l})
+
\sum_{k\neq l}
\left(
-\frac{y_{t,k}}{\hat y_{t,k}}
\right)
(-\hat y_{t,k}\hat y_{t,l}).
\tag{16.3}
\]

The first part is:

\[
-y_{t,l}(1-\hat y_{t,l})
=
-y_{t,l}+y_{t,l}\hat y_{t,l}.
\tag{16.4}
\]

The second part is:

\[
\hat y_{t,l}
\sum_{k\neq l}y_{t,k}.
\tag{16.5}
\]

For a one-hot target:

\[
\sum_k y_{t,k}=1
\]

so:

\[
\sum_{k\neq l}y_{t,k}
=
1-y_{t,l}.
\tag{16.6}
\]

Therefore:

\[
\frac{\partial L_t}{\partial z_{t,l}}
=
-y_{t,l}+y_{t,l}\hat y_{t,l}
+
\hat y_{t,l}(1-y_{t,l}).
\]

Expand:

\[
=
-y_{t,l}
+
y_{t,l}\hat y_{t,l}
+
\hat y_{t,l}
-
y_{t,l}\hat y_{t,l}.
\]

The middle terms cancel:

\[
\boxed{
\frac{\partial L_t}{\partial z_{t,l}}
=
\hat y_{t,l}-y_{t,l}.
}
\tag{16.7}
\]

Since this is true for every \(l\):

\[
\boxed{
\frac{\partial L_t}{\partial z_t}
=
\hat y_t-y_t.
}
\tag{16.8}
\]

This simplification is one of the most important formulas in the entire RNN derivation.

---

# 17. Introducing the output error signal

Define:

\[
\boxed{
\delta_{z,t}
\equiv
\frac{\partial L_t}{\partial z_t}.
}
\tag{17.1}
\]

Then:

\[
\boxed{
\delta_{z,t}
=
\hat y_t-y_t.
}
\tag{17.2}
\]

Why name this quantity?

Because it is used repeatedly:

- to calculate \(dW_{hy}\),
- to calculate \(db_y\),
- to propagate gradient into \(h_t\).

It is useful to keep the name distinct from the hidden recurrent error \(\delta_t\).

---

# 18. NOW \(\delta_{z,t}\) gives the output-layer gradients

We have:

\[
z_t=W_{hy}h_t+b_y.
\]

The parameters \(W_{hy}\) and \(b_y\) directly create \(z_t\).

Therefore \(\delta_{z,t}\) is exactly the gradient we need to calculate their local parameter gradients.

---

## 18.1 Gradient of \(W_{hy}\)

At scalar level:

\[
z_{t,k}
=
\sum_j
W_{hy,kj}h_{t,j}
+
b_{y,k}.
\]

Therefore:

\[
\frac{\partial z_{t,k}}
{\partial W_{hy,kj}}
=
h_{t,j}.
\tag{18.1}
\]

Apply the chain rule:

\[
\frac{\partial L_t}
{\partial W_{hy,kj}}
=
\frac{\partial L_t}{\partial z_{t,k}}
\frac{\partial z_{t,k}}
{\partial W_{hy,kj}}.
\]

Therefore:

\[
\boxed{
\frac{\partial L_t}
{\partial W_{hy,kj}}
=
\delta_{z,t,k}h_{t,j}.
}
\tag{18.2}
\]

In matrix form:

\[
\boxed{
\frac{\partial L_t}{\partial W_{hy}}
=
\delta_{z,t}h_t^T.
}
\tag{18.3}
\]

Dimensions:

\[
(C\times1)(1\times H)
=
C\times H,
\]

exactly the shape of \(W_{hy}\).

---

## 18.2 Gradient of \(b_y\)

Because:

\[
z_{t,k}=\cdots+b_{y,k},
\]

we have:

\[
\frac{\partial z_{t,k}}{\partial b_{y,k}}=1.
\]

Therefore:

\[
\boxed{
\frac{\partial L_t}{\partial b_y}
=
\delta_{z,t}.
}
\tag{18.4}
\]

---

# 19. Propagate the current loss backward into \(h_t\)

We now need:

\[
\frac{\partial L_t}{\partial h_t}.
\]

From:

\[
z_t=W_{hy}h_t+b_y,
\]

at scalar level:

\[
\frac{\partial z_{t,k}}{\partial h_{t,j}}
=
W_{hy,kj}.
\]

Therefore:

\[
\frac{\partial L_t}{\partial h_{t,j}}
=
\sum_k
\frac{\partial L_t}{\partial z_{t,k}}
\frac{\partial z_{t,k}}{\partial h_{t,j}}.
\]

Substitute:

\[
\frac{\partial L_t}{\partial h_{t,j}}
=
\sum_k
\delta_{z,t,k}W_{hy,kj}.
\]

In vector form:

\[
\boxed{
\frac{\partial L_t}{\partial h_t}
=
W_{hy}^T\delta_{z,t}.
}
\tag{19.1}
\]

This is only the **current-output contribution**.

It is not yet the full:

\[
\frac{\partial L}{\partial h_t}
\]

because future losses also depend on \(h_t\).

---

# 20. Why the gradient into \(h_t\) has two sources

This is the central conceptual step.

The hidden state \(h_t\) is used in:

\[
z_t=W_{hy}h_t+b_y
\]

and:

\[
a_{t+1}
=
W_{xh}x_{t+1}
+
W_{hh}h_t
+
b_h.
\]

Therefore, the total sequence loss depends on \(h_t\) through two branches:

```text
                         h_t
                       /     \
                      /       \
                     ▼         ▼
                  z_t       a_{t+1}
                   │           │
                   ▼           ▼
                  L_t        h_{t+1}
                                 │
                                 ▼
                              L_{t+1}
                                 │
                                ...
                                 │
                                 ▼
                               L_T
```

The multivariable chain rule therefore gives:

\[
\boxed{
\frac{\partial L}{\partial h_t}
=
\frac{\partial L_t}{\partial h_t}
+
\frac{\partial L_{t+1:T}}{\partial h_t}.
}
\tag{20.1}
\]

The first term is:

\[
W_{hy}^T\delta_{z,t}.
\]

The second term is the recurrent contribution.

---

# 21. Derive the future contribution

The future loss reaches \(h_t\) first through \(a_{t+1}\):

\[
h_t\rightarrow a_{t+1}.
\]

Recall:

\[
a_{t+1}
=
W_{xh}x_{t+1}
+
W_{hh}h_t
+
b_h.
\]

At scalar level:

\[
a_{t+1,i}
=
\sum_jW_{hh,ij}h_{t,j}
+\text{terms independent of }h_t.
\]

Therefore:

\[
\frac{\partial a_{t+1,i}}
{\partial h_{t,j}}
=
W_{hh,ij}.
\tag{21.1}
\]

Define:

\[
\boxed{
\delta_{t+1}
\equiv
\frac{\partial L}{\partial a_{t+1}}.
}
\tag{21.2}
\]

Then:

\[
\frac{\partial L_{t+1:T}}
{\partial h_{t,j}}
=
\sum_i
\frac{\partial L_{t+1:T}}
{\partial a_{t+1,i}}
\frac{\partial a_{t+1,i}}
{\partial h_{t,j}}.
\]

So:

\[
=
\sum_i
\delta_{t+1,i}W_{hh,ij}.
\]

In vector notation:

\[
\boxed{
\frac{\partial L_{t+1:T}}
{\partial h_t}
=
W_{hh}^T\delta_{t+1}.
}
\tag{21.3}
\]

This is the quantity carried backward through time.

---

# 22. The complete hidden-state gradient

Combine the current-output contribution and future contribution:

\[
\boxed{
\frac{\partial L}{\partial h_t}
=
W_{hy}^T\delta_{z,t}
+
W_{hh}^T\delta_{t+1}.
}
\tag{22.1}
\]

This is the central BPTT equation.

Notice what it says:

- \(W_{hy}^T\delta_{z,t}\) tells us how the current prediction contributes to the hidden state gradient.
- \(W_{hh}^T\delta_{t+1}\) tells us how future time steps contribute.

The two must be added because both are valid paths from \(h_t\) to the total loss.

---

# 23. Boundary condition at the final time step

At \(t=T\), there is no future output.

Therefore:

\[
\boxed{
\delta_{T+1}=0.
}
\tag{23.1}
\]

So:

\[
\boxed{
\frac{\partial L}{\partial h_T}
=
W_{hy}^T\delta_{z,T}.
}
\tag{23.2}
\]

Then we use that to derive \(\delta_T\).

This gives a natural starting point for the backward pass:

\[
T,T-1,\ldots,1.
\]

---

# 24. Derivative of \(\tanh\)

We have:

\[
h_t=\tanh(a_t).
\]

The scalar derivative is:

\[
\boxed{
\frac{d}{da}\tanh(a)
=
1-\tanh^2(a).
}
\tag{24.1}
\]

Because:

\[
h_t=\tanh(a_t),
\]

we can write:

\[
\boxed{
\frac{\partial h_{t,i}}
{\partial a_{t,i}}
=
1-h_{t,i}^2.
}
\tag{24.2}
\]

The derivative is elementwise, so:

\[
\boxed{
\frac{\partial L}{\partial a_t}
=
\frac{\partial L}{\partial h_t}
\odot
(1-h_t\odot h_t).
}
\tag{24.3}
\]

The symbol:

\[
\odot
\]

means elementwise multiplication.

---

# 25. Introducing the hidden recurrent error signal

Define:

\[
\boxed{
\delta_t
\equiv
\frac{\partial L}{\partial a_t}.
}
\tag{25.1}
\]

Then:

\[
\boxed{
\delta_t
=
\left(
W_{hy}^T\delta_{z,t}
+
W_{hh}^T\delta_{t+1}
\right)
\odot
(1-h_t\odot h_t).
}
\tag{25.2}
\]

This is the central BPTT recurrence.

At the final time:

\[
\boxed{
\delta_T
=
\left(
W_{hy}^T\delta_{z,T}
\right)
\odot
(1-h_T\odot h_T).
}
\tag{25.3}
\]

Then:

\[
\delta_T
\rightarrow
\delta_{T-1}
\rightarrow
\cdots
\rightarrow
\delta_1.
\]

---

# 26. Scalar derivation of the BPTT recurrence

For hidden unit \(i\):

\[
\delta_{t,i}
=
\frac{\partial L}{\partial a_{t,i}}.
\]

The chain rule gives:

\[
\frac{\partial L}{\partial a_{t,i}}
=
\frac{\partial L}{\partial h_{t,i}}
\frac{\partial h_{t,i}}
{\partial a_{t,i}}.
\]

We know:

\[
\frac{\partial h_{t,i}}
{\partial a_{t,i}}
=
1-h_{t,i}^2.
\]

Now:

\[
\frac{\partial L}{\partial h_{t,i}}
=
\frac{\partial L_t}{\partial h_{t,i}}
+
\frac{\partial L_{t+1:T}}{\partial h_{t,i}}.
\]

The current-output contribution is:

\[
\frac{\partial L_t}{\partial h_{t,i}}
=
\sum_k
\delta_{z,t,k}W_{hy,ki}.
\]

The future contribution is:

\[
\frac{\partial L_{t+1:T}}{\partial h_{t,i}}
=
\sum_j
\delta_{t+1,j}W_{hh,ji}.
\]

Therefore:

\[
\boxed{
\delta_{t,i}
=
\left[
\sum_kW_{hy,ki}\delta_{z,t,k}
+
\sum_jW_{hh,ji}\delta_{t+1,j}
\right]
(1-h_{t,i}^2).
}
\tag{26.1}
\]

This is the scalar equation underlying the vector formula.

---

# 27. Why \(W_{hh}^T\) appears

Forward recurrence:

\[
a_{t+1}=W_{hh}h_t+\cdots
\]

The matrix \(W_{hh}\) maps:

\[
H\rightarrow H.
\]

Backward, we need a mapping from the gradient on \(a_{t+1}\) back to the gradient on \(h_t\).

At scalar level:

\[
\frac{\partial L}{\partial h_{t,j}}
=
\sum_i
\delta_{t+1,i}W_{hh,ij}.
\]

This is exactly:

\[
\boxed{
W_{hh}^T\delta_{t+1}.
}
\tag{27.1}
\]

The transpose is not a convention chosen because "backward uses transpose." It is the matrix form of the scalar summation.

---

# 28. NOW the hidden error gives the hidden parameter gradients

We now have:

\[
\delta_t
=
\frac{\partial L}{\partial a_t}.
\]

And the hidden parameters directly create:

\[
a_t=W_{xh}x_t+W_{hh}h_{t-1}+b_h.
\]

So \(\delta_t\) gives us the local parameter gradients.

---

# 29. Gradient of \(W_{xh}\)

At scalar level:

\[
a_{t,i}
=
\sum_jW_{xh,ij}x_{t,j}
+\cdots
\]

Therefore:

\[
\frac{\partial a_{t,i}}
{\partial W_{xh,ij}}
=
x_{t,j}.
\]

By the chain rule:

\[
\frac{\partial L}
{\partial W_{xh,ij}}
\Bigg|_t
=
\frac{\partial L}{\partial a_{t,i}}
\frac{\partial a_{t,i}}
{\partial W_{xh,ij}}.
\]

Hence:

\[
\boxed{
\frac{\partial L}
{\partial W_{xh,ij}}
\Bigg|_t
=
\delta_{t,i}x_{t,j}.
}
\tag{29.1}
\]

In matrix form:

\[
\boxed{
\frac{\partial L}{\partial W_{xh}}
\Bigg|_t
=
\delta_t x_t^T.
}
\tag{29.2}
\]

Dimensions:

\[
(H\times1)(1\times D)=H\times D.
\]

---

# 30. Gradient of \(W_{hh}\)

At scalar level:

\[
a_{t,i}
=
\sum_jW_{hh,ij}h_{t-1,j}
+\cdots
\]

Therefore:

\[
\frac{\partial a_{t,i}}
{\partial W_{hh,ij}}
=
h_{t-1,j}.
\]

Then:

\[
\frac{\partial L}
{\partial W_{hh,ij}}
\Bigg|_t
=
\delta_{t,i}h_{t-1,j}.
\]

So:

\[
\boxed{
\frac{\partial L}{\partial W_{hh}}
\Bigg|_t
=
\delta_t h_{t-1}^T.
}
\tag{30.1}
\]

Dimensions:

\[
(H\times1)(1\times H)=H\times H.
\]

---

# 31. Gradient of \(b_h\)

The hidden bias enters additively:

\[
a_t=\cdots+b_h.
\]

For each component:

\[
\frac{\partial a_{t,i}}
{\partial b_{h,i}}
=
1.
\]

Therefore:

\[
\boxed{
\frac{\partial L}{\partial b_h}
\Bigg|_t
=
\delta_t.
}
\tag{31.1}
\]

---

# 32. Why parameter gradients are summed across time

The same parameters are reused at every time step.

For example:

\[
a_1=W_{xh}x_1+W_{hh}h_0+b_h
\]

\[
a_2=W_{xh}x_2+W_{hh}h_1+b_h
\]

\[
a_3=W_{xh}x_3+W_{hh}h_2+b_h.
\]

There is only one:

\[
W_{hh}.
\]

Therefore the total loss depends on that same parameter through many paths.

By the multivariable chain rule, those contributions add:

\[
\boxed{
\frac{\partial L}{\partial W_{hh}}
=
\sum_{t=1}^{T}
\delta_t h_{t-1}^T.
}
\tag{32.1}
\]

Similarly:

\[
\boxed{
\frac{\partial L}{\partial W_{xh}}
=
\sum_{t=1}^{T}
\delta_t x_t^T.
}
\tag{32.2}
\]

and:

\[
\boxed{
\frac{\partial L}{\partial b_h}
=
\sum_{t=1}^{T}
\delta_t.
}
\tag{32.3}
\]

For the output layer:

\[
\boxed{
\frac{\partial L}{\partial W_{hy}}
=
\sum_{t=1}^{T}
\delta_{z,t}h_t^T.
}
\tag{32.4}
\]

and:

\[
\boxed{
\frac{\partial L}{\partial b_y}
=
\sum_{t=1}^{T}
\delta_{z,t}.
}
\tag{32.5}
\]

---

# 33. The complete BPTT derivation

We can now put the entire backward pass in one sequence.

Set:

\[
\boxed{
\delta_{T+1}=0.
}
\tag{33.1}
\]

For:

\[
t=T,T-1,\ldots,1:
\]

### Step 1 — output error

\[
\boxed{
\delta_{z,t}=\hat y_t-y_t.
}
\tag{33.2}
\]

Why?

Because softmax plus cross-entropy reduces to this expression.

### Step 2 — current-output gradient into hidden state

\[
W_{hy}^T\delta_{z,t}.
\]

### Step 3 — future gradient into hidden state

\[
W_{hh}^T\delta_{t+1}.
\]

### Step 4 — add the two paths

\[
\boxed{
dh_t
=
W_{hy}^T\delta_{z,t}
+
W_{hh}^T\delta_{t+1}.
}
\tag{33.3}
\]

Here:

\[
dh_t
\equiv
\frac{\partial L}{\partial h_t}.
\]

### Step 5 — cross tanh

\[
\boxed{
\delta_t
=
dh_t
\odot
(1-h_t\odot h_t).
}
\tag{33.4}
\]

### Step 6 — accumulate output gradients

\[
dW_{hy}\mathrel{+}=\delta_{z,t}h_t^T
\tag{33.5}
\]

\[
db_y\mathrel{+}=\delta_{z,t}.
\tag{33.6}
\]

### Step 7 — accumulate hidden gradients

\[
dW_{xh}\mathrel{+}=\delta_t x_t^T
\tag{33.7}
\]

\[
dW_{hh}\mathrel{+}=\delta_t h_{t-1}^T
\tag{33.8}
\]

\[
db_h\mathrel{+}=\delta_t.
\tag{33.9}
\]

### Step 8 — carry the recurrent error backward

\[
\boxed{
\delta_{t}\text{ becomes the next-step recurrent gradient.}
}
\tag{33.10}
\]

Then continue to \(t-1\).

---

# 34. The exact dependency chain during BPTT

At each time step:

\[
L_t
\rightarrow
\delta_{z,t}
\]

then:

\[
\delta_{z,t}
\rightarrow
\left\{
dW_{hy},db_y,
W_{hy}^T\delta_{z,t}
\right\}.
\]

The hidden state also receives:

\[
W_{hh}^T\delta_{t+1}.
\]

So:

\[
\boxed{
dh_t
=
W_{hy}^T\delta_{z,t}
+
W_{hh}^T\delta_{t+1}.
}
\]

Then:

\[
dh_t
\rightarrow
\delta_t
\]

through:

\[
\delta_t
=
dh_t\odot(1-h_t^2).
\]

Then:

\[
\delta_t
\rightarrow
\left\{
dW_{xh},
dW_{hh},
db_h,
\text{gradient passed to }h_{t-1}
\right\}.
\]

That last branch is what makes the computation recursive through time.

---

# 35. Why \(dW_{hh}\) is summed even when the future gradient is already included in \(\delta_t\)

This is an easy point to misunderstand.

The value:

\[
\delta_t
=
\frac{\partial L}{\partial a_t}
\]

already contains information from:

- \(L_t\),
- \(L_{t+1}\),
- \(\ldots\),
- \(L_T\).

Therefore the single local quantity:

\[
\delta_t h_{t-1}^T
\]

is the total contribution of time \(t\)'s use of \(W_{hh}\) to the full sequence loss.

But \(W_{hh}\) is used again at time \(t-1\), \(t-2\), etc.

So we still need:

\[
\sum_t\delta_t h_{t-1}^T.
\]

There are two different notions of accumulation:

1. **within \(\delta_t\):** future losses are accumulated through the recurrent path,
2. **across the parameter gradient:** every use of the shared parameter is accumulated across time.

---

# 36. A scalar derivation of the total \(W_{hh}\) gradient

Take one parameter:

\[
W_{hh,ij}.
\]

The total loss is:

\[
L=\sum_tL_t.
\]

Therefore:

\[
\frac{\partial L}{\partial W_{hh,ij}}
=
\sum_t
\frac{\partial L}{\partial W_{hh,ij}}
\Bigg|_t.
\]

At time \(t\):

\[
\frac{\partial L}{\partial W_{hh,ij}}
\Bigg|_t
=
\frac{\partial L}{\partial a_{t,i}}
\frac{\partial a_{t,i}}
{\partial W_{hh,ij}}.
\]

The first factor is:

\[
\delta_{t,i}.
\]

The second is:

\[
h_{t-1,j}.
\]

Therefore:

\[
\boxed{
\frac{\partial L}{\partial W_{hh,ij}}
=
\sum_{t=1}^{T}
\delta_{t,i}h_{t-1,j}.
}
\tag{36.1}
\]

The matrix equation:

\[
dW_{hh}
=
\sum_t\delta_t h_{t-1}^T
\]

is simply the compact form of this scalar equation.

---

# 37. Scalar interpretation of every final gradient

Matrices can hide the meaning. Pick one parameter at a time.

## 37.1 One output-layer weight

Take:

\[
W_{hy,kj}.
\]

It connects hidden unit \(j\) to output logit \(k\).

Its gradient is:

\[
\boxed{
\frac{\partial L}{\partial W_{hy,kj}}
=
\sum_t
\underbrace{\delta_{z,t,k}}_{\text{output error at time }t}
\underbrace{h_{t,j}}_{\text{hidden activation at time }t}.
}
\tag{37.1}
\]

So:

> output error × value entering the weight, summed over all uses of the weight.

---

## 37.2 One input-to-hidden weight

Take:

\[
W_{xh,ij}.
\]

It connects input component \(j\) to hidden neuron \(i\).

Its gradient is:

\[
\boxed{
\frac{\partial L}{\partial W_{xh,ij}}
=
\sum_t
\delta_{t,i}x_{t,j}.
}
\tag{37.2}
\]

---

## 37.3 One recurrent weight

Take:

\[
W_{hh,ij}.
\]

It connects hidden unit \(j\) at time \(t-1\) to hidden unit \(i\) at time \(t\).

Its gradient is:

\[
\boxed{
\frac{\partial L}{\partial W_{hh,ij}}
=
\sum_t
\delta_{t,i}h_{t-1,j}.
}
\tag{37.3}
\]

This is the defining recurrent parameter gradient.

---

## 37.4 One hidden bias

\[
\boxed{
\frac{\partial L}{\partial b_{h,i}}
=
\sum_t\delta_{t,i}.
}
\tag{37.4}
\]

---

## 37.5 One output bias

\[
\boxed{
\frac{\partial L}{\partial b_{y,k}}
=
\sum_t\delta_{z,t,k}.
}
\tag{37.5}
\]

---

# 38. The general affine-layer pattern appears again

For an affine transformation:

\[
z=Wx+b,
\]

if:

\[
\delta=\frac{\partial L}{\partial z},
\]

then:

\[
\boxed{
\frac{\partial L}{\partial W}
=
\delta x^T
}
\tag{38.1}
\]

and:

\[
\boxed{
\frac{\partial L}{\partial b}
=
\delta.
}
\tag{38.2}
\]

An RNN simply applies this pattern at every time step.

For the input-to-hidden affine transformation:

\[
x=x_t.
\]

For the recurrent affine transformation:

\[
x=h_{t-1}.
\]

For the output affine transformation:

\[
x=h_t.
\]

The only extra complication is that:

\[
W_{hh}
\]

and the other parameters are shared across time.

---

# 39. Why outer products appear

For example:

\[
\delta_t\in\mathbb R^H
\]

and:

\[
x_t\in\mathbb R^D.
\]

We want a matrix of shape:

\[
H\times D.
\]

Therefore:

\[
\boxed{
\delta_t x_t^T
}
\]

has exactly the right dimensions:

\[
(H\times1)(1\times D)=H\times D.
\]

Similarly:

\[
\delta_t h_{t-1}^T
\]

has shape:

\[
(H\times1)(1\times H)=H\times H.
\]

And:

\[
\delta_{z,t}h_t^T
\]

has shape:

\[
(C\times1)(1\times H)=C\times H.
\]

The matrix formulas are simply compact versions of the scalar formulas.

---

# 40. Dimension checks — a powerful debugging tool

Dimension checks can catch many implementation errors before you inspect any numerical value.

## 40.1 Forward

\[
W_{xh}x_t:
\quad
(H\times D)(D\times1)
=
H\times1.
\]

\[
W_{hh}h_{t-1}:
\quad
(H\times H)(H\times1)
=
H\times1.
\]

Therefore:

\[
a_t\in\mathbb R^H.
\]

Then:

\[
W_{hy}h_t:
\quad
(C\times H)(H\times1)
=
C\times1.
\]

Therefore:

\[
z_t\in\mathbb R^C.
\]

---

## 40.2 Backward

\[
W_{hy}^T\delta_{z,t}:
\quad
(H\times C)(C\times1)
=
H\times1.
\]

\[
W_{hh}^T\delta_{t+1}:
\quad
(H\times H)(H\times1)
=
H\times1.
\]

Therefore:

\[
dh_t\in\mathbb R^H.
\]

Then:

\[
\delta_t\in\mathbb R^H.
\]

---

## 40.3 Parameter gradients

\[
\delta_t x_t^T:
\quad
(H\times1)(1\times D)
=
H\times D.
\]

\[
\delta_t h_{t-1}^T:
\quad
(H\times1)(1\times H)
=
H\times H.
\]

\[
\delta_{z,t}h_t^T:
\quad
(C\times1)(1\times H)
=
C\times H.
\]

Each matches the corresponding parameter matrix.

---

# 41. The complete dependency tree

This is the main reference diagram.

```text
                         FORWARD THROUGH TIME

x_t ───────────────┐
                   │
h_{t-1} ───────────┤
                   ▼
        a_t = Wxh x_t + Whh h_{t-1} + bh
                   │
                 tanh
                   │
                   ▼
                  h_t
                /     \
               /       \
              ▼         ▼
       z_t = Why h_t    h_t → a_{t+1}
             + by
              │
           softmax
              │
              ▼
             ŷ_t
              │
         cross-entropy
              │
              ▼
             L_t


                         BACKWARD

L_t
│
▼
δz_t = ŷ_t - y_t
│
├──────────────────────► dWhy += δz_t h_tᵀ
│
├──────────────────────► dby  += δz_t
│
▼
Whyᵀ δz_t
│
├──────────────────────┐
│                      │
│                      ▼
│               Whhᵀ δ_{t+1}
│                      │
└──────────────┬───────┘
               ▼
        dh_t = ∂L/∂h_t
               │
               ▼
       δ_t = dh_t ⊙ (1-h_t²)
               │
       ┌───────┼─────────────┐
       ▼       ▼             ▼
dWxh += δ_t x_tᵀ
dWhh += δ_t h_{t-1}ᵀ
dbh  += δ_t
       │
       ▼
 carry δ_t to previous time step
```

---

# 42. The final equations only

## Forward pass

\[
\boxed{
a_t
=
W_{xh}x_t
+
W_{hh}h_{t-1}
+
b_h
}
\tag{42.1}
\]

\[
\boxed{
h_t=\tanh(a_t)
}
\tag{42.2}
\]

\[
\boxed{
z_t=W_{hy}h_t+b_y
}
\tag{42.3}
\]

\[
\boxed{
\hat y_t=\operatorname{softmax}(z_t)
}
\tag{42.4}
\]

\[
\boxed{
L_t=-\sum_k y_{t,k}\log\hat y_{t,k}
}
\tag{42.5}
\]

\[
\boxed{
L=\sum_tL_t
}
\tag{42.6}
\]

---

## Backward pass

Initialize:

\[
\boxed{
\delta_{T+1}=0.
}
\tag{42.7}
\]

Then for:

\[
t=T,T-1,\ldots,1:
\]

\[
\boxed{
\delta_{z,t}
=
\hat y_t-y_t
}
\tag{42.8}
\]

\[
\boxed{
dh_t
=
W_{hy}^T\delta_{z,t}
+
W_{hh}^T\delta_{t+1}
}
\tag{42.9}
\]

\[
\boxed{
\delta_t
=
dh_t\odot(1-h_t\odot h_t)
}
\tag{42.10}
\]

Accumulate:

\[
\boxed{
dW_{hy}\mathrel{+}=
\delta_{z,t}h_t^T
}
\tag{42.11}
\]

\[
\boxed{
db_y\mathrel{+}=
\delta_{z,t}
}
\tag{42.12}
\]

\[
\boxed{
dW_{xh}\mathrel{+}=
\delta_t x_t^T
}
\tag{42.13}
\]

\[
\boxed{
dW_{hh}\mathrel{+}=
\delta_t h_{t-1}^T
}
\tag{42.14}
\]

\[
\boxed{
db_h\mathrel{+}=
\delta_t
}
\tag{42.15}
\]

Then move to \(t-1\).

---

# 43. Why \(\delta_t\) must be carried backward

The quantity:

\[
\delta_t
=
\frac{\partial L}{\partial a_t}
\]

contains the effect of the current time step **and all future time steps**.

The recurrent path is:

\[
a_t
\rightarrow
h_t
\rightarrow
a_{t+1}.
\]

So \(\delta_t\) influences the previous hidden state through:

\[
\frac{\partial L}{\partial h_{t-1}}
=
W_{hh}^T\delta_t.
\tag{43.1}
\]

When processing time \(t-1\), that becomes part of:

\[
\frac{\partial L}{\partial h_{t-1}}.
\]

Therefore the backward pass naturally carries a recurrent gradient variable.

In implementation, a common name is:

```text
dh_next
```

or:

```text
delta_next
```

depending on exactly which quantity you choose to carry.

Be explicit about whether your variable represents:

\[
\frac{\partial L}{\partial h_t}
\]

or:

\[
\frac{\partial L}{\partial a_t}.
\]

Mixing them is a common source of bugs.

---

# 44. A clean implementation convention

For a straightforward C implementation, the clearest convention is:

```text
dh_next = gradient coming from the future into the current hidden state
```

At time \(t\), calculate:

```text
dh = Whyᵀ * dz + dh_next
da = dh * (1 - h*h)
```

Then:

```text
dh_next = Whhᵀ * da
```

This is equivalent to the mathematical recurrence, but it avoids confusing:

\[
\frac{\partial L}{\partial h_t}
\]

with:

\[
\frac{\partial L}{\partial a_t}.
\]

Under this convention:

\[
\boxed{
dh_t
=
W_{hy}^T\delta_{z,t}
+
dh_{\text{next}}
}
\tag{44.1}
\]

and:

\[
\boxed{
dh_{\text{next}}
=
W_{hh}^T\delta_t.
}
\tag{44.2}
\]

This is the implementation-friendly form of the same mathematics.

---

# 45. Why the two backward equations are equivalent

The compact recurrence says:

\[
\frac{\partial L}{\partial h_t}
=
W_{hy}^T\delta_{z,t}
+
W_{hh}^T\delta_{t+1}.
\]

If the implementation stores:

\[
dh_{\text{next}}
=
W_{hh}^T\delta_{t+1},
\]

then:

\[
dh_t
=
W_{hy}^T\delta_{z,t}
+
dh_{\text{next}}.
\]

After tanh:

\[
\delta_t
=
dh_t\odot(1-h_t^2).
\]

Then the gradient passed to the previous hidden state is:

\[
dh_{\text{next}}
\leftarrow
W_{hh}^T\delta_t.
\]

So there are two equally valid ways to organize the algebra:

### Mathematical notation

\[
\boxed{
dh_t=
W_{hy}^T\delta_{z,t}
+
W_{hh}^T\delta_{t+1}.
}
\]

### Implementation notation

\[
\boxed{
dh=
W_{hy}^Tdz+dh_{\text{next}}
}
\]

then:

\[
\boxed{
dh_{\text{next}}=W_{hh}^Tda.
}
\]

The second is often easier to code.

---

# 46. A small scalar example

Take one hidden neuron:

\[
H=1.
\]

Suppose:

\[
x_1=1,\qquad x_2=2,
\]

\[
h_0=0,
\]

\[
w_{xh}=0.5,\qquad
w_{hh}=0.8,\qquad
b_h=0.
\]

At time 1:

\[
a_1
=
0.5(1)+0.8(0)
=
0.5.
\]

Therefore:

\[
h_1=\tanh(0.5)\approx0.4621.
\]

At time 2:

\[
a_2
=
0.5(2)
+
0.8(0.4621)
\approx1.3697.
\]

Therefore:

\[
h_2
=
\tanh(1.3697)
\approx0.8783.
\]

Notice:

\[
h_2
\]

depends on both:

\[
x_2
\]

and:

\[
h_1.
\]

So information from the earlier input can influence the later state.

---

# 47. Scalar derivative example for tanh

Suppose:

\[
\frac{\partial L}{\partial h_2}=0.7.
\]

Since:

\[
h_2\approx0.8783,
\]

the local derivative is:

\[
1-h_2^2
\approx
1-(0.8783)^2
\approx0.2286.
\]

Therefore:

\[
\delta_2
=
0.7(0.2286)
\approx0.1600.
\]

This illustrates the role of the activation derivative:

\[
\boxed{
\frac{\partial L}{\partial a_2}
=
\frac{\partial L}{\partial h_2}
\frac{\partial h_2}{\partial a_2}.
}
\]

---

# 48. Many-to-many and many-to-one

The same recurrence can support different output structures.

## 48.1 Many-to-many

Every hidden state has an output:

\[
h_t\rightarrow z_t\rightarrow\hat y_t.
\]

The total loss is:

\[
L=\sum_tL_t.
\]

This is the architecture used in the main derivation.

Examples include:

- sequence labeling,
- next-character prediction,
- next-token prediction.

---

## 48.2 Many-to-one

Only the final state produces an output:

\[
h_T\rightarrow z\rightarrow\hat y.
\]

Then:

\[
L=L_T.
\]

The final output gradient is still:

\[
\delta_z=\hat y-y.
\]

But earlier time steps have no direct output loss.

Their gradients are generated entirely by the recurrent path from the final hidden state.

For example:

\[
\delta_T
=
(W_{hy}^T\delta_z)
\odot
(1-h_T^2),
\]

then:

\[
\delta_{T-1}
=
(W_{hh}^T\delta_T)
\odot
(1-h_{T-1}^2),
\]

and so on.

So BPTT still applies.

---

# 49. Why BPTT is just ordinary backpropagation on an unrolled network

Once we unroll the RNN over \(T\) steps, the graph becomes a deep feed-forward graph:

\[
x_1\rightarrow h_1\rightarrow h_2\rightarrow\cdots\rightarrow h_T.
\]

The important difference is:

\[
W_{hh}
\]

is reused at every step.

Therefore BPTT consists of:

1. unroll the recurrent computation,
2. apply ordinary chain-rule backpropagation,
3. sum gradients from every use of the shared parameters.

This viewpoint is extremely useful.

The RNN is not using a new form of calculus.

It is using ordinary calculus on a repeated computation graph.

---

# 50. Why gradients from future time steps reach earlier states

Consider:

\[
x_1
\rightarrow
h_1
\rightarrow
h_2
\rightarrow
h_3
\rightarrow
L_3.
\]

Then:

\[
\frac{\partial L_3}{\partial x_1}
=
\frac{\partial L_3}{\partial h_3}
\frac{\partial h_3}{\partial h_2}
\frac{\partial h_2}{\partial h_1}
\frac{\partial h_1}{\partial x_1}.
\]

Therefore the loss at a later time can influence parameters used much earlier.

This is the mechanism through which the network can learn temporal dependencies.

---

# 51. The Jacobian of the recurrent state transition

The recurrent transition is:

\[
h_t
=
\tanh(
W_{xh}x_t
+
W_{hh}h_{t-1}
+
b_h
).
\]

Define:

\[
D_t
=
\operatorname{diag}
(1-h_{t,1}^2,\ldots,1-h_{t,H}^2).
\]

Then:

\[
\boxed{
\frac{\partial h_t}{\partial h_{t-1}}
=
D_tW_{hh}.
}
\tag{51.1}
\]

This is the one-step recurrent Jacobian.

It is the mathematical object that is repeatedly multiplied during long-range gradient propagation.

---

# 52. Repeated Jacobians and vanishing/exploding gradients

To propagate a gradient from time \(T\) back to time \(k\), we obtain a product:

\[
\boxed{
\frac{\partial h_T}{\partial h_k}
=
(D_TW_{hh})
(D_{T-1}W_{hh})
\cdots
(D_{k+1}W_{hh}).
}
\tag{52.1}
\]

The number of factors grows with the temporal distance.

If the repeated transformations tend to shrink the gradient, we get:

\[
\boxed{\text{vanishing gradients}.}
\]

If they tend to enlarge it, we get:

\[
\boxed{\text{exploding gradients}.}
\]

This follows directly from the chain rule.

---

# 53. Why \(W_{hh}\) is special

Compare:

\[
W_{xh}
\]

and:

\[
W_{hh}.
\]

\(W_{xh}\) maps current input into hidden space.

But:

\[
W_{hh}
\]

is applied again and again:

\[
h_{t-1}\rightarrow h_t,
\]

\[
h_t\rightarrow h_{t+1},
\]

\[
h_{t+1}\rightarrow h_{t+2},
\]

and so on.

Therefore:

\[
W_{hh}
\]

controls the propagation of both information and gradients through time.

---

# 54. Why tanh can contribute to vanishing gradients

For:

\[
h=\tanh(a),
\]

we have:

\[
\tanh'(a)=1-h^2.
\]

This quantity is at most 1.

When \(|a|\) becomes large, tanh saturates:

\[
h\approx1
\]

or:

\[
h\approx-1.
\]

Then:

\[
1-h^2\approx0.
\]

So the local derivative becomes small.

Repeated multiplication by small derivatives can cause gradients to shrink rapidly.

---

# 55. Gradient clipping

When gradients become very large, training can become unstable.

A common solution is gradient clipping.

Let all gradient entries across all parameters be collected conceptually into one vector \(g\).

Its norm is:

\[
\boxed{
\|g\|
=
\sqrt{
\sum_i g_i^2
}.
}
\tag{55.1}
\]

Choose a threshold \(c\).

If:

\[
\|g\|>c,
\]

rescale:

\[
\boxed{
g_i
\leftarrow
g_i
\frac{c}{\|g\|}.
}
\tag{55.2}
\]

This keeps the direction and limits the magnitude.

For an educational implementation, you can implement global-norm clipping across:

\[
dW_{xh},dW_{hh},db_h,dW_{hy},db_y.
\]

---

# 56. Stable softmax

The mathematical formula is:

\[
\hat y_{t,k}
=
\frac{e^{z_{t,k}}}
{\sum_j e^{z_{t,j}}}.
\]

But exponentials can overflow numerically.

Define:

\[
m_t=\max_jz_{t,j}.
\]

Then compute:

\[
\boxed{
\hat y_{t,k}
=
\frac{e^{z_{t,k}-m_t}}
{\sum_j e^{z_{t,j}-m_t}}.
}
\tag{56.1}
\]

Subtracting the same value from every logit does not change the softmax probabilities.

This is the form you should implement.

---

# 57. Stable cross-entropy via log-sum-exp

For a one-hot target with correct class \(c\):

\[
L_t=-\log\hat y_{t,c}.
\]

Using softmax:

\[
L_t
=
-z_{t,c}
+
\log\sum_j e^{z_{t,j}}.
\tag{57.1}
\]

To compute this stably, let:

\[
m_t=\max_jz_{t,j}.
\]

Then:

\[
\boxed{
L_t
=
-z_{t,c}
+
m_t
+
\log
\sum_j
e^{z_{t,j}-m_t}.
}
\tag{57.2}
\]

This avoids overflowing exponentials.

---

# 58. Parameter count

The number of trainable parameters is:

### Input-to-hidden

\[
HD.
\]

### Hidden-to-hidden

\[
H^2.
\]

### Hidden-to-output

\[
CH.
\]

### Hidden bias

\[
H.
\]

### Output bias

\[
C.
\]

Therefore:

\[
\boxed{
N_{\text{params}}
=
HD+H^2+CH+H+C.
}
\tag{58.1}
\]

Notice that the number of parameters does **not** depend on \(T\).

The sequence length affects computation and activation memory, but not the number of shared parameters.

---

# 59. Computational complexity

At each time step:

\[
W_{xh}x_t
\]

costs approximately:

\[
O(HD).
\]

The recurrent multiplication:

\[
W_{hh}h_{t-1}
\]

costs approximately:

\[
O(H^2).
\]

The output multiplication:

\[
W_{hy}h_t
\]

costs approximately:

\[
O(CH).
\]

Across \(T\) time steps:

\[
\boxed{
O\left(
T(HD+H^2+CH)
\right)
}
\tag{59.1}
\]

for the forward pass, with BPTT having the same broad order.

The recurrent term:

\[
TH^2
\]

often dominates when \(H\) is large.

---

# 60. What must be stored for BPTT

During the forward pass, the backward pass later needs the values that appear in local derivatives and parameter gradients.

At minimum, a straightforward implementation stores:

\[
h_0,h_1,\ldots,h_T.
\]

Why?

Because:

\[
dW_{hh}
\]

needs:

\[
h_{t-1},
\]

and the tanh derivative needs:

\[
h_t.
\]

You may also store logits or probabilities for each time step so that:

\[
\delta_{z,t}
=
\hat y_t-y_t
\]

can be computed directly during the backward pass.

A simple implementation therefore stores:

```text
x[t]          input
h[t]          hidden state before processing x[t]
h[t+1]        hidden state after processing x[t]
logits[t]     output logits
probs[t]      output probabilities
target[t]     target
```

---

# 61. A clean zero-based C indexing convention

C arrays are naturally zero-based.

A convenient convention is:

```text
x[0]     = x_1
x[1]     = x_2
...
x[T-1]   = x_T
```

and:

```text
h[0]     = h_0
h[1]     = h_1
...
h[T]     = h_T
```

Then:

\[
\boxed{
h[t+1]
=
\tanh(
W_{xh}x[t]
+
W_{hh}h[t]
+
b_h
)
}
\tag{61.1}
\]

This has a nice property:

> `h[t]` is exactly the previous hidden state used to compute `h[t+1]`.

That makes the backward indexing straightforward.

---

# 62. Forward pass in implementation order

For:

\[
t=0,\ldots,T-1:
\]

compute:

\[
a=W_{xh}x[t]+W_{hh}h[t]+b_h
\]

then:

\[
h[t+1]=\tanh(a)
\]

then:

\[
z=W_{hy}h[t+1]+b_y
\]

then:

\[
\hat y=\operatorname{softmax}(z).
\]

Then compute:

\[
L_t.
\]

The important storage relationship is:

\[
\boxed{
h[t+1]\text{ is the current hidden state, while }h[t]\text{ is the previous state.}
}
\tag{62.1}
\]

---

# 63. Backward pass in implementation order

Initialize:

```text
dh_next = 0
```

Then:

```text
for t = T-1 down to 0
```

compute:

\[
dz=\hat y[t]-y[t].
\]

Then:

\[
dh
=
W_{hy}^Tdz
+
dh_{\text{next}}.
\]

Then:

\[
da
=
dh\odot(1-h[t+1]^2).
\]

Then accumulate:

\[
dW_{hy}
\mathrel{+}=
dz\,h[t+1]^T
\]

\[
db_y
\mathrel{+}=
dz
\]

\[
dW_{xh}
\mathrel{+}=
da\,x[t]^T
\]

\[
dW_{hh}
\mathrel{+}=
da\,h[t]^T
\]

\[
db_h
\mathrel{+}=
da.
\]

Finally:

\[
dh_{\text{next}}
=
W_{hh}^Tda.
\]

Then decrement \(t\).

---

# 64. Why `dh_next` is initialized to zero

At the final time step:

\[
T
\]

there is no future hidden state.

Therefore there is no gradient coming from a future recurrent step.

Mathematically:

\[
\frac{\partial L_{T+1:T}}{\partial h_T}=0.
\]

So:

\[
\boxed{
dh_{\text{next}}=0
}
\tag{64.1}
\]

at the start of the backward pass.

This is not an arbitrary programming initialization.

It is the boundary condition of the computation graph.

---

# 65. Why \(h[t+1]\) appears in the tanh derivative

In the zero-based convention:

\[
h[t+1]=\tanh(a_t)
\]

where the `t` loop corresponds to the mathematical time step \(t+1\).

Therefore:

\[
\frac{\partial h[t+1]}{\partial a_t}
=
1-h[t+1]^2.
\]

That is why the implementation line is conceptually:

```text
da[i] = dh[i] * (1 - h[t+1][i] * h[t+1][i]);
```

The activation value itself is enough; you do not need to store \(a_t\) merely to compute tanh's derivative.

---

# 66. Why gradients must be zeroed before each sequence

The gradient arrays:

\[
dW_{xh},
dW_{hh},
db_h,
dW_{hy},
db_y
\]

contain accumulated contributions from the current sequence.

Therefore, before starting a new independent training example or sequence, initialize them to zero:

\[
\boxed{
dW_{xh}=0,\quad
dW_{hh}=0,\quad
db_h=0,\quad
dW_{hy}=0,\quad
db_y=0.
}
\tag{66.1}
\]

Otherwise gradients from a previous sequence would contaminate the current update.

For a mini-batch, you intentionally accumulate across multiple sequences before updating.

---

# 67. Update parameters only after BPTT is complete

The correct order for a full sequence is:

```text
zero gradients

forward through all T time steps

backward through all T time steps

optionally clip gradients

update parameters
```

Do not update \(W_{hh}\) halfway through BPTT.

The forward graph assumed a single fixed parameter value throughout that sequence.

Updating the parameters before BPTT is finished would make the backward calculation inconsistent with the forward computation you actually performed.

---

# 68. The complete algorithm in dependency order

This is the ordering to use when mentally reconstructing or implementing the RNN.

## Step A — Initialize

1. Set:

\[
h_0=0
\]

or another explicitly chosen initial state.

2. Set all gradient arrays to zero.

---

## Step B — Forward through time

For each:

\[
t=1,\ldots,T:
\]

3. Compute:

\[
a_t=W_{xh}x_t+W_{hh}h_{t-1}+b_h.
\]

4. Compute:

\[
h_t=\tanh(a_t).
\]

5. Compute:

\[
z_t=W_{hy}h_t+b_y.
\]

6. Compute stable softmax:

\[
\hat y_t=\operatorname{softmax}(z_t).
\]

7. Compute:

\[
L_t=-\sum_k y_{t,k}\log\hat y_{t,k}.
\]

8. Store the hidden state and any output values needed by BPTT.

---

## Step C — Backward through time

9. Initialize:

\[
\delta_{T+1}=0.
\]

Or in implementation form:

```text
dh_next = 0.
```

10. For:

\[
t=T,T-1,\ldots,1:
\]

compute:

\[
\delta_{z,t}=\hat y_t-y_t.
\]

11. Compute the total gradient entering the hidden state:

\[
dh_t
=
W_{hy}^T\delta_{z,t}
+
W_{hh}^T\delta_{t+1}.
\]

12. Cross tanh:

\[
\delta_t
=
dh_t\odot(1-h_t^2).
\]

13. Accumulate:

\[
dW_{hy}\mathrel{+}=\delta_{z,t}h_t^T.
\]

14. Accumulate:

\[
db_y\mathrel{+}=\delta_{z,t}.
\]

15. Accumulate:

\[
dW_{xh}\mathrel{+}=\delta_tx_t^T.
\]

16. Accumulate:

\[
dW_{hh}\mathrel{+}=\delta_th_{t-1}^T.
\]

17. Accumulate:

\[
db_h\mathrel{+}=\delta_t.
\]

18. Carry the recurrent gradient:

\[
\delta_{t-1}^{\text{future}}
=
W_{hh}^T\delta_t.
\]

Then continue backward.

---

## Step D — Clip if desired

19. Compute a global gradient norm.

20. Clip if necessary.

---

## Step E — Update

21. Apply:

\[
W_{xh}
\leftarrow
W_{xh}
-
\eta dW_{xh}.
\]

22. Apply:

\[
W_{hh}
\leftarrow
W_{hh}
-
\eta dW_{hh}.
\]

23. Apply:

\[
b_h
\leftarrow
b_h
-
\eta db_h.
\]

24. Apply:

\[
W_{hy}
\leftarrow
W_{hy}
-
\eta dW_{hy}.
\]

25. Apply:

\[
b_y
\leftarrow
b_y
-
\eta db_y.
\]

That is one complete RNN training step for one sequence.

---

# 69. Direct mapping from mathematics to C arrays

A practical layout might be:

```text
Parameters:

Wxh : H × D
Whh : H × H
Why : C × H

bh  : H
by  : C
```

Stored sequence values:

```text
x      : T × D
h      : (T + 1) × H
logits : T × C
probs  : T × C
target : T × C
```

Gradient arrays:

```text
dWxh : H × D
dWhh : H × H
dWhy : C × H

dbh  : H
dby  : C
```

Temporary arrays:

```text
dz       : C
dh       : H
da       : H
dh_next  : H
```

The exact memory layout can be flattened into one-dimensional C arrays if desired.

---

# 70. Flattened-array indexing in C

If a matrix has shape:

\[
R\times C,
\]

a row-major flattened representation can use:

```text
A[row * C + col]
```

So for:

\[
W_{xh}\in\mathbb R^{H\times D},
\]

use:

```text
Wxh[i * D + j]
```

for:

\[
W_{xh,ij}.
\]

For:

\[
W_{hh}\in\mathbb R^{H\times H},
\]

use:

```text
Whh[i * H + j]
```

For:

\[
W_{hy}\in\mathbb R^{C\times H},
\]

use:

```text
Why[k * H + j]
```

This indexing follows directly from the matrix dimensions.

---

# 71. Forward pseudocode

```text
h[0] = zero vector

for t = 0 ... T-1:

    for i = 0 ... H-1:
        sum = bh[i]

        for j = 0 ... D-1:
            sum += Wxh[i,j] * x[t,j]

        for j = 0 ... H-1:
            sum += Whh[i,j] * h[t,j]

        h[t+1,i] = tanh(sum)

    for k = 0 ... C-1:
        sum = by[k]

        for j = 0 ... H-1:
            sum += Why[k,j] * h[t+1,j]

        logits[t,k] = sum

    stable_softmax(logits[t], probs[t])

    loss += cross_entropy(probs[t], target[t])
```

Every block corresponds to one of the forward equations.

---

# 72. Backward pseudocode

```text
zero dWxh, dWhh, dWhy, dbh, dby

dh_next = zero vector

for t = T-1 ... 0:

    for k = 0 ... C-1:
        dz[k] = probs[t,k] - target[t,k]

    # Current output contribution:
    # dh = Whyᵀ dz

    # Add future recurrent contribution:
    # dh += dh_next

    for i = 0 ... H-1:
        da[i] =
            dh[i] *
            (1 - h[t+1,i] * h[t+1,i])

    # Output parameter gradients
    for k = 0 ... C-1:
        for j = 0 ... H-1:
            dWhy[k,j] += dz[k] * h[t+1,j]

        dby[k] += dz[k]

    # Hidden parameter gradients
    for i = 0 ... H-1:
        dbh[i] += da[i]

        for j = 0 ... D-1:
            dWxh[i,j] += da[i] * x[t,j]

        for j = 0 ... H-1:
            dWhh[i,j] += da[i] * h[t,j]

    # Prepare recurrent gradient for previous step:
    # dh_next = Whhᵀ da
```

The order matters.

---

# 73. Deriving the `dh` loop

The mathematical equation is:

\[
dh_t
=
W_{hy}^Tdz_t
+
dh_{\text{next}}.
\]

For hidden component \(j\):

\[
dh_{t,j}
=
\sum_k
W_{hy,kj}dz_{t,k}
+
dh_{\text{next},j}.
\]

Therefore the C loop is conceptually:

```text
for j = 0 ... H-1:
    dh[j] = 0

    for k = 0 ... C-1:
        dh[j] += Why[k,j] * dz[k]

    dh[j] += dh_next[j]
```

This is simply the scalar equation written as nested loops.

---

# 74. Deriving the `dh_next` loop

We need:

\[
dh_{\text{next}}
=
W_{hh}^Tda.
\]

For component \(j\):

\[
dh_{\text{next},j}
=
\sum_i
W_{hh,ij}da_i.
\]

Therefore:

```text
for j = 0 ... H-1:
    dh_next_new[j] = 0

    for i = 0 ... H-1:
        dh_next_new[j] += Whh[i,j] * da[i]
```

This is exactly the transpose multiplication:

\[
W_{hh}^Tda.
\]

---

# 75. Why one hidden gradient cannot simply be copied backward

A common incorrect idea is:

```text
dh_next = da
```

That is generally wrong.

The previous hidden state is connected through:

\[
W_{hh}.
\]

Therefore:

\[
\boxed{
\frac{\partial L}{\partial h_{t-1}}
=
W_{hh}^T
\frac{\partial L}{\partial a_t}.
}
\tag{75.1}
\]

So the correct recurrent propagation is:

\[
\boxed{
dh_{\text{next}}
=
W_{hh}^Tda.
}
\tag{75.2}
\]

The matrix multiplication is necessary.

---

# 76. A subtle but important implementation distinction

There are three related quantities:

\[
\boxed{
dz_t=\frac{\partial L_t}{\partial z_t}
}
\]

\[
\boxed{
dh_t=\frac{\partial L}{\partial h_t}
}
\]

\[
\boxed{
da_t=\frac{\partial L}{\partial a_t}.
}
\]

They are not interchangeable.

Their dependencies are:

\[
dz_t
=
\hat y_t-y_t
\]

\[
dh_t
=
W_{hy}^Tdz_t+dh_{\text{next}}
\]

\[
da_t
=
dh_t\odot(1-h_t^2).
\]

Then:

\[
dh_{\text{next}}
=
W_{hh}^Tda_t.
\]

Keeping these four transitions explicit makes BPTT much easier to implement correctly.

---

# 77. Why \(dW_{xh}\) uses \(x_t\), not \(h_t\)

The input-to-hidden weights directly multiply:

\[
x_t.
\]

Recall:

\[
a_t
=
W_{xh}x_t+\cdots
\]

so:

\[
\frac{\partial a_t}{\partial W_{xh}}
\]

depends on:

\[
x_t.
\]

Therefore:

\[
\boxed{
dW_{xh}
\mathrel{+}=
da_t x_t^T.
}
\]

It would be incorrect to use \(h_t\) there.

---

# 78. Why \(dW_{hh}\) uses \(h_{t-1}\), not \(h_t\)

The recurrent weights directly multiply the previous state:

\[
a_t
=
W_{hh}h_{t-1}+\cdots
\]

Therefore:

\[
\boxed{
dW_{hh}
\mathrel{+}=
da_t h_{t-1}^T.
}
\]

Again, the incoming quantity to a parameter determines the second factor in the outer product.

---

# 79. Why \(dW_{hy}\) uses \(h_t\)

The output weights directly multiply the current hidden state:

\[
z_t
=
W_{hy}h_t+b_y.
\]

Therefore:

\[
\boxed{
dW_{hy}
\mathrel{+}=
dz_t h_t^T.
}
\]

The same affine-layer rule is appearing again.

---

# 80. Why the hidden-state gradient is added, not overwritten

Suppose:

\[
h_t
\]

has one path to the current loss and another path to future losses.

The multivariable chain rule says:

\[
\frac{\partial L}{\partial h_t}
=
\frac{\partial L_t}{\partial h_t}
+
\frac{\partial L_{t+1:T}}{\partial h_t}.
\]

Therefore, in implementation:

```text
dh = current contribution + future contribution
```

not:

```text
dh = current contribution
```

and not:

```text
dh = future contribution
```

Both paths matter.

This addition is the defining operation of BPTT.

---

# 81. Many-to-one backward equations

Suppose there is only one loss:

\[
L=L_T.
\]

Then:

\[
dz_T=\hat y-y.
\]

At the final step:

\[
dh_T=W_{hy}^Tdz_T.
\]

Then:

\[
da_T
=
dh_T\odot(1-h_T^2).
\]

For the previous step:

\[
dh_{T-1}
=
W_{hh}^Tda_T.
\]

Then:

\[
da_{T-1}
=
dh_{T-1}\odot(1-h_{T-1}^2).
\]

Continue until:

\[
t=1.
\]

There is no direct:

\[
dz_t
\]

for earlier steps because there is no output loss there.

---

# 82. Truncated BPTT

For very long sequences, full BPTT may be expensive.

Suppose:

\[
T=10,000
\]

but we choose a truncation length:

\[
K=50.
\]

We then backpropagate only through the most recent \(50\) time steps for each update.

Conceptually, the full chain:

\[
L_T
\rightarrow
h_T
\rightarrow
h_{T-1}
\rightarrow
\cdots
\rightarrow
h_1
\]

is deliberately cut after \(K\) recurrent transitions.

This makes the gradient computation cheaper, but it means the update does not fully account for dependencies farther than \(K\) time steps into the past.

---

# 83. Parameter initialization

A first implementation can use small random weight values:

\[
W_{xh,ij}\sim U(-\epsilon,\epsilon)
\]

\[
W_{hh,ij}\sim U(-\epsilon,\epsilon)
\]

\[
W_{hy,kj}\sim U(-\epsilon,\epsilon).
\]

Biases can start at:

\[
b_h=0,
\qquad
b_y=0.
\]

The recurrent matrix is particularly important because it is applied repeatedly.

For educational purposes, keep the initialization simple and focus first on getting the forward and backward equations correct.

---

# 84. What happens with one-hot inputs?

Suppose the input vocabulary has size:

\[
D.
\]

A one-hot input has exactly one component equal to 1.

Then:

\[
x_t=e_j
\]

for some basis vector \(e_j\).

Therefore:

\[
W_{xh}x_t
\]

selects column \(j\) of \(W_{xh}\).

Mathematically:

\[
\boxed{
W_{xh}e_j
=
\text{column }j\text{ of }W_{xh}.
}
\tag{84.1}
\]

This can be used to optimize a character-level or word-level C implementation.

The general matrix equation remains the cleanest way to understand the mathematics.

---

# 85. Initial hidden state as a parameter or external state

The simplest choice is:

\[
h_0=0.
\]

But mathematically, \(h_0\) could be:

- a supplied state,
- a learned parameter,
- the final state of a previous chunk.

If \(h_0\) is learnable, its gradient can be obtained from:

\[
\frac{\partial L}{\partial h_0}
=
W_{hh}^T\delta_1.
\tag{85.1}
\]

Then \(h_0\) could also be updated by gradient descent.

For the first implementation, zero initialization is simpler.

---

# 86. The gradient with respect to the input

Although input gradients are not required to update the RNN's trainable weights, they are useful conceptually.

Recall:

\[
a_t=W_{xh}x_t+W_{hh}h_{t-1}+b_h.
\]

Therefore:

\[
\boxed{
\frac{\partial L}{\partial x_t}
=
W_{xh}^T\delta_t.
}
\tag{86.1}
\]

At scalar level:

\[
\frac{\partial L}{\partial x_{t,j}}
=
\sum_i
\delta_{t,i}W_{xh,ij}.
\]

This follows the same backward-affine rule used elsewhere.

---

# 87. Gradient descent update

Once the complete sequence gradients are accumulated, update:

\[
\boxed{
W_{xh}
\leftarrow
W_{xh}-\eta dW_{xh}
}
\tag{87.1}
\]

\[
\boxed{
W_{hh}
\leftarrow
W_{hh}-\eta dW_{hh}
}
\tag{87.2}
\]

\[
\boxed{
b_h
\leftarrow
b_h-\eta db_h
}
\tag{87.3}
\]

\[
\boxed{
W_{hy}
\leftarrow
W_{hy}-\eta dW_{hy}
}
\tag{87.4}
\]

\[
\boxed{
b_y
\leftarrow
b_y-\eta db_y.
}
\tag{87.5}
\]

The signs are identical to ordinary gradient descent.

The complicated part was obtaining the gradients.

---

# 88. A useful view of the five trainable objects

Think of the RNN as three affine operations:

### Hidden recurrence

\[
x_t,h_{t-1}
\rightarrow
a_t
\]

using:

\[
W_{xh},W_{hh},b_h.
\]

### Hidden nonlinearity

\[
a_t\rightarrow h_t.
\]

### Output layer

\[
h_t\rightarrow z_t
\]

using:

\[
W_{hy},b_y.
\]

Then:

\[
z_t\rightarrow\hat y_t\rightarrow L_t.
\]

So the RNN is just an affine network plus a recurrent feedback connection.

---

# 89. The one-step computation graph

At one time step:

```text
          x_t
           │
           ▼
        Wxh x_t
           │
           ├───────────┐
                       │
h_{t-1} → Whh h_{t-1}  │
                       ▼
                       a_t
                        │
                      tanh
                        │
                        ▼
                       h_t
                        │
                     Why h_t
                        │
                        ▼
                       z_t
                        │
                     softmax
                        │
                        ▼
                      ŷ_t
                        │
                   cross-entropy
                        │
                        ▼
                       L_t
```

But:

\[
h_t
\]

also feeds the next time step:

```text
h_t ───────► a_{t+1}
```

That one additional arrow is what transforms the ordinary feed-forward derivation into BPTT.

---

# 90. The full sequence graph

For \(T=4\):

```text
x1 ─► a1 ─► h1 ─► a2 ─► h2 ─► a3 ─► h3 ─► a4 ─► h4
       │             │             │             │
       ▼             ▼             ▼             ▼
      z1            z2            z3            z4
       │             │             │             │
       ▼             ▼             ▼             ▼
      ŷ1            ŷ2            ŷ3            ŷ4
       │             │             │             │
       ▼             ▼             ▼             ▼
      L1            L2            L3            L4
```

Each recurrence uses the same:

\[
W_{xh},W_{hh},b_h.
\]

Each output uses the same:

\[
W_{hy},b_y.
\]

Therefore every shared parameter receives a sum of gradients from all relevant time steps.

---

# 91. Why BPTT is called "through time"

The network itself is not differentiating with respect to the word "time."

"Time" simply labels the repeated computation:

\[
1,2,\ldots,T.
\]

BPTT means that the computation graph is unrolled over these repeated steps and backpropagation is performed across them in reverse order:

\[
T\rightarrow T-1\rightarrow\cdots\rightarrow1.
\]

---

# 92. Why sequence length changes the graph depth

For one step:

\[
x_1\rightarrow h_1.
\]

For two:

\[
x_1\rightarrow h_1\rightarrow h_2.
\]

For \(T\):

\[
x_1\rightarrow h_1\rightarrow\cdots\rightarrow h_T.
\]

So the same RNN cell is being repeated, making the unrolled graph deeper as \(T\) increases.

This is why long sequences create long gradient paths.

---

# 93. A dependency table for every quantity

| Quantity | What it means | Why we calculate it | Where it is used next |
|---|---|---|---|
| \(a_t\) | hidden preactivation | input to tanh; direct affine output | produces \(h_t\), and its gradient gives hidden parameter gradients |
| \(h_t\) | hidden state | representation of current input + past | produces \(z_t\) and feeds next recurrent step |
| \(z_t\) | output logits | input to softmax | produces \(\hat y_t\) |
| \(\hat y_t\) | predicted probabilities | needed by cross-entropy | produces \(L_t\), then \(dz_t\) |
| \(L_t\) | loss at one time step | contributes to sequence objective | starting point for backward pass |
| \(\delta_{z,t}\) | \(\partial L_t/\partial z_t\) | output error signal | \(dW_{hy},db_y,dh_t\) |
| \(dh_t\) | \(\partial L/\partial h_t\) | combines current and future paths | crosses tanh |
| \(\delta_t\) | \(\partial L/\partial a_t\) | hidden error signal | \(dW_{xh},dW_{hh},db_h\), recurrent propagation |
| \(dW_{xh}\) | gradient of input weights | needed for learning | update \(W_{xh}\) |
| \(dW_{hh}\) | gradient of recurrent weights | needed for learning | update \(W_{hh}\) |
| \(db_h\) | gradient of hidden bias | needed for learning | update \(b_h\) |
| \(dW_{hy}\) | gradient of output weights | needed for learning | update \(W_{hy}\) |
| \(db_y\) | gradient of output bias | needed for learning | update \(b_y\) |

---

# 94. The scalar chain for one recurrent weight

Take:

\[
W_{hh,ij}.
\]

Its effect at time \(t\) is:

\[
W_{hh,ij}
\rightarrow
a_{t,i}
\rightarrow
h_{t,i}
\rightarrow
\text{current and future losses}.
\]

The total derivative is:

\[
\boxed{
\frac{\partial L}{\partial W_{hh,ij}}
=
\sum_t
\frac{\partial L}{\partial a_{t,i}}
\frac{\partial a_{t,i}}
{\partial W_{hh,ij}}.
}
\tag{94.1}
\]

Now:

\[
\frac{\partial L}{\partial a_{t,i}}
=
\delta_{t,i}
\]

and:

\[
\frac{\partial a_{t,i}}
{\partial W_{hh,ij}}
=
h_{t-1,j}.
\]

Therefore:

\[
\boxed{
\frac{\partial L}{\partial W_{hh,ij}}
=
\sum_t
\delta_{t,i}h_{t-1,j}.
}
\tag{94.2}
\]

Every complicated-looking part of BPTT is ultimately feeding into the first factor:

\[
\delta_{t,i}.
\]

Once that error signal is known, the local parameter gradient is straightforward.

---

# 95. The scalar chain for one recurrent state

Take:

\[
h_{t,i}.
\]

The total loss depends on it through:

1. \(z_t\rightarrow L_t\),
2. \(a_{t+1}\rightarrow h_{t+1}\rightarrow\cdots\).

Therefore:

\[
\boxed{
\frac{\partial L}{\partial h_{t,i}}
=
\sum_k
\delta_{z,t,k}W_{hy,ki}
+
\sum_j
\delta_{t+1,j}W_{hh,ji}.
}
\tag{95.1}
\]

Then:

\[
\boxed{
\delta_{t,i}
=
\frac{\partial L}{\partial h_{t,i}}
(1-h_{t,i}^2).
}
\tag{95.2}
\]

This pair of equations is the cleanest scalar statement of BPTT.

---

# 96. Why future information is added before tanh's derivative

The hidden activation is:

\[
h_t=\tanh(a_t).
\]

Both the current output path and the future recurrent path converge at \(h_t\).

Therefore the correct order is:

\[
\text{current contribution}
+
\text{future contribution}
\]

first, giving:

\[
\frac{\partial L}{\partial h_t}.
\]

Only then do we cross the local nonlinearity:

\[
\frac{\partial L}{\partial a_t}
=
\frac{\partial L}{\partial h_t}
\odot
(1-h_t^2).
\]

So you should not independently multiply the two branches by tanh's derivative and then add without understanding that both branches first contribute to the same \(h_t\) gradient.

The mathematically clean chain is:

\[
\boxed{
L
\rightarrow
h_t
\rightarrow
a_t.
}
\]

---

# 97. Numerical sanity checks

When implementing from scratch, several simple checks are useful.

## Check 1 — Probability normalization

For every \(t\):

\[
\boxed{
\sum_k\hat y_{t,k}\approx1.
}
\]

## Check 2 — Softmax probabilities

Every probability should satisfy:

\[
0<\hat y_{t,k}<1.
\]

## Check 3 — Finite loss

The loss should be finite:

\[
L<\infty.
\]

## Check 4 — Dimension consistency

Every temporary buffer should match the dimensions listed earlier.

## Check 5 — Gradient signs

A correct gradient should reduce the loss under a sufficiently small gradient-descent step more often than not.

## Check 6 — Finite-difference gradient checking

For a small network, compare analytical gradients to numerical gradients.

---

# 98. Numerical gradient checking

For one scalar parameter \(\theta\), approximate:

\[
\boxed{
\frac{\partial L}{\partial\theta}
\approx
\frac{
L(\theta+\epsilon)
-
L(\theta-\epsilon)
}{
2\epsilon
}.
}
\tag{98.1}
\]

This is the central finite-difference approximation.

Compare it to the analytical gradient returned by BPTT.

For a correctly implemented gradient, the values should be close for a suitably small \(\epsilon\), subject to floating-point error.

This is one of the best ways to debug a from-scratch RNN.

---

# 99. Why finite differences are especially useful for RNNs

An error in:

- the recurrent transpose,
- the time indexing,
- the hidden-state storage,
- the `dh_next` propagation,
- the tanh derivative,
- or the gradient accumulation,

can produce a plausible-looking training loop that is mathematically wrong.

Finite differences test the entire chain:

\[
\theta
\rightarrow
\text{forward through all time}
\rightarrow
L.
\]

So they can reveal mistakes that are difficult to spot from code inspection.

For initial debugging, use a tiny network such as:

\[
D=2,\quad H=3,\quad C=2,\quad T=4.
\]

---

# 100. Why you should test one parameter at a time first

A full RNN has many parameters.

You do not initially need to compare every gradient.

Pick:

- one entry of \(W_{xh}\),
- one entry of \(W_{hh}\),
- one entry of \(W_{hy}\),
- one hidden bias,
- one output bias.

For each parameter:

1. compute the analytical gradient,
2. perturb by \(+\epsilon\),
3. perturb by \(-\epsilon\),
4. recompute the full sequence loss,
5. compare using the central difference.

Once those pass, test more parameters.

---

# 101. A useful relative-error measure

For analytical gradient \(g_a\) and numerical gradient \(g_n\), use:

\[
\boxed{
\text{relative error}
=
\frac{|g_a-g_n|}
{\max(1,|g_a|,|g_n|)}.
}
\tag{101.1}
\]

For sufficiently small models and a correct implementation, the relative error should usually be very small.

Do not demand exact equality because the finite-difference approximation itself and floating-point arithmetic introduce error.

---

# 102. Common implementation mistakes and the mathematical reason they are wrong

## Mistake 1 — Using \(h_t\) instead of \(h_{t-1}\) in \(dW_{hh}\)

Wrong:

\[
dW_{hh}\mathrel{+}=\delta_t h_t^T.
\]

Correct:

\[
\boxed{
dW_{hh}\mathrel{+}=\delta_t h_{t-1}^T.
}
\]

Why?

Because \(W_{hh}\) directly multiplies \(h_{t-1}\).

---

## Mistake 2 — Forgetting the future gradient

Wrong:

\[
dh_t=W_{hy}^T\delta_{z,t}.
\]

Correct:

\[
\boxed{
dh_t
=
W_{hy}^T\delta_{z,t}
+
dh_{\text{next}}.
}
\]

Why?

Because \(h_t\) affects future losses.

---

## Mistake 3 — Setting \(dh_{\text{next}}=da\)

Wrong:

\[
dh_{\text{next}}=da.
\]

Correct:

\[
\boxed{
dh_{\text{next}}
=
W_{hh}^Tda.
}
\]

Why?

Because the recurrent connection is parameterized by \(W_{hh}\).

---

## Mistake 4 — Forgetting to accumulate shared parameter gradients

Wrong:

\[
dW_{hh}=\delta_t h_{t-1}^T
\]

at every time step, overwriting the previous value.

Correct:

\[
\boxed{
dW_{hh}\mathrel{+}=\delta_t h_{t-1}^T.
}
\]

Why?

Because the same matrix is used at every time step.

---

## Mistake 5 — Updating parameters before finishing BPTT

The forward computation used one set of parameters.

The backward pass should correspond to that same forward computation.

Therefore update only after gradients have been accumulated.

---

## Mistake 6 — Unstable softmax

Directly calculating:

\[
e^{z_i}
\]

can overflow.

Use:

\[
z_i-m.
\]

---

# 103. Why an RNN can be implemented without automatic differentiation

Every operation in the vanilla RNN has an explicit derivative:

### Addition

\[
\frac{d}{dx}(x+c)=1.
\]

### Matrix multiplication

\[
z=Wx
\]

gives:

\[
\frac{\partial L}{\partial x}=W^T\frac{\partial L}{\partial z}.
\]

### Tanh

\[
\tanh'(x)=1-\tanh^2(x).
\]

### Softmax + cross entropy

\[
\frac{\partial L}{\partial z}
=
\hat y-y.
\]

Therefore BPTT can be implemented directly with loops.

Automatic differentiation is useful for large systems, but it is not necessary for understanding or implementing this model.

---

# 104. The minimal conceptual C model

The core implementation can be mentally reduced to:

```text
FORWARD

h[0] = 0

for t:
    h[t+1] = tanh(Wxh*x[t] + Whh*h[t] + bh)
    z[t]   = Why*h[t+1] + by
    p[t]   = softmax(z[t])


BACKWARD

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


UPDATE

Wxh -= η*dWxh
Whh -= η*dWhh
bh  -= η*dbh

Why -= η*dWhy
by  -= η*dby
```

If you understand every line in that block mathematically, you understand the vanilla RNN training algorithm.

---

# 105. The final reconstruction sheet

When you come back later and have forgotten everything, start here.

## What do I need?

\[
\boxed{
dW_{xh},
dW_{hh},
dbh,
dW_{hy},
dby
}
\]

## Forward

\[
\boxed{
a_t
=
W_{xh}x_t
+
W_{hh}h_{t-1}
+
b_h
}
\]

\[
\boxed{
h_t=\tanh(a_t)
}
\]

\[
\boxed{
z_t=W_{hy}h_t+b_y
}
\]

\[
\boxed{
\hat y_t=\operatorname{softmax}(z_t)
}
\]

\[
\boxed{
L_t=-\sum_k y_{t,k}\log\hat y_{t,k}
}
\]

## Backward

Start at:

\[
\boxed{
\delta_{T+1}=0.
}
\]

At each time step, backward:

\[
\boxed{
\delta_{z,t}=\hat y_t-y_t
}
\]

\[
\boxed{
dh_t
=
W_{hy}^T\delta_{z,t}
+
W_{hh}^T\delta_{t+1}
}
\]

\[
\boxed{
\delta_t
=
dh_t\odot(1-h_t^2)
}
\]

Then:

\[
\boxed{
dW_{hy}\mathrel{+}=\delta_{z,t}h_t^T
}
\]

\[
\boxed{
db_y\mathrel{+}=\delta_{z,t}
}
\]

\[
\boxed{
dW_{xh}\mathrel{+}=\delta_t x_t^T
}
\]

\[
\boxed{
dW_{hh}\mathrel{+}=\delta_t h_{t-1}^T
}
\]

\[
\boxed{
db_h\mathrel{+}=\delta_t
}
\]

Then move one time step backward.

Finally:

\[
\boxed{
\theta\leftarrow\theta-\eta\,d\theta.
}
\]

---

# 106. One sentence that ties the whole derivation together

> **Start from the sequence loss, walk backward through the unrolled recurrent computation graph using the chain rule, add all gradient paths that meet at a hidden state, and whenever you reach the preactivation of an affine transformation, you have exactly the gradient needed to calculate that transformation's parameter gradients.**

For the vanilla RNN:

\[
\boxed{
L
\rightarrow
\hat y_t
\rightarrow
z_t
\rightarrow
\left\{
W_{hy},b_y,h_t
\right\}
}
\]

then:

\[
\boxed{
h_t
\rightarrow
\left\{
\text{current output path},
\text{future recurrent path}
\right\}
}
\]

then:

\[
\boxed{
\frac{\partial L}{\partial h_t}
\rightarrow
\frac{\partial L}{\partial a_t}
\rightarrow
\left\{
W_{xh},W_{hh},b_h
\right\}.
}
\]

And because the same parameters are reused at every time step:

\[
\boxed{
dW
=
\sum_t dW_t.
}
\]

The entire RNN derivation is therefore one dependency chain:

\[
\boxed{
(x_t,h_{t-1})
\rightarrow
a_t
\rightarrow
h_t
\rightarrow
z_t
\rightarrow
\hat y_t
\rightarrow
L_t
}
\]

forward, and:

\[
\boxed{
L_t
\rightarrow
\delta_{z,t}
\rightarrow
dh_t
\rightarrow
\delta_t
\rightarrow
\{dW_{xh},dW_{hh},db_h\}
}
\]

backward, with:

\[
\boxed{
dh_t
=
W_{hy}^T\delta_{z,t}
+
W_{hh}^T\delta_{t+1}
}
\]

being the key equation that makes the network recurrent during backpropagation.
