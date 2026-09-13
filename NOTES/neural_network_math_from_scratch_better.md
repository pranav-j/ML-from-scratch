# Neural Network Mathematics From First Principles

## A self-contained derivation of a \(784 \rightarrow 300 \rightarrow 10\) network

This document is organized around one question:

> **We ultimately need the gradients of the weights and biases. What derivative do we need first, why do we need it, and where does it get plugged next?**

The network is:

\[
784 \rightarrow 300 \rightarrow 10
\]

with:

- sigmoid in the hidden layer,
- softmax in the output layer,
- cross-entropy loss,
- gradient descent.

The derivation is for **one training example**. Mini-batches are an extension at the end.

---

# 0. THE MAP — READ THIS FIRST

The single most important thing in this document is the dependency structure.

## 0.1 What we ultimately need

The trainable parameters are:

\[
W^{(1)},\quad b^{(1)},\quad W^{(2)},\quad b^{(2)}.
\]

To update them, we need:

\[
\boxed{
\frac{\partial L}{\partial W^{(1)}},\quad
\frac{\partial L}{\partial b^{(1)}},\quad
\frac{\partial L}{\partial W^{(2)}},\quad
\frac{\partial L}{\partial b^{(2)}}
}
\tag{0.1}
\]

So **these four gradients are the actual destination** of the whole derivation.

---

## 0.2 Forward dependency

The network computes, in this order:

\[
\boxed{
x
\rightarrow
z^{(1)}
\rightarrow
a^{(1)}
\rightarrow
z^{(2)}
\rightarrow
\hat y
\rightarrow
L
}
\tag{0.2}
\]

The equations are:

\[
z^{(1)}=W^{(1)}x+b^{(1)}
\]

\[
a^{(1)}=\sigma(z^{(1)})
\]

\[
z^{(2)}=W^{(2)}a^{(1)}+b^{(2)}
\]

\[
\hat y=\operatorname{softmax}(z^{(2)})
\]

\[
L=-\sum_k y_k\log\hat y_k.
\]

---

## 0.3 Backward dependency

Backpropagation reverses the arrows:

\[
\boxed{
L
\rightarrow
\hat y
\rightarrow
z^{(2)}
\rightarrow
a^{(1)}
\rightarrow
z^{(1)}
\rightarrow
W^{(1)},b^{(1)}
}
\tag{0.3}
\]

But there is a fork at \(z^{(2)}\), because once we know its gradient we can both:

1. obtain the final-layer parameter gradients, and
2. continue backward into the hidden layer.

The full map is:

```text
                           FORWARD

x ──► z¹ ──► a¹ ──► z² ──► ŷ ──► L
     ▲       ▲       ▲       ▲
     │       │       │       │
    W¹,b¹   sigmoid W²,b²   softmax

                           BACKWARD

L
│
▼
∂L/∂ŷ
│
│  combine with softmax derivative
▼
∂L/∂z² = δ²
│
├──────────────► ∂L/∂W² ──► update W²
│
├──────────────► ∂L/∂b² ──► update b²
│
│
│  propagate through second affine layer
▼
∂L/∂a¹
│
│  multiply by sigmoid derivative
▼
∂L/∂z¹ = δ¹
│
├──────────────► ∂L/∂W¹ ──► update W¹
│
└──────────────► ∂L/∂b¹ ──► update b¹
```

---

# 0.4 The prerequisite table

| What we ultimately need | What must be known first | Why that prerequisite is needed |
|---|---|---|
| \(\partial L/\partial W^{(2)}\) | \(\partial L/\partial z^{(2)}\) | \(W^{(2)}\) directly creates \(z^{(2)}\) |
| \(\partial L/\partial b^{(2)}\) | \(\partial L/\partial z^{(2)}\) | \(b^{(2)}\) directly creates \(z^{(2)}\) |
| \(\partial L/\partial W^{(1)}\) | \(\partial L/\partial z^{(1)}\) | \(W^{(1)}\) directly creates \(z^{(1)}\) |
| \(\partial L/\partial b^{(1)}\) | \(\partial L/\partial z^{(1)}\) | \(b^{(1)}\) directly creates \(z^{(1)}\) |
| \(\partial L/\partial z^{(2)}\) | \(\partial L/\partial\hat y\) and \(\partial\hat y/\partial z^{(2)}\) | We must cross cross-entropy and softmax |
| \(\partial L/\partial a^{(1)}\) | \(\partial L/\partial z^{(2)}\) | We must cross the second affine layer |
| \(\partial L/\partial z^{(1)}\) | \(\partial L/\partial a^{(1)}\) and \(\partial a^{(1)}/\partial z^{(1)}\) | We must cross the sigmoid |

So the actual derivation order is:

\[
\boxed{
\frac{\partial L}{\partial\hat y}
\rightarrow
\frac{\partial\hat y}{\partial z^{(2)}}
\rightarrow
\frac{\partial L}{\partial z^{(2)}}
}
\]

then:

\[
\boxed{
\frac{\partial L}{\partial z^{(2)}}
\rightarrow
\left\{
\begin{array}{c}
\frac{\partial L}{\partial W^{(2)}}\\
\frac{\partial L}{\partial b^{(2)}}\\
\frac{\partial L}{\partial a^{(1)}}
\end{array}
\right.
}
\]

then:

\[
\boxed{
\frac{\partial L}{\partial a^{(1)}}
\rightarrow
\frac{\partial a^{(1)}}{\partial z^{(1)}}
\rightarrow
\frac{\partial L}{\partial z^{(1)}}
}
\]

and finally:

\[
\boxed{
\frac{\partial L}{\partial z^{(1)}}
\rightarrow
\left\{
\begin{array}{c}
\frac{\partial L}{\partial W^{(1)}}\\
\frac{\partial L}{\partial b^{(1)}}
\end{array}
\right.
}
\]

**If you lose track, come back to this page.**

---

# 1. Notation and dimensions

We use column vectors.

## 1.1 Input

\[
\boxed{x\in\mathbb R^{784}}
\tag{1.1}
\]

Write:

\[
x=
\begin{bmatrix}
x_1\\x_2\\\vdots\\x_{784}
\end{bmatrix}.
\]

---

## 1.2 First-layer parameters

There are 300 hidden neurons, so:

\[
\boxed{W^{(1)}\in\mathbb R^{300\times784}}
\tag{1.2}
\]

and:

\[
\boxed{b^{(1)}\in\mathbb R^{300}}
\tag{1.3}
\]

---

## 1.3 Hidden quantities

The hidden preactivation and activation are both 300-dimensional:

\[
\boxed{z^{(1)},a^{(1)}\in\mathbb R^{300}}
\tag{1.4}
\]

---

## 1.4 Second-layer parameters

There are 10 output classes:

\[
\boxed{W^{(2)}\in\mathbb R^{10\times300}}
\tag{1.5}
\]

\[
\boxed{b^{(2)}\in\mathbb R^{10}}
\tag{1.6}
\]

---

## 1.5 Output and target

\[
\boxed{z^{(2)},\hat y,y\in\mathbb R^{10}}
\tag{1.7}
\]

For a one-hot target:

\[
\sum_{k=1}^{10}y_k=1.
\tag{1.8}
\]

That identity will be used in the softmax + cross-entropy derivation.

---

# 2. The forward pass

We now compute everything in the order the network needs it.

---

## 2.1 First affine transformation

The first layer computes:

\[
\boxed{z^{(1)}=W^{(1)}x+b^{(1)}}
\tag{2.1}
\]

Dimensions:

\[
(300\times784)(784\times1)+(300\times1)
=(300\times1).
\]

So:

\[
z^{(1)}\in\mathbb R^{300}.
\]

At scalar level:

\[
\boxed{
 z^{(1)}_j
 =
 \sum_{i=1}^{784}W^{(1)}_{ji}x_i+b^{(1)}_j
}
\tag{2.2}
\]

for \(j=1,\ldots,300\).

---

## 2.2 Sigmoid activation

The hidden layer applies sigmoid elementwise:

\[
\boxed{a^{(1)}=\sigma(z^{(1)})}
\tag{2.3}
\]

where:

\[
\boxed{\sigma(z)=\frac{1}{1+e^{-z}}}
\tag{2.4}
\]

So:

\[
\boxed{a^{(1)}_j=\sigma(z^{(1)}_j)}
\tag{2.5}
\]

Why do we need \(a^{(1)}\)? Because it is the input to the second layer.

---

## 2.3 Second affine transformation

The output layer computes:

\[
\boxed{z^{(2)}=W^{(2)}a^{(1)}+b^{(2)}}
\tag{2.6}
\]

Dimensions:

\[
(10\times300)(300\times1)+(10\times1)
=(10\times1).
\]

At scalar level:

\[
\boxed{
 z^{(2)}_k
 =
 \sum_{j=1}^{300}W^{(2)}_{kj}a^{(1)}_j+b^{(2)}_k
}
\tag{2.7}
\]

for \(k=1,\ldots,10\).

These \(z^{(2)}_k\) are the logits.

---

## 2.4 Softmax

Softmax converts the 10 logits into probabilities:

\[
\boxed{
\hat y_k
=
\frac{e^{z^{(2)}_k}}
{\sum_{r=1}^{10}e^{z^{(2)}_r}}
}
\tag{2.8}
\]

or:

\[
\boxed{\hat y=\operatorname{softmax}(z^{(2)})}
\tag{2.9}
\]

and:

\[
0<\hat y_k<1,
\qquad
\sum_k\hat y_k=1.
\tag{2.10}
\]

---

## 2.5 Cross-entropy loss

For one example:

\[
\boxed{
L=-\sum_{k=1}^{10}y_k\log\hat y_k
}
\tag{2.11}
\]

If the correct class is \(c\), then the target is one-hot and:

\[
\boxed{L=-\log\hat y_c.}
\tag{2.12}
\]

So the complete forward chain is:

\[
\boxed{
 x
\rightarrow
 z^{(1)}
\rightarrow
 a^{(1)}
\rightarrow
 z^{(2)}
\rightarrow
 \hat y
\rightarrow
 L
}
\tag{2.13}
\]

---

# 3. What backpropagation is trying to do

Gradient descent needs the gradient of the loss with respect to each parameter.

For a parameter \(\theta\):

\[
\boxed{
\theta_{new}=\theta-\eta\frac{\partial L}{\partial\theta}
}
\tag{3.1}
\]

Therefore, for the whole network:

\[
W^{(1)}\leftarrow W^{(1)}-
\eta\frac{\partial L}{\partial W^{(1)}}
\tag{3.2}
\]

\[
b^{(1)}\leftarrow b^{(1)}-
\eta\frac{\partial L}{\partial b^{(1)}}
\tag{3.3}
\]

\[
W^{(2)}\leftarrow W^{(2)}-
\eta\frac{\partial L}{\partial W^{(2)}}
\tag{3.4}
\]

\[
b^{(2)}\leftarrow b^{(2)}-
\eta\frac{\partial L}{\partial b^{(2)}}.
\tag{3.5}
\]

So backpropagation is simply a way to efficiently calculate these four gradients.

---

# 4. The chain rule is the entire engine

For scalar functions:

\[
L=L(u),\qquad u=u(v),
\]

then:

\[
\boxed{
\frac{\partial L}{\partial v}
=
\frac{\partial L}{\partial u}
\frac{\partial u}{\partial v}
}
\tag{4.1}
\]

Interpretation:

> The sensitivity of the loss to \(v\) is the sensitivity of the loss to \(u\), multiplied by how sensitive \(u\) is to \(v\).

For a multivariable function, if one variable influences the loss through multiple paths, the path contributions **add**.

This is exactly what happens with softmax and later with the hidden layer.

---

# 5. FIRST DESTINATION: \(\partial L/\partial z^{(2)}\)

This is the most important intermediate derivative because it directly unlocks the gradients of the final layer.

We know:

\[
z^{(2)}
ightarrow\hat y\rightarrow L.
\]

Therefore we first want:

\[
\boxed{\frac{\partial L}{\partial z^{(2)}}}
\tag{5.1}
\]

To obtain it, we need:

\[
\frac{\partial L}{\partial\hat y}
\]

and:

\[
\frac{\partial\hat y}{\partial z^{(2)}}.
\]

This is why the next two sections derive those quantities.

---

# 6. Derivative of cross-entropy with respect to \(\hat y\)

Start from:

\[
L=-\sum_k y_k\log\hat y_k.
\]

Differentiate with respect to \(\hat y_l\):

\[
\frac{\partial L}{\partial\hat y_l}
=
-y_l\frac{1}{\hat y_l}.
\]

Therefore:

\[
\boxed{
\frac{\partial L}{\partial\hat y_l}
=-\frac{y_l}{\hat y_l}
}
\tag{6.1}
\]

At this point we **do not yet have** the derivative we actually want. We still need to cross softmax.

---

# 7. Softmax derivative — why there are two cases

For this derivation, temporarily write:

\[
\hat y_k=\frac{e^{z_k}}{S},
\tag{7.1}
\]

where:

\[
\boxed{S=\sum_j e^{z_j}.}
\tag{7.2}
\]

We want:

\[
\boxed{
\frac{\partial\hat y_k}{\partial z_l}
}
\tag{7.3}
\]

Here:

- \(k\) tells us **which output probability** we are looking at,
- \(l\) tells us **which logit** we are changing.

There are two cases:

\[
k=l
\]

and:

\[
k\neq l.
\]

The two cases differ because when \(k=l\), the same variable appears in both numerator and denominator.

---

# 8. A small but important derivative: \(\partial S/\partial z_k\)

Recall:

\[
S=e^{z_1}+e^{z_2}+\cdots+e^{z_k}+\cdots+e^{z_{10}}.
\]

Differentiate with respect to \(z_k\):

\[
\frac{\partial S}{\partial z_k}
=
\frac{\partial e^{z_1}}{\partial z_k}
+\cdots+
\frac{\partial e^{z_k}}{\partial z_k}
+\cdots+
\frac{\partial e^{z_{10}}}{\partial z_k}.
\]

Every term not containing \(z_k\) is constant with respect to \(z_k\), so its derivative is 0. Only one term survives:

\[
\frac{\partial e^{z_k}}{\partial z_k}=e^{z_k}.
\]

Therefore:

\[
\boxed{
\frac{\partial S}{\partial z_k}=e^{z_k}
}
\tag{8.1}
\]

This same idea is compactly represented by the Kronecker delta; that notation is explained later.

---

# 9. Kronecker delta

The Kronecker delta is:

\[
\boxed{
\delta_{ij}
=
\begin{cases}
1,&i=j,\\
0,&i\neq j.
\end{cases}}
\tag{9.1}
\]

It is simply an “index equality detector.”

For example:

\[
\delta_{11}=1,
\qquad
\delta_{27}=0.
\]

A useful identity is:

\[
\boxed{
\frac{\partial x_i}{\partial x_j}=\delta_{ij}
}
\tag{9.2}
\]

because differentiating \(x_i\) with respect to itself gives 1, while differentiating it with respect to another independent variable gives 0.

Since:

\[
\frac{\partial z_j}{\partial z_l}=\delta_{jl},
\]

we could write:

\[
\frac{\partial e^{z_j}}{\partial z_l}
=e^{z_j}\delta_{jl}.
\]

Then:

\[
\frac{\partial S}{\partial z_l}
=
\sum_j e^{z_j}\delta_{jl}
=e^{z_l}.
\]

The delta is only shorthand; the underlying calculus is the same.

---

# 10. Softmax derivative: case \(k=l\)

We have:

\[
\hat y_k=\frac{e^{z_k}}{S}.
\]

Now differentiate with respect to \(z_k\).

Because both numerator and denominator depend on \(z_k\), use the quotient rule:

\[
\frac{\partial\hat y_k}{\partial z_k}
=
\frac{
S\frac{\partial e^{z_k}}{\partial z_k}
-
 e^{z_k}\frac{\partial S}{\partial z_k}
}{S^2}.
\tag{10.1}
\]

We know:

\[
\frac{\partial e^{z_k}}{\
partial z_k}=e^{z_k}
\]

and from equation (8.1):

\[
\frac{\partial S}{\partial z_k}=e^{z_k}.
\]

Therefore:

\[
\frac{\partial\hat y_k}{\partial z_k}
=
\frac{Se^{z_k}-e^{2z_k}}{S^2}.
\tag{10.2}
\]

Factor \(e^{z_k}/S\):

\[
=
\frac{e^{z_k}}{S}
\left(1-\frac{e^{z_k}}S\right).
\tag{10.3}
\]

Using:

\[
\hat y_k=\frac{e^{z_k}}S,
\]

we obtain:

\[
\boxed{
\frac{\partial\hat y_k}{\partial z_k}
=
\hat y_k(1-\hat y_k)
}
\tag{10.4}
\]

---

# 11. Softmax derivative: case \(k\neq l\)

Now consider:

\[
\frac{\partial\hat y_k}{\partial z_l},
\qquad k\neq l.
\]

We still have:

\[
\hat y_k=\frac{e^{z_k}}S.
\]

But now the numerator does **not** depend on \(z_l\). Thus:

\[
\frac{\partial e^{z_k}}{\partial z_l}=0.
\]

The denominator does depend on \(z_l\), and:

\[
\frac{\partial S}{\partial z_l}=e^{z_l}.
\]

The quotient rule gives:

\[
\frac{\partial\hat y_k}{\partial z_l}
=
\frac{
S(0)-e^{z_k}e^{z_l}
}{S^2}.
\tag{11.1}
\]

Therefore:

\[
\frac{\partial\hat y_k}{\partial z_l}
=
-\frac{e^{z_k}e^{z_l}}{S^2}.
\tag{11.2}
\]

Recognize:

\[
\frac{e^{z_k}}S=\hat y_k
\]

and:

\[
\frac{e^{z_l}}S=\hat y_l.
\]

Therefore:

\[
\boxed{
\frac{\partial\hat y_k}{\partial z_l}
=-\hat y_k\hat y_l,
\qquad k\neq l
}
\tag{11.3}
\]

The negative sign is intuitive: increasing one logit increases its probability and steals probability from the other classes because all probabilities must still sum to 1.

---

# 12. The softmax Jacobian

Because softmax has 10 outputs and 10 inputs, its derivative is a \(10\times10\) Jacobian.

A Jacobian is simply the matrix of all partial derivatives:

\[
\boxed{
J_{kl}=\frac{\partial\hat y_k}{\partial z_l}
}
\tag{12.1}
\]

Its rows correspond to output probabilities; its columns correspond to logits.

The entries are:

\[
\boxed{
J_{kl}
=
\begin{cases}
\hat y_k(1-\hat y_k),&k=l,\\[4pt]
-\hat y_k\hat y_l,&k\neq l.
\end{cases}}
\tag{12.2}
\]

In compact matrix notation:

\[
\boxed{
J=\operatorname{diag}(\hat y)-\hat y\hat y^T.
}
\tag{12.3}
\]

For ordinary implementation with softmax + cross-entropy, we will not actually construct this Jacobian. It is enough to know what it represents because the product simplifies dramatically.

---

# 13. NOW combine cross-entropy and softmax

We want:

\[
\boxed{
\frac{\partial L}{\partial z_l}
}
\tag{13.1}
\]

Because \(z_l\) affects every softmax output, the multivariable chain rule gives:

\[
\boxed{
\frac{\partial L}{\partial z_l}
=
\sum_k
\frac{\partial L}{\partial\hat y_k}
\frac{\partial\hat y_k}{\partial z_l}
}
\tag{13.2}
\]

Substitute equation (6.1):

\[
\frac{\partial L}{\partial z_l}
=
\sum_k
\left(-\frac{y_k}{\hat y_k}\right)
\frac{\partial\hat y_k}{\partial z_l}.
\tag{13.3}
\]

We must split the sum because softmax's derivative has different forms when \(k=l\) and \(k\neq l\):

\[
\frac{\partial L}{\partial z_l}
=
\left(-\frac{y_l}{\hat y_l}\right)
\frac{\partial\hat y_l}{\partial z_l}
+
\sum_{k\neq l}
\left(-\frac{y_k}{\hat y_k}\right)
\frac{\partial\hat y_k}{\partial z_l}.
\tag{13.4}
\]

### The \(k=l\) term

Using equation (10.4):

\[
\left(-\frac{y_l}{\hat y_l}\right)
\hat y_l(1-\hat y_l)
=-y_l(1-\hat y_l).
\]

Expand:

\[
=-y_l+y_l\hat y_l.
\tag{13.5}
\]

### The \(k\neq l\) terms

Using equation (11.3):

\[
\sum_{k\neq l}
\left(-\frac{y_k}{\hat y_k}\right)
(-\hat y_k\hat y_l).
\]

The negatives cancel and \(\hat y_k\) cancels:

\[
=
\sum_{k\neq l}y_k\hat y_l.
\]

Pull \(\hat y_l\) outside:

\[
=
\hat y_l\sum_{k\neq l}y_k.
\tag{13.6}
\]

Because the target is one-hot:

\[
\sum_k y_k=1,
\]

so:

\[
\sum_{k\neq l}y_k=1-y_l.
\tag{13.7}
\]

Hence the off-diagonal contribution is:

\[
\hat y_l(1-y_l).
\tag{13.8}
\]

### Combine the two parts

From equations (13.5) and (13.8):

\[
\frac{\partial L}{\partial z_l}
=
-y_l+y_l\hat y_l+\hat y_l-y_l\hat y_l.
\]

The middle terms cancel:

\[
\boxed{
\frac{\partial L}{\partial z_l}
=\hat y_l-y_l.
}
\tag{13.9}
\]

Since this is true for every output index \(l\):

\[
\boxed{
\frac{\partial L}{\partial z^{(2)}}
=\hat y-y.
}
\tag{13.10}
\]

This is the famous softmax + cross-entropy simplification.

---

# 14. Introducing \(\delta^{(2)}\)

We repeatedly use the quantity:

\[
\frac{\partial L}{\partial z^{(2)}}.
\]

So we give it a short name:

\[
\boxed{
\delta^{(2)}
\equiv
\frac{\partial L}{\partial z^{(2)}}
}
\tag{14.1}
\]

Then equation (13.10) becomes:

\[
\boxed{
\delta^{(2)}=\hat y-y.
}
\tag{14.2}
\]

The superscript \((2)\) means **layer 2**. It is not an exponent.

There is nothing special about the Greek letter delta; it is simply conventional notation for the gradient with respect to a layer's preactivation.

---

# 15. NOW \(\delta^{(2)}\) gives the final-layer gradients

This is the point of deriving \(\partial L/\partial z^{(2)}\).

Recall:

\[
z^{(2)}=W^{(2)}a^{(1)}+b^{(2)}.
\]

We now use \(\delta^{(2)}\) to obtain the gradients of \(W^{(2)}\) and \(b^{(2)}\).

---

## 15.1 Derivative needed for \(W^{(2)}\)

At scalar level:

\[
z^{(2)}_k
=
\sum_jW^{(2)}_{kj}a^{(1)}_j+b^{(2)}_k.
\]

We want:

\[
\frac{\partial L}{\partial W^{(2)}_{kj}}.
\]

The chain rule gives:

\[
\frac{\partial L}{\partial W^{(2)}_{kj}}
=
\frac{\partial L}{\partial z^{(2)}_k}
\frac{\partial z^{(2)}_k}{\partial W^{(2)}_{kj}}.
\tag{15.1}
\]

The first factor is:

\[
\delta^{(2)}_k.
\]

The second factor is:

\[
\frac{\partial z^{(2)}_k}{\partial W^{(2)}_{kj}}
=a^{(1)}_j.
\tag{15.2}
\]

Therefore:

\[
\boxed{
\frac{\partial L}{\partial W^{(2)}_{kj}}
=
\delta^{(2)}_k a^{(1)}_j.
}
\tag{15.3}
\]

This is an outer product:

\[
\boxed{
\frac{\partial L}{\partial W^{(2)}}
=
\delta^{(2)}(a^{(1)})^T.
}
\tag{15.4}
\]

Dimensions:

\[
(10\times1)(1\times300)=10\times300,
\]

exactly the shape of \(W^{(2)}\).

---

## 15.2 Derivative needed for \(b^{(2)}\)

Because:

\[
z^{(2)}_k=\cdots+b^{(2)}_k,
\]

we have:

\[
\frac{\partial z^{(2)}_k}{\partial b^{(2)}_k}=1.
\]

Therefore:

\[
\boxed{
\frac{\partial L}{\partial b^{(2)}}
=\delta^{(2)}.
}
\tag{15.5}
\]

So the final layer is now completely handled:

\[
\boxed{
\begin{aligned}
\frac{\partial L}{\partial W^{(2)}}
&=\delta^{(2)}(a^{(1)})^T,\\
\frac{\partial L}{\partial b^{(2)}}
&=\delta^{(2)}.
\end{aligned}}
\tag{15.6}
\]

---

# 16. But we still need the first-layer gradients

The first layer affects the loss through:

\[
W^{(1)},b^{(1)}
\rightarrow
z^{(1)}
\rightarrow
a^{(1)}
\rightarrow
z^{(2)}
\rightarrow
\hat y
\rightarrow
L.
\]

So we need to continue backward from \(z^{(2)}\) into \(a^{(1)}\).

The next required quantity is:

\[
\boxed{
\frac{\partial L}{\partial a^{(1)}}.
}
\tag{16.1}
\]

Why?

Because \(a^{(1)}\) is what the sigmoid produces, and the sigmoid's input is \(z^{(1)}\). We cannot reach \(z^{(1)}\) until we know the gradient with respect to \(a^{(1)}\).

---

# 17. Propagate from \(z^{(2)}\) back to \(a^{(1)}\)

Recall:

\[
z^{(2)}=W^{(2)}a^{(1)}+b^{(2)}.
\]

At scalar level:

\[
z^{(2)}_k
=
\sum_jW^{(2)}_{kj}a^{(1)}_j+b^{(2)}_k.
\]

For hidden unit \(j\):

\[
\frac{\partial z^{(2)}_k}{\partial a^{(1)}_j}
=W^{(2)}_{kj}.
\tag{17.1}
\]

Since \(a^{(1)}_j\) affects every output logit \(z^{(2)}_k\), the chain rule gives:

\[
\frac{\partial L}{\partial a^{(1)}_j}
=
\sum_k
\frac{\partial L}{\partial z^{(2)}_k}
\frac{\partial z^{(2)}_k}{\partial a^{(1)}_j}.
\tag{17.2}
\]

Substitute:

\[
\frac{\partial L}{\partial a^{(1)}_j}
=
\sum_k
\delta^{(2)}_kW^{(2)}_{kj}.
\tag{17.3}
\]

In matrix form:

\[
\boxed{
\frac{\partial L}{\partial a^{(1)}}
=(W^{(2)})^T\delta^{(2)}.
}
\tag{17.4}
\]

Dimensions:

\[
(300\times10)(10\times1)=300\times1.
\]

This is why the transpose appears: we are propagating a 10-dimensional gradient back into 300 hidden units.

---

# 18. Now we must cross the sigmoid

We have:

\[
a^{(1)}=\sigma(z^{(1)}).
\]

To reach \(z^{(1)}\), we need:

\[
\boxed{
\frac{\partial a^{(1)}}{\partial z^{(1)}}.
}
\tag{18.1}
\]

Then the chain rule gives:

\[
\boxed{
\frac{\partial L}{\partial z^{(1)}}
=
\frac{\partial L}{\partial a^{(1)}}
\odot
\frac{\partial a^{(1)}}{\partial z^{(1)}}.
}
\tag{18.2}
\]

Because sigmoid is applied elementwise, this becomes elementwise multiplication.

---

# 19. Derivative of sigmoid

Start with:

\[
\sigma(z)=\frac{1}{1+e^{-z}}.
\]

Write:

\[
\sigma(z)=(1+e^{-z})^{-1}.
\]

Differentiate:

\[
\frac{d\sigma}{dz}
=
-(1+e^{-z})^{-2}(-e^{-z}).
\]

Therefore:

\[
\frac{d\sigma}{dz}
=
\frac{e^{-z}}{(1+e^{-z})^2}.
\tag{19.1}
\]

Now:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

and:

\[
1-\sigma(z)
=
1-\frac{1}{1+e^{-z}}
=
\frac{e^{-z}}{1+e^{-z}}.
\tag{19.2}
\]

Multiply:

\[
\sigma(z)(1-\sigma(z))
=
\frac{e^{-z}}{(1+e^{-z})^2}.
\]

Therefore:

\[
\boxed{
\sigma'(z)=\sigma(z)(1-\sigma(z)).
}
\tag{19.3}
\]

Since:

\[
a^{(1)}=\sigma(z^{(1)}),
\]

we can write:

\[
\boxed{
\frac{\partial a^{(1)}_j}{\partial z^{(1)}_j}
=a^{(1)}_j(1-a^{(1)}_j).
}
\tag{19.4}
\]

---

# 20. Hidden-layer delta

We now have everything required to calculate the loss gradient with respect to \(z^{(1)}\).

Define:

\[
\boxed{
\delta^{(1)}
\equiv
\frac{\partial L}{\partial z^{(1)}}.
}
\tag{20.1}
\]

From equations (17.4) and (19.4):

\[
\delta^{(1)}
=
\left[(W^{(2)})^T\delta^{(2)}\right]
\odot
\left[a^{(1)}\odot(1-a^{(1)})\right].
\]

Thus:

\[
\boxed{
\delta^{(1)}
=
(W^{(2)})^T\delta^{(2)}
\odot
 a^{(1)}
\odot
(1-a^{(1)}).
}
\tag{20.2}
\]

Now we have reached the quantity directly produced by the first affine layer:

\[
W^{(1)},b^{(1)}
ightarrow z^{(1)}.
\]

So we can finally calculate the first-layer parameter gradients.

---

# 21. Gradient of the first-layer weights

Recall:

\[
z^{(1)}=W^{(1)}x+b^{(1)}.
\]

At scalar level:

\[
z^{(1)}_j
=
\sum_iW^{(1)}_{ji}x_i+b^{(1)}_j.
\]

Therefore:

\[
\frac{\partial z^{(1)}_j}{\partial W^{(1)}_{ji}}
=x_i.
\tag{21.1}
\]

By the chain rule:

\[
\frac{\partial L}{\partial W^{(1)}_{ji}}
=
\frac{\partial L}{\partial z^{(1)}_j}
\frac{\partial z^{(1)}_j}{\partial W^{(1)}_{ji}}.
\]

Hence:

\[
\boxed{
\frac{\partial L}{\partial W^{(1)}_{ji}}
=
\delta^{(1)}_j x_i.
}
\tag{21.2}
\]

In matrix form:

\[
\boxed{
\frac{\partial L}{\partial W^{(1)}}
=\delta^{(1)}x^T.
}
\tag{21.3}
\]

Dimensions:

\[
(300\times1)(1\times784)=300\times784,
\]

exactly the shape of \(W^{(1)}\).

---

# 22. Gradient of the first-layer bias

This step is deceptively short in the final formula, but it is important to understand **why** it is true. The key idea is the same chain-rule pattern used for every parameter gradient:

> Find the gradient with respect to the quantity the parameter directly produces, then multiply by the local derivative of that quantity with respect to the parameter.

Here the parameter is the first-layer bias:

\[
b^{(1)}_j.
\]

It directly appears in the first-layer preactivation:

\[
z^{(1)}_j
=\sum_{i=1}^{784}W^{(1)}_{ji}x_i+b^{(1)}_j.
\tag{22.1}
\]

Recall that earlier in the backward pass we defined:

\[
\boxed{
\delta^{(1)}_j
\equiv
\frac{\partial L}{\partial z^{(1)}_j}.
}
\tag{22.2}
\]

So by the time we reach the bias, the upstream gradient we need is already known:

\[
\frac{\partial L}{\partial z^{(1)}_j}=\delta^{(1)}_j.
\tag{22.3}
\]

Now ask: **how does changing $b^{(1)}_j$ change $z^{(1)}_j$?**

From (22.1), $b^{(1)}_j$ appears as a plain additive term. Therefore:

\[
\boxed{
\frac{\partial z^{(1)}_j}{\partial b^{(1)}_j}=1.
}
\tag{22.4}
\]

The chain rule now connects these two pieces:

\[
\frac{\partial L}{\partial b^{(1)}_j}
=
\frac{\partial L}{\partial z^{(1)}_j}
\frac{\partial z^{(1)}_j}{\partial b^{(1)}_j}.
\tag{22.5}
\]

Substitute (22.3) and (22.4):

\[
\frac{\partial L}{\partial b^{(1)}_j}
=\delta^{(1)}_j\cdot1.
\tag{22.6}
\]

Therefore:

\[
\boxed{
\frac{\partial L}{\partial b^{(1)}_j}=\delta^{(1)}_j.
}
\tag{22.7}
\]

## 22.1 Why do the other preactivations not contribute?

A fully explicit multivariable chain rule makes the structure even clearer. We are differentiating the total loss with respect to one particular bias $b^{(1)}_j$. The loss depends on all 300 first-layer preactivations, so formally:

\[
\frac{\partial L}{\partial b^{(1)}_j}
=
\sum_{r=1}^{300}
\frac{\partial L}{\partial z^{(1)}_r}
\frac{\partial z^{(1)}_r}{\partial b^{(1)}_j}.
\tag{22.8}
\]

Now consider $r\neq j$. The preactivation $z^{(1)}_r$ contains $b^{(1)}_r$, not $b^{(1)}_j$, so:

\[
\frac{\partial z^{(1)}_r}{\partial b^{(1)}_j}=0,\qquad r\neq j.
\tag{22.9}
\]

For $r=j$:

\[
\frac{\partial z^{(1)}_j}{\partial b^{(1)}_j}=1.
\tag{22.10}
\]

This can be written compactly using the Kronecker delta:

\[
\boxed{
\frac{\partial z^{(1)}_r}{\partial b^{(1)}_j}=\delta_{rj}.
}
\tag{22.11}
\]

Substituting into (22.8):

\[
\frac{\partial L}{\partial b^{(1)}_j}
=
\sum_{r=1}^{300}
\delta^{(1)}_r\delta_{rj}.
\tag{22.12}
\]

The Kronecker delta kills every term except $r=j$:

\[
\frac{\partial L}{\partial b^{(1)}_j}
=\delta^{(1)}_j.
\tag{22.13}
\]

So the short derivation was not skipping any mysterious step; it was simply using the fact that each bias affects exactly one corresponding preactivation directly.

## 22.2 Why the vector equation is just the componentwise result

We have shown, for every hidden neuron $j=1,\ldots,300$, that:

\[
\frac{\partial L}{\partial b^{(1)}_j}=\delta^{(1)}_j.
\tag{22.14}
\]

Therefore the entire gradient vector is:

\[
\boxed{
\frac{\partial L}{\partial b^{(1)}}=\delta^{(1)}.
}
\tag{22.15}
\]

This is why the bias gradient is so simple: the local derivative of an additive bias is 1.

## 22.3 Compare with the first-layer weight gradient

The weight derivation follows exactly the same chain-rule template. From:

\[
z^{(1)}_j=\sum_iW^{(1)}_{ji}x_i+b^{(1)}_j,
\]

we have:

\[
\frac{\partial z^{(1)}_j}{\partial W^{(1)}_{ji}}=x_i.
\tag{22.16}
\]

Therefore:

\[
\frac{\partial L}{\partial W^{(1)}_{ji}}
=
\frac{\partial L}{\partial z^{(1)}_j}
\frac{\partial z^{(1)}_j}{\partial W^{(1)}_{ji}}
=\delta^{(1)}_j x_i.
\tag{22.17}
\]

Compare this with the bias:

\[
\frac{\partial L}{\partial b^{(1)}_j}
=
\frac{\partial L}{\partial z^{(1)}_j}
\frac{\partial z^{(1)}_j}{\partial b^{(1)}_j}
=\delta^{(1)}_j\cdot1.
\tag{22.18}
\]

So the general pattern for an affine layer

\[
z=Wx+b
\]

is:

\[
\boxed{
\frac{\partial L}{\partial W}=\delta x^T
}
\qquad
\boxed{
\frac{\partial L}{\partial b}=\delta.
}
\tag{22.19}
\]

The bias has no extra input factor because its local derivative is simply 1.

**At this point all four parameter gradients have been obtained.**

---

# 23. The full derivation as one dependency tree

This is the main reference diagram.

```text
FORWARD

x
│
▼
z¹ = W¹x + b¹
│
│  sigmoid
▼
a¹ = σ(z¹)
│
▼
z² = W²a¹ + b²
│
│  softmax
▼
ŷ
│
│  cross-entropy
▼
L

BACKWARD

L
│
▼
∂L/∂ŷ
│
│  combine with ∂ŷ/∂z²
▼
δ² = ∂L/∂z² = ŷ - y
│
├─────────────────────┐
│                     │
│                     │
▼                     ▼
∂L/∂W²              ∂L/∂b²
│                     │
│                     │
└──────────┬──────────┘
           │
           │ update final layer
           │
           ▼
       ∂L/∂a¹
           │
           │  multiply by sigmoid derivative
           ▼
δ¹ = ∂L/∂z¹
│
├─────────────────────┐
│                     │
▼                     ▼
∂L/∂W¹              ∂L/∂b¹
│                     │
│                     │
└──────────┬──────────┘
           │
           ▼
      update first layer
```

---

# 24. The final equations only

## Forward pass

\[
\boxed{z^{(1)}=W^{(1)}x+b^{(1)}}
\tag{24.1}
\]

\[
\boxed{a^{(1)}=\sigma(z^{(1)})}
\tag{24.2}
\]

\[
\boxed{z^{(2)}=W^{(2)}a^{(1)}+b^{(2)}}
\tag{24.3}
\]

\[
\boxed{\hat y=\operatorname{softmax}(z^{(2)})}
\tag{24.4}
\]

\[
\boxed{L=-\sum_k y_k\log\hat y_k}
\tag{24.5}
\]

## Backward pass

\[
\boxed{
\frac{\partial L}{\partial\hat y_k}
=-\frac{y_k}{\hat y_k}
}
\tag{24.6}
\]

\[
\boxed{
\frac{\partial\hat y_k}{\partial z_l}
=
\begin{cases}
\hat y_k(1-\hat y_k),&k=l,\\[4pt]
-\hat y_k\hat y_l,&k\neq l
\end{cases}
}
\tag{24.7}
\]

\[
\boxed{
\delta^{(2)}
\equiv
\frac{\partial L}{\partial z^{(2)}}
=\hat y-y
}
\tag{24.8}
\]

\[
\boxed{
\frac{\partial L}{\partial W^{(2)}}
=\delta^{(2)}(a^{(1)})^T
}
\tag{24.9}
\]

\[
\boxed{
\frac{\partial L}{\partial b^{(2)}}
=\delta^{(2)}
}
\tag{24.10}
\]

\[
\boxed{
\frac{\partial L}{\partial a^{(1)}}
=(W^{(2)})^T\delta^{(2)}
}
\tag{24.11}
\]

\[
\boxed{
\sigma'(z)=\sigma(z)(1-\sigma(z))
}
\tag{24.12}
\]

\[
\boxed{
\delta^{(1)}
\equiv
\frac{\partial L}{\partial z^{(1)}}
=
(W^{(2)})^T\delta^{(2)}
\odot a^{(1)}
\odot(1-a^{(1)})
}
\tag{24.13}
\]

\[
\boxed{
\frac{\partial L}{\partial W^{(1)}}
=\delta^{(1)}x^T
}
\tag{24.14}
\]

\[
\boxed{
\frac{\partial L}{\partial b^{(1)}}
=\delta^{(1)}
}
\tag{24.15}
\]

---

# 25. Why every intermediate exists

This is the most useful table for memory refresh.

| Quantity | What it means | Why we calculate it | Where it goes |
|---|---|---|---|
| \(z^{(1)}\) | Hidden preactivation | It is the input to sigmoid | Produces \(a^{(1)}\) |
| \(a^{(1)}\) | Hidden activation | It is the input to layer 2 | Produces \(z^{(2)}\), and appears in \(dW^{(2)}\) |
| \(z^{(2)}\) | Output logits | It is the input to softmax | Produces \(\hat y\) |
| \(\hat y\) | Predicted probabilities | It enters the loss | Produces \(L\) |
| \(L\) | Scalar error | Training minimizes it | Starting point for backprop |
| \(\partial L/\partial\hat y\) | Loss sensitivity to probabilities | Needed to cross softmax | Combines with softmax Jacobian |
| \(\partial\hat y/\partial z^{(2)}\) | Softmax sensitivity | Needed to cross softmax | Produces \(\partial L/\partial z^{(2)}\) |
| \(\delta^{(2)}\) | Loss sensitivity to output logits | Directly unlocks output-layer gradients | \(dW^{(2)},db^{(2)}\), and \(d a^{(1)}\) |
| \(\partial L/\partial a^{(1)}\) | Loss sensitivity to hidden activations | Needed to cross sigmoid | Combines with sigmoid derivative |
| \(\partial a^{(1)}/\partial z^{(1)}\) | Sigmoid local derivative | Needed to cross sigmoid | Produces \(\delta^{(1)}\) |
| \(\delta^{(1)}\) | Loss sensitivity to hidden preactivations | Directly unlocks first-layer gradients | \(dW^{(1)},db^{(1)}\) |
| \(dW^{(1)}\) | Loss sensitivity to first weights | Needed for learning | Update \(W^{(1)}\) |
| \(db^{(1)}\) | Loss sensitivity to first biases | Needed for learning | Update \(b^{(1)}\) |
| \(dW^{(2)}\) | Loss sensitivity to second weights | Needed for learning | Update \(W^{(2)}\) |
| \(db^{(2)}\) | Loss sensitivity to second biases | Needed for learning | Update \(b^{(2)}\) |

---

# 26. Scalar interpretation of each final gradient

Matrices can hide the meaning. Pick one parameter at a time.

## 26.1 One output-layer weight

Take \(W^{(2)}_{kj}\).

It connects hidden neuron \(j\) to output neuron \(k\).

Its gradient is:

\[
\boxed{
\frac{\partial L}{\partial W^{(2)}_{kj}}
=
\underbrace{\delta^{(2)}_k}_{\text{how much output }k\text{ wants to change}}
\underbrace{a^{(1)}_j}_{\text{how active hidden }j\text{ was}}
}
\tag{26.1}
\]

---

## 26.2 One first-layer weight

Take \(W^{(1)}_{ji}\).

It connects input \(i\) to hidden neuron \(j\).

Its gradient is:

\[
\boxed{
\frac{\partial L}{\partial W^{(1)}_{ji}}
=
\underbrace{\delta^{(1)}_j}_{\text{how much hidden preactivation affects loss}}
\underbrace{x_i}_{\text{input feeding that connection}}
}
\tag{26.2}
\]

The pattern is:

\[
\boxed{
\text{gradient of a weight}
=
\text{gradient at its output}
\times
\text{value entering that weight}
}
\tag{26.3}
\]

---

# 27. Why transpose appears in backward propagation

Forward through the second layer:

\[
z^{(2)}=W^{(2)}a^{(1)}.
\]

The matrix \(W^{(2)}\) maps:

\[
300\rightarrow10.
\]

Backward we need to map:

\[
10\rightarrow300.
\]

Therefore the matrix must have shape:

\[
300\times10,
\]

which is exactly:

\[
(W^{(2)})^T.
\]

More importantly, the transpose is not chosen by intuition; it follows from the scalar equation:

\[
\frac{\partial L}{\partial a^{(1)}_j}
=
\sum_k
\delta^{(2)}_kW^{(2)}_{kj}.
\]

That summation is exactly the component form of:

\[
\boxed{
\frac{\partial L}{\partial a^{(1)}}
=(W^{(2)})^T\delta^{(2)}.
}
\tag{27.1}
\]

---

# 28. Why outer products appear for weight gradients

For the second layer:

\[
\delta^{(2)}\in\mathbb R^{10},
\qquad
 a^{(1)}\in\mathbb R^{300}.
\]

We want a matrix of shape:

\[
10\times300.
\]

Therefore:

\[
\delta^{(2)}(a^{(1)})^T
\]

has exactly that shape:

\[
(10\times1)(1\times300)=10\times300.
\]

Likewise:

\[
\delta^{(1)}x^T
\]

has shape:

\[
(300\times1)(1\times784)=300\times784.
\]

The matrix formulas are simply the compact version of the scalar formulas.

---

# 29. Another way to view \(\delta\)

For any layer, think of:

\[
\boxed{
\delta^{(l)}=\frac{\partial L}{\partial z^{(l)}}
}
\tag{29.1}
\]

It is the gradient of the loss with respect to the layer's **preactivation**.

Why is this particular quantity convenient?

Because the parameters directly produce \(z^{(l)}\):

\[
W^{(l)},b^{(l)}\rightarrow z^{(l)}.
\]

Once \(\delta^{(l)}\) is known, the parameter gradients follow immediately.

For the second layer:

\[
\delta^{(2)}\rightarrow dW^{(2)},db^{(2)}.
\]

For the first layer:

\[
\delta^{(1)}\rightarrow dW^{(1)},db^{(1)}.
\]

So delta is best thought of as the **error signal sitting at a layer's preactivation**.

---

# 30. Why we do NOT update sigmoid or softmax

Sigmoid and softmax are functions, not trainable parameter matrices in this network.

We do not do something like:

\[
\sigma\leftarrow\sigma-\eta(\text{gradient}).
\]

Instead, we differentiate through them so that the gradient can reach the actual parameters:

\[
W^{(2)},b^{(2)},W^{(1)},b^{(1)}.
\]

So the accurate mental model is:

\[
\boxed{
\text{forward through sigmoid/softmax}
\quad\text{and}\quad
\text{backward through sigmoid/softmax}
}
\]

while:

\[
\boxed{
\text{weights and biases are updated.}
}
\]

---

# 31. Why an activation is needed

Suppose there were no sigmoid between the two affine layers.

We would have:

\[
z^{(1)}=W^{(1)}x+b^{(1)}
\]

and:

\[
z^{(2)}=W^{(2)}z^{(1)}+b^{(2)}.
\]

Substitute:

\[
z^{(2)}
=W^{(2)}(W^{(1)}x+b^{(1)})+b^{(2)}.
\]

So:

\[
z^{(2)}
=W^{(2)}W^{(1)}x+W^{(2)}b^{(1)}+b^{(2)}.
\]

Let:

\[
W'=W^{(2)}W^{(1)},
\qquad
b'=W^{(2)}b^{(1)}+b^{(2)}.
\]

Then:

\[
z^{(2)}=W'x+b'.
\]

Thus several affine layers without a nonlinearity collapse into a single affine transformation.

The activation function is what makes the composition nonlinear and therefore more expressive.

---

# 32. Numerical stability of softmax

The mathematical softmax formula is:

\[
\hat y_k=
\frac{e^{z_k}}{\sum_j e^{z_j}}.
\]

For large positive logits, the exponentials can overflow numerically.

Let:

\[
m=\max_j z_j.
\]

Then calculate:

\[
\boxed{
\hat y_k
=
\frac{e^{z_k-m}}
{\sum_j e^{z_j-m}}
}
\tag{32.1}
\]

because:

\[
\frac{e^{z_k-m}}{\sum_j e^{z_j-m}}
=
\frac{e^{z_k}e^{-m}}
{\sum_j e^{z_j}e^{-m}}
=
\frac{e^{z_k}}{\sum_j e^{z_j}}.
\tag{32.2}
\]

So this is mathematically identical but numerically safer.

For an actual implementation, this is the version to use.

---

# 33. Parameter count

First layer weights:

\[
784\times300=235200.
\]

First layer biases:

\[
300.
\]

Second layer weights:

\[
300\times10=3000.
\]

Second layer biases:

\[
10.
\]

Total:

\[
\boxed{
235200+300+3000+10=238510
}
\tag{33.1}
\]

So the network has 238,510 trainable parameters.

---

# 34. Why backpropagation is efficient

A network with 238,510 parameters could in principle be differentiated parameter-by-parameter, but that would repeatedly redo the same calculations.

Backpropagation computes reusable intermediate gradients such as:

\[
\delta^{(2)}
\]

and:

\[
\delta^{(1)}
\]

and then reuses them to obtain many parameter gradients.

That is the computational advantage of the chain rule organized as a backward pass.

---

# 35. The most important conceptual pattern

Each layer has the structure:

\[
\boxed{
\text{parameters}
\rightarrow
\text{preactivation }z
\rightarrow
\text{activation }a
}
\tag{35.1}
\]

During backpropagation, we reverse it:

\[
\boxed{
\text{loss gradient}
\rightarrow
\text{activation gradient}
\rightarrow
\text{preactivation gradient}
\rightarrow
\text{parameter gradients}
}
\tag{35.2}
\]

The recurring operation is:

\[
\boxed{
\text{upstream gradient}
\times
\text{local derivative}
=
\text{gradient passed to the left}
}
\tag{35.3}
\]

When a node has multiple downstream paths, the gradient contributions from those paths are added.

That is the general computational-graph view of backpropagation.

---

# 36. Dimension checks — a powerful debugging tool

The matrix dimensions should always work.

## Forward

\[
W^{(1)}x:
\qquad
(300\times784)(784\times1)=300\times1.
\]

\[
W^{(2)}a^{(1)}:
\qquad
(10\times300)(300\times1)=10\times1.
\]

## Backward

\[
(W^{(2)})^T\delta^{(2)}:
\qquad
(300\times10)(10\times1)=300\times1.
\]

\[
\delta^{(2)}(a^{(1)})^T:
\qquad
(10\times1)(1\times300)=10\times300.
\]

\[
\delta^{(1)}x^T:
\qquad
(300\times1)(1\times784)=300\times784.
\]

Thus:

\[
\operatorname{shape}(dW^{(2)})=
\operatorname{shape}(W^{(2)}),
\]

and:

\[
\operatorname{shape}(dW^{(1)})=
\operatorname{shape}(W^{(1)}).
\]

A gradient for a parameter must have exactly the same shape as that parameter.

---

# 37. Mapping the mathematics directly to a C implementation

For one example, the core arrays are conceptually:

```text
x      [784]
z1     [300]
a1     [300]
z2     [10]
yhat   [10]

d1     [300]
d2     [10]

W1     [300][784]
b1     [300]
W2     [10][300]
b2     [10]

dW1    [300][784]
db1    [300]
dW2    [10][300]
db2    [10]
```

Forward equation:

\[
z^{(1)}_j=
\sum_iW^{(1)}_{ji}x_i+b^{(1)}_j
\]

maps directly to:

```text
for (j = 0; j < 300; ++j) {
    z1[j] = b1[j];
    for (i = 0; i < 784; ++i) {
        z1[j] += W1[j][i] * x[i];
    }
}
```

Hidden activation:

```text
for (j = 0; j < 300; ++j) {
    a1[j] = sigmoid(z1[j]);
}
```

Second affine layer:

```text
for (k = 0; k < 10; ++k) {
    z2[k] = b2[k];
    for (j = 0; j < 300; ++j) {
        z2[k] += W2[k][j] * a1[j];
    }
}
```

The stable softmax uses:

\[
m=\max_k z^{(2)}_k,
\]

then:

\[
e_k=e^{z^{(2)}_k-m}
\]

and normalizes by the sum of the \(e_k\)'s.

The output delta is simply:

```text
d2[k] = yhat[k] - y[k];
```

Final-layer weight gradient:

```text
for (k = 0; k < 10; ++k) {
    for (j = 0; j < 300; ++j) {
        dW2[k][j] = d2[k] * a1[j];
    }
}
```

Final-layer bias gradient:

```text
db2[k] = d2[k];
```

Hidden gradient:

\[
\frac{\partial L}{\partial a^{(1)}}
=(W^{(2)})^T d2,
\]

then:

```text
for (j = 0; j < 300; ++j) {
    double da1 = 0.0;
    for (k = 0; k < 10; ++k) {
        da1 += W2[k][j] * d2[k];
    }
    d1[j] = da1 * a1[j] * (1.0 - a1[j]);
}
```

First-layer weight gradient:

```text
for (j = 0; j < 300; ++j) {
    for (i = 0; i < 784; ++i) {
        dW1[j][i] = d1[j] * x[i];
    }
}
```

First-layer bias gradient:

```text
db1[j] = d1[j];
```

Then update every parameter using:

\[
\theta\leftarrow\theta-\eta\,d\theta.
\]

The code is not a different algorithm from the mathematics. The loops are simply the scalar equations written out explicitly.

---

# 38. One complete scalar chain for one parameter

To see the entire concept at the lowest useful level, take one output-layer weight \(W^{(2)}_{kj}\).

Its complete path is:

\[
W^{(2)}_{kj}
\rightarrow
z^{(2)}_k
\rightarrow
\hat y_1,\ldots,\hat y_{10}
\rightarrow
L.
\]

Therefore its total derivative is the sum of its effects through all output probabilities:

\[
\boxed{
\frac{\partial L}{\partial W^{(2)}_{kj}}
=
\sum_r
\frac{\partial L}{\partial\hat y_r}
\frac{\partial\hat y_r}{\partial z^{(2)}_k}
\frac{\partial z^{(2)}_k}{\partial W^{(2)}_{kj}}.
}
\tag{38.1}
\]

The first two factors combine into:

\[
\frac{\partial L}{\partial z^{(2)}_k}
=\hat y_k-y_k.
\]

The final local derivative is:

\[
\frac{\partial z^{(2)}_k}{\partial W^{(2)}_{kj}}
=a^{(1)}_j.
\]

Thus:

\[
\boxed{
\frac{\partial L}{\partial W^{(2)}_{kj}}
=(\hat y_k-y_k)a^{(1)}_j.
}
\tag{38.2}
\]

This example shows the basic philosophy of backpropagation:

> Walk backward through the computational path, multiplying by local derivatives, and add contributions when a variable has multiple downstream paths.

---

# 39. What if we also wanted the input gradient?

The input is not normally updated, but mathematically its gradient exists.

Once we have \(\delta^{(1)}\):

\[
z^{(1)}=W^{(1)}x+b^{(1)}.
\]

Therefore:

\[
\boxed{
\frac{\partial L}{\partial x}
=(W^{(1)})^T\delta^{(1)}.
}
\tag{39.1}
\]

This is useful conceptually because it shows that backprop can continue all the way to the input.

---

# 40. What the symbols mean — quick glossary

### \(x\)
The input vector.

### \(W^{(1)}\), \(b^{(1)}\)
Weights and bias of the first affine layer.

### \(z^{(1)}\)
Hidden-layer preactivation: weighted input plus bias.

### \(a^{(1)}\)
Hidden-layer activation after sigmoid.

### \(W^{(2)}\), \(b^{(2)}\)
Weights and bias of the output affine layer.

### \(z^{(2)}\)
Output logits before softmax.

### \(\hat y\)
Predicted probability vector.

### \(y\)
Target vector.

### \(L\)
Scalar loss.

### \(\delta^{(2)}\)
A name for:

\[
\frac{\partial L}{\partial z^{(2)}}.
\]

### \(\delta^{(1)}\)
A name for:

\[
\frac{\partial L}{\partial z^{(1)}}.
\]

### \(\odot\)
Elementwise multiplication.

### \(\delta_{ij}\)
Kronecker delta: 1 if \(i=j\), otherwise 0.

### Jacobian
The matrix of all partial derivatives between a vector output and vector input.

---

# 41. The complete algorithm in dependency order

This is the ordering to use when mentally reconstructing or implementing the network.

## Step A — Forward

1. Compute:
   \[
   z^{(1)}=W^{(1)}x+b^{(1)}.
   \]

2. Compute:
   \[
   a^{(1)}=\sigma(z^{(1)}).
   \]

3. Compute:
   \[
   z^{(2)}=W^{(2)}a^{(1)}+b^{(2)}.
   \]

4. Compute stable softmax:
   \[
   \hat y=\operatorname{softmax}(z^{(2)}).
   \]

5. Compute loss:
   \[
   L=-\sum_k y_k\log\hat y_k.
   \]

## Step B — Backward through output

6. Compute:
   \[
   \delta^{(2)}=\hat y-y.
   \]

7. Use it to compute:
   \[
   dW^{(2)}=\delta^{(2)}(a^{(1)})^T.
   \]

8. Use it to compute:
   \[
   db^{(2)}=\delta^{(2)}.
   \]

## Step C — Continue backward into hidden layer

9. Propagate through the second affine layer:
   \[
   \frac{\partial L}{\partial a^{(1)}}
   =(W^{(2)})^T\delta^{(2)}.
   \]

10. Cross sigmoid:
   \[
   \delta^{(1)}
   =
   \frac{\partial L}{\partial a^{(1)}}
   \odot
   a^{(1)}
   \odot
   (1-a^{(1)}).
   \]

11. Compute:
   \[
   dW^{(1)}=\delta^{(1)}x^T.
   \]

12. Compute:
   \[
   db^{(1)}=\delta^{(1)}.
   \]

## Step D — Update parameters

13. Apply:
   \[
   W^{(1)}\leftarrow W^{(1)}-\eta dW^{(1)}.
   \]

14. Apply:
   \[
   b^{(1)}\leftarrow b^{(1)}-\eta db^{(1)}.
   \]

15. Apply:
   \[
   W^{(2)}\leftarrow W^{(2)}-\eta dW^{(2)}.
   \]

16. Apply:
   \[
   b^{(2)}\leftarrow b^{(2)}-\eta db^{(2)}.
   \]

That is the complete training step for one example.

---

# 42. Ultra-short reconstruction sheet

When you come back years later, start here.

## What do I need?

\[
\boxed{dW^{(1)},db^{(1)},dW^{(2)},db^{(2)}}
\]

## Where do parameter gradients come from?

A parameter first affects its layer preactivation \(z\), so first obtain:

\[
\boxed{\delta^{(l)}=\frac{\partial L}{\partial z^{(l)}}.}
\]

## Output layer

\[
\hat y=\operatorname{softmax}(z^{(2)})
\]

\[
L=-\sum_k y_k\log\hat y_k
\]

\[
\boxed{\delta^{(2)}=\hat y-y}
\]

Then:

\[
\boxed{dW^{(2)}=\delta^{(2)}(a^{(1)})^T}
\]

\[
\boxed{db^{(2)}=\delta^{(2)}}
\]

## Propagate backward

\[
\boxed{
\frac{\partial L}{\partial a^{(1)}}
=(W^{(2)})^T\delta^{(2)}
}
\]

## Hidden sigmoid

\[
\boxed{a^{(1)}=\sigma(z^{(1)})}
\]

\[
\boxed{\sigma'(z)=\sigma(z)(1-\sigma(z))}
\]

so:

\[
\boxed{
\delta^{(1)}
=(W^{(2)})^T\delta^{(2)}
\odot a^{(1)}
\odot(1-a^{(1)})
}
\]

Then:

\[
\boxed{dW^{(1)}=\delta^{(1)}x^T}
\]

\[
\boxed{db^{(1)}=\delta^{(1)}}
\]

Finally:

\[
\boxed{\theta\leftarrow\theta-\eta\,d\theta.}
\]

---

# 43. The one sentence that ties the whole derivation together

> **Start from the loss, walk backward through the computational graph using the chain rule, and whenever you reach the preactivation of a layer, you now have exactly the gradient needed to calculate that layer's weight and bias gradients.**

For this network, that gives the chain:

\[
\boxed{
L
\rightarrow
\hat y
\rightarrow
z^{(2)}
\rightarrow
\left\{
W^{(2)},b^{(2)},a^{(1)}
\right\}
\rightarrow
z^{(1)}
\rightarrow
\left\{
W^{(1)},b^{(1)}
\right\}
}
\tag{43.1}
\]

and every detailed derivative in this document exists only because it is the next link required by that chain.
