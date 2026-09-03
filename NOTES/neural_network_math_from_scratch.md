# Neural Network Math From First Principles

## A self-contained derivation of a \(784 \rightarrow 300 \rightarrow 10\) network

This document reconstructs the mathematics of a small feed-forward neural network from first principles.

The network is:

\[
784 \rightarrow 300 \rightarrow 10
\]

with:

- a sigmoid activation in the hidden layer,
- a softmax activation in the output layer,
- cross-entropy loss,
- gradient descent for learning.

The goal is not just to memorize backpropagation formulas. The goal is to understand **why every intermediate quantity exists, where it comes from, and where it goes next**.

The derivation is written for **one training example at a time**. Mini-batches and full-batch training are discussed later as extensions.

---

# 1. The big picture

A neural network is a composition of functions.

For our network, the forward computation is:

\[
 x
 \rightarrow z^{(1)}
 \rightarrow a^{(1)}
 \rightarrow z^{(2)}
 \rightarrow \hat y
 \rightarrow L
\]

where:

- \(x\): input vector,
- \(z^{(1)}\): hidden-layer pre-activation,
- \(a^{(1)}\): hidden-layer activation,
- \(z^{(2)}\): output-layer pre-activation / logits,
- \(\hat y\): predicted probabilities,
- \(L\): scalar loss.

Training reverses this dependency:

\[
L
\rightarrow \hat y
\rightarrow z^{(2)}
\rightarrow a^{(1)}
\rightarrow z^{(1)}
\rightarrow W^{(1)},b^{(1)},W^{(2)},b^{(2)}.
\]

The central mathematical task is therefore:

> Compute how much the loss changes when each parameter changes.

That means finding:

\[
\boxed{
\frac{\partial L}{\partial W^{(1)}},\quad
\frac{\partial L}{\partial b^{(1)}},\quad
\frac{\partial L}{\partial W^{(2)}},\quad
\frac{\partial L}{\partial b^{(2)}}
}
\tag{1}
\]

Once those gradients are known, gradient descent supplies the parameter updates:

\[
\boxed{
\theta_{\text{new}}
=
\theta-
\eta\frac{\partial L}{\partial\theta}
}
\tag{2}
\]

where \(\eta>0\) is the learning rate.

**Correction:** the actual equation is simply

\[
\boxed{
\theta_{\text{new}}
=
\theta-
\eta\frac{\partial L}{\partial\theta}
}
\tag{2}
\]

No other mysterious operation is involved. Training is repeated computation of the gradients followed by parameter updates.

---

# 2. Network dimensions and notation

We use column vectors throughout.

## 2.1 Input

There are 784 input values:

\[
\boxed{x\in\mathbb R^{784}}
\tag{3}
\]

For a flattened \(28\times28\) image, the 784 entries are the pixels.

We write:

\[
x=
\begin{bmatrix}
 x_1\\
 x_2\\
 \vdots\\
 x_{784}
\end{bmatrix}.
\]

---

## 2.2 Hidden-layer parameters

The hidden layer has 300 neurons.

Therefore the first weight matrix must map a 784-dimensional vector to a 300-dimensional vector:

\[
\boxed{W^{(1)}\in\mathbb R^{300\times784}}
\tag{4}
\]

The bias has one value per hidden neuron:

\[
\boxed{b^{(1)}\in\mathbb R^{300}}
\tag{5}
\]

---

## 2.3 Output-layer parameters

There are 10 output classes.

The second weight matrix maps 300 hidden activations to 10 output logits:

\[
\boxed{W^{(2)}\in\mathbb R^{10\times300}}
\tag{6}
\]

and:

\[
\boxed{b^{(2)}\in\mathbb R^{10}}
\tag{7}
\]

---

## 2.4 Target

The correct answer is represented by a 10-dimensional one-hot vector:

\[
\boxed{y\in\mathbb R^{10}}
\tag{8}
\]

For example, if class 3 is correct:

\[
y=
\begin{bmatrix}
0\\0\\0\\1\\0\\0\\0\\0\\0\\0
\end{bmatrix}.
\]

Thus:

\[
\sum_{k=1}^{10}y_k=1.
\tag{9}
\]

That fact is used critically in the softmax + cross-entropy derivation.

---

# 3. A neuron, at the lowest level

Before doing matrix algebra, it is useful to see exactly what one neuron does.

Take hidden neuron \(j\).

It receives the 784 input values:

\[
x_1,x_2,\ldots,x_{784}
\]

and has 784 weights:

\[
W^{(1)}_{j1},W^{(1)}_{j2},\ldots,W^{(1)}_{j,784}.
\]

It computes a weighted sum plus bias:

\[
\boxed{
z^{(1)}_j
=
\sum_{i=1}^{784}W^{(1)}_{ji}x_i+b^{(1)}_j
}
\tag{10}
\]

Then it applies sigmoid:

\[
\boxed{
a^{(1)}_j=\sigma(z^{(1)}_j)}
\tag{11}
\]

So one neuron is literally:

\[
\boxed{
a_j
=
\sigma\left(\sum_i w_ix_i+b\right)
}
\tag{12}
\]

There is no deeper mystery to an individual neuron.

---

# 4. Why the affine transformation comes first

For the complete hidden layer, all 300 neurons can be calculated simultaneously:

\[
\boxed{
z^{(1)}=W^{(1)}x+b^{(1)}}
\tag{13}
\]

Check the dimensions:

\[
(300\times784)(784\times1)
=
300\times1.
\]

So:

\[
z^{(1)}\in\mathbb R^{300}.
\tag{14}
\]

The component form of equation (13) is exactly equation (10):

\[
z^{(1)}_j
=
\sum_iW^{(1)}_{ji}x_i+b^{(1)}_j.
\]

The matrix notation is therefore just a compact way of expressing all 300 scalar equations at once.

Strictly speaking, \(W x+b\) is an **affine** transformation because of the bias term. In neural-network terminology it is commonly called a linear layer anyway.

---

# 5. Why we need a nonlinear activation

Suppose we had no sigmoid.

Then the hidden layer would be:

\[
z^{(1)}=W^{(1)}x+b^{(1)}
\]

and the next layer would be:

\[
z^{(2)}=W^{(2)}z^{(1)}+b^{(2)}.
\]

Substitute the first equation into the second:

\[
z^{(2)}
=
W^{(2)}(W^{(1)}x+b^{(1)})+b^{(2)}.
\tag{15}
\]

Distribute:

\[
z^{(2)}
=
W^{(2)}W^{(1)}x
+
W^{(2)}b^{(1)}
+
 b^{(2)}.
\tag{16}
\]

Define:

\[
W'=W^{(2)}W^{(1)}
\tag{17}
\]

and:

\[
b'=W^{(2)}b^{(1)}+b^{(2)}.
\tag{18}
\]

Then:

\[
\boxed{z^{(2)}=W'x+b'}
\tag{19}
\]

So several purely affine layers collapse into a single affine transformation.

Therefore the activation function is what gives the network additional nonlinear expressive power.

---

# 6. Sigmoid

The sigmoid function is:

\[
\boxed{
\sigma(z)=\frac{1}{1+e^{-z}}
}
\tag{20}
\]

It maps real numbers to the interval \((0,1)\).

Important values and limits are:

\[
\sigma(0)=\frac12,
\tag{21}
\]

\[
\lim_{z\to\infty}\sigma(z)=1,
\tag{22}
\]

\[
\lim_{z\to-\infty}\sigma(z)=0.
\tag{23}
\]

The hidden activation is therefore:

\[
\boxed{a^{(1)}_j=\sigma(z^{(1)}_j)}
\tag{24}
\]

or vectorially:

\[
\boxed{a^{(1)}=\sigma(z^{(1)})}
\tag{25}
\]

where sigmoid is applied element-by-element.

---

# 7. Derivative of sigmoid

We will need this later because the backward pass must pass through the sigmoid.

Start with:

\[
\sigma(z)=(1+e^{-z})^{-1}.
\]

Differentiate:

\[
\frac{d\sigma}{dz}
=
-(1+e^{-z})^{-2}(-e^{-z}).
\tag{26}
\]

Therefore:

\[
\frac{d\sigma}{dz}
=
\frac{e^{-z}}{(1+e^{-z})^2}.
\tag{27}
\]

Now observe:

\[
\sigma(z)=\frac1{1+e^{-z}}.
\tag{28}
\]

Also:

\[
1-\sigma(z)
=
1-rac1{1+e^{-z}}
=
\frac{e^{-z}}{1+e^{-z}}.
\tag{29}
\]

Multiply equations (28) and (29):

\[
\sigma(z)(1-\sigma(z))
=
\frac{e^{-z}}{(1+e^{-z})^2}.
\tag{30}
\]

Compare with equation (27):

\[
\boxed{
\sigma'(z)=\sigma(z)(1-\sigma(z))
}
\tag{31}
\]

Since \(a=\sigma(z)\), we can write:

\[
\boxed{
\sigma'(z)=a(1-a)
}
\tag{32}
\]

This is useful in code because the forward pass has already computed \(a\).

---

# 8. Second affine transformation

The hidden activation becomes the input to the output layer:

\[
\boxed{
z^{(2)}=W^{(2)}a^{(1)}+b^{(2)}}
\tag{33}
\]

Dimensions:

\[
(10\times300)(300\times1)=10\times1.
\]

Therefore:

\[
z^{(2)}\in\mathbb R^{10}.
\tag{34}
\]

Component-wise:

\[
\boxed{
z^{(2)}_k
=
\sum_{j=1}^{300}W^{(2)}_{kj}a^{(1)}_j+b^{(2)}_k
}
\tag{35}
\]

These \(z^{(2)}_k\) are the **logits**.

They are not probabilities.

---

# 9. Softmax

We want the 10 logits to become 10 probabilities that sum to 1.

For class \(k\):

\[
\boxed{
\hat y_k
=
\frac{e^{z^{(2)}_k}}
{\sum_{r=1}^{10}e^{z^{(2)}_r}}
}
\tag{36}
\]

Vectorially:

\[
\boxed{\hat y=\operatorname{softmax}(z^{(2)})}
\tag{37}
\]

Softmax satisfies:

\[
0<\hat y_k<1
\tag{38}
\]

and:

\[
\boxed{\sum_k\hat y_k=1.}
\tag{39}
\]

---

# 10. Numerical stability of softmax

Equation (36) is mathematically correct but can overflow numerically.

For example, \(e^{1000}\) is beyond ordinary floating-point range.

The key identity is that adding or subtracting the same constant from all logits does not change softmax.

Let:

\[
m=\max_r z_r.
\tag{40}
\]

Then calculate:

\[
\boxed{
\hat y_k
=
\frac{e^{z_k-m}}
{\sum_r e^{z_r-m}}
}
\tag{41}
\]

because:

\[
\frac{e^{z_k-m}}
{\sum_r e^{z_r-m}}
=
\frac{e^{z_k}e^{-m}}
{e^{-m}\sum_r e^{z_r}}
=
\frac{e^{z_k}}
{\sum_r e^{z_r}}.
\tag{42}
\]

This is the numerically stable form to implement.

---

# 11. Cross-entropy loss

For one-hot targets, use cross-entropy:

\[
\boxed{
L=-\sum_{k=1}^{10}y_k\log\hat y_k
}
\tag{43}
\]

If the correct class is \(c\), then \(y_c=1\) and all other \(y_k=0\), so:

\[
L=-\log\hat y_c.
\tag{44}
\]

Examples:

If the correct-class probability is \(0.8\):

\[
L=-\log(0.8)\approx0.223.
\]

If it is \(0.01\):

\[
L=-\log(0.01)\approx4.605.
\]

So confident wrong predictions are heavily penalized.

---

# 12. Complete forward pass

The complete forward computation is:

\[
\boxed{z^{(1)}=W^{(1)}x+b^{(1)}}
\tag{45}
\]

\[
\boxed{a^{(1)}=\sigma(z^{(1)})}
\tag{46}
\]

\[
\boxed{z^{(2)}=W^{(2)}a^{(1)}+b^{(2)}}
\tag{47}
\]

\[
\boxed{\hat y=\operatorname{softmax}(z^{(2)})}
\tag{48}
\]

\[
\boxed{L=-\sum_k y_k\log\hat y_k}
\tag{49}
\]

As one composition:

\[
\boxed{
\hat y
=
\operatorname{softmax}
\left(
W^{(2)}\sigma(W^{(1)}x+b^{(1)})+b^{(2)}
\right)
}
\tag{50}
\]

and the loss is:

\[
\boxed{
L=-\sum_k y_k\log \hat y_k.
}
\tag{51}
\]

---

# 13. What the backward pass is actually trying to do

We now want the parameter gradients from equation (1).

The forward dependency is:

\[
W^{(1)},b^{(1)}
\rightarrow z^{(1)}
\rightarrow a^{(1)}
\rightarrow z^{(2)}
\rightarrow\hat y
\rightarrow L.
\]

Therefore the backward dependency is:

\[
L
\rightarrow\hat y
\rightarrow z^{(2)}
\rightarrow a^{(1)}
\rightarrow z^{(1)}
\rightarrow W^{(1)},b^{(1)}.
\]

This is backpropagation.

The crucial idea is:

> At every node, compute the gradient of the loss with respect to that node, then use the local derivative to move one step to the left.

In one sentence:

\[
\boxed{
\text{backprop}=
\text{repeated application of the chain rule in reverse order}
}
\tag{52}
\]

---

# 14. Why we first calculate \(\partial L/\partial z^{(2)}\)

Suppose we want the final-layer weight gradient:

\[
\frac{\partial L}{\partial W^{(2)}}.
\]

The parameter \(W^{(2)}\) does not directly create \(L\).

Its path is:

\[
W^{(2)}
\rightarrow
z^{(2)}
\rightarrow
\hat y
\rightarrow
L.
\]

Therefore the chain rule gives a product of derivatives through that path.

So the first useful quantity is:

\[
\boxed{
\frac{\partial L}{\partial z^{(2)}}
}
\tag{53}
\]

Once we know that, the final-layer parameter gradients become straightforward.

---

# 15. Kronecker delta

The **Kronecker delta** is a compact notation for equality of indices:

\[
\boxed{
\delta_{ij}
=
\begin{cases}
1,&i=j\\
0,&i\neq j.
\end{cases}
}
\tag{54}
\]

It is an “index equality detector.”

For example:

\[
\delta_{11}=1,
\quad
\delta_{25}=0,
\quad
\delta_{77}=1.
\]

A key identity is:

\[
\boxed{
\frac{\partial x_i}{\partial x_j}=\delta_{ij}
}
\tag{55}
\]

because:

- if \(i=j\), \(\partial x_i/\partial x_i=1\),
- if \(i\neq j\), \(x_i\) does not depend on \(x_j\), so the derivative is 0.

The Kronecker delta is also the component form of the identity matrix:

\[
\boxed{I_{ij}=\delta_{ij}.}
\tag{56}
\]

---

# 16. Jacobian

A scalar derivative answers:

> How does one scalar output change when one scalar input changes?

For a vector-valued function, there are many such derivatives.

Suppose:

\[
\mathbf y=f(\mathbf x),
\]

where:

\[
\mathbf x\in\mathbb R^n,
\qquad
\mathbf y\in\mathbb R^m.
\]

Then the **Jacobian matrix** contains all partial derivatives:

\[
\boxed{
J_{ij}
=
\frac{\partial y_i}{\partial x_j}
}
\tag{57}
\]

So the Jacobian has shape:

\[
\boxed{m\times n}
\tag{58}
\]

because:

- row \(i\): output \(y_i\),
- column \(j\): input \(x_j\).

For a small example:

\[
y_1=x_1^2+3x_2,
\qquad
 y_2=x_1x_2.
\]

Then:

\[
J=
\begin{bmatrix}
2x_1&3\\
x_2&x_1
\end{bmatrix}.
\tag{59}
\]

---

# 17. Why the Jacobian matters

For a small perturbation \(d\mathbf x\):

\[
\boxed{
d\mathbf y\approx J\,d\mathbf x}
\tag{60}
\]

This is the multivariable analogue of:

\[
dy\approx f'(x)dx.
\]

Equivalently:

\[
\boxed{
f(\mathbf x+\Delta\mathbf x)
\approx
f(\mathbf x)+J\Delta\mathbf x.}
\tag{61}
\]

So the Jacobian is the matrix describing the **best local linear approximation** to a vector-valued nonlinear function.

---

# 18. Softmax derivative: setup

For clarity, temporarily suppress the layer superscript and write:

\[
\hat y_k
=
\frac{e^{z_k}}{S}
\tag{62}
\]

with:

\[
\boxed{S=\sum_j e^{z_j}.}
\tag{63}
\]

We want:

\[
\boxed{
\frac{\partial \hat y_k}{\partial z_l}
}
\tag{64}
\]

Interpretation:

> How much does output probability \(k\) change if logit \(l\) changes?

The two indices play different roles:

- \(k\): which softmax output we are observing,
- \(l\): which logit we are perturbing.

When \(k=l\), it is the same class.

When \(k\neq l\), it is a different class.

---

# 19. Why \(\partial S/\partial z_k=e^{z_k}\)

We have:

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
\tag{65}
\]

Every term except \(e^{z_k}\) treats \(z_k\) as absent, so those derivatives are zero.

Thus:

\[
\frac{\partial S}{\partial z_k}
=
\frac{\partial e^{z_k}}{\partial z_k}.
\tag{66}
\]

Since:

\[
\frac{d}{dx}e^x=e^x,
\]

we get:

\[
\boxed{
\frac{\partial S}{\partial z_k}=e^{z_k}.
}
\tag{67}
\]

Using the Kronecker delta, the same derivation can be written compactly:

\[
\frac{\partial e^{z_j}}{\partial z_k}
=
 e^{z_j}\frac{\partial z_j}{\partial z_k}
=
 e^{z_j}\delta_{jk}.
\tag{68}
\]

Therefore:

\[
\frac{\partial S}{\partial z_k}
=
\sum_j e^{z_j}\delta_{jk}
=e^{z_k}.
\tag{69}
\]

The delta simply kills every term except the one whose index matches \(k\).

---

# 20. Softmax derivative: case \(k=l\)

Start with:

\[
\hat y_k=\frac{e^{z_k}}S.
\]

If differentiating with respect to \(z_k\), both numerator and denominator depend on \(z_k\).

Using the quotient rule:

\[
\frac{\partial\hat y_k}{\partial z_k}
=
\frac{
S\frac{\partial e^{z_k}}{\partial z_k}
-
 e^{z_k}\frac{\partial S}{\partial z_k}
}{S^2}.
\tag{70}
\]

Using:

\[
\frac{\partial e^{z_k}}{\partial z_k}=e^{z_k}
\]

and equation (67):

\[
\frac{\partial S}{\partial z_k}=e^{z_k},
\]

we get:

\[
\frac{\partial\hat y_k}{\partial z_k}
=
\frac{Se^{z_k}-e^{2z_k}}{S^2}.
\tag{71}
\]

Factor:

\[
=
\frac{e^{z_k}}S
\left(1-\frac{e^{z_k}}S\right).
\tag{72}
\]

Recognize the softmax probabilities:

\[
\frac{e^{z_k}}S=\hat y_k.
\]

Therefore:

\[
\boxed{
\frac{\partial\hat y_k}{\partial z_k}
=
\hat y_k(1-\hat y_k).
}
\tag{73}
\]

---

# 21. Softmax derivative: case \(k\neq l\)

Now ask:

\[
\frac{\partial\hat y_k}{\partial z_l}
\qquad(k\neq l).
\]

We have:

\[
\hat y_k=\frac{e^{z_k}}S.
\]

Since \(k\neq l\), the numerator does not depend on \(z_l\):

\[
\frac{\partial e^{z_k}}{\partial z_l}=0.
\tag{74}
\]

The denominator does depend on \(z_l\), and:

\[
\frac{\partial S}{\partial z_l}=e^{z_l}.
\tag{75}
\]

Therefore the quotient rule gives:

\[
\frac{\partial\hat y_k}{\partial z_l}
=
\frac{
S(0)-e^{z_k}e^{z_l}
}{S^2}.
\tag{76}
\]

Thus:

\[
\frac{\partial\hat y_k}{\partial z_l}
=
-\frac{e^{z_k}e^{z_l}}{S^2}.
\tag{77}
\]

But:

\[
\hat y_k\hat y_l
=
\frac{e^{z_k}}S
\frac{e^{z_l}}S
=
\frac{e^{z_k}e^{z_l}}{S^2}.
\tag{78}
\]

Therefore:

\[
\boxed{
\frac{\partial\hat y_k}{\partial z_l}
=-\hat y_k\hat y_l,
\qquad k\neq l.
}
\tag{79}
\]

The negative sign makes intuitive sense: increasing one class's logit increases that class's share of the total probability and therefore decreases the others.

---

# 22. The softmax Jacobian

Combining the two cases:

\[
\boxed{
\frac{\partial\hat y_k}{\partial z_l}
=
\begin{cases}
\hat y_k(1-\hat y_k),&k=l,\\[4pt]
-\hat y_k\hat y_l,&k\neq l.
\end{cases}
}
\tag{80}
\]

These are the entries of the softmax Jacobian:

\[
\boxed{
J_{kl}
=
\frac{\partial\hat y_k}{\partial z_l}.
}
\tag{81}
\]

For 3 classes:

\[
J=
\begin{bmatrix}
\hat y_1(1-\hat y_1)&-\hat y_1\hat y_2&-\hat y_1\hat y_3\\
-\hat y_2\hat y_1&\hat y_2(1-\hat y_2)&-\hat y_2\hat y_3\\
-\hat y_3\hat y_1&-\hat y_3\hat y_2&\hat y_3(1-\hat y_3)
\end{bmatrix}.
\tag{82}
\]

In matrix form:

\[
\boxed{
J=\operatorname{diag}(\hat y)-\hat y\hat y^T.
}
\tag{83}
\]

To verify this, note:

\[
\operatorname{diag}(\hat y)
=
\begin{bmatrix}
\hat y_1&0&0\\
0&\hat y_2&0\\
0&0&\hat y_3
\end{bmatrix}
\]

and:

\[
\hat y\hat y^T
=
\begin{bmatrix}
\hat y_1^2&\hat y_1\hat y_2&\hat y_1\hat y_3\\
\hat y_2\hat y_1&\hat y_2^2&\hat y_2\hat y_3\\
\hat y_3\hat y_1&\hat y_3\hat y_2&\hat y_3^2
\end{bmatrix}.
\]

Subtracting gives the entries in equation (82).

---

# 23. Derivative of cross-entropy with respect to probabilities

Recall:

\[
L=-\sum_k y_k\log\hat y_k.
\]

For one term:

\[
-y_k\log\hat y_k.
\]

Differentiate with respect to \(\hat y_k\):

\[
\boxed{
\frac{\partial L}{\partial\hat y_k}
=-\frac{y_k}{\hat y_k}.
}
\tag{84}
\]

---

# 24. Why the output gradient involves a sum over all classes

We want:

\[
\frac{\partial L}{\partial z_l}.
\]

But \(z_l\) influences **every** softmax probability, not only \(\hat y_l\), because all probabilities share the denominator.

So the multivariable chain rule gives:

\[
\boxed{
\frac{\partial L}{\partial z_l}
=
\sum_k
\frac{\partial L}{\partial\hat y_k}
\frac{\partial\hat y_k}{\partial z_l}.
}
\tag{85}
\]

This is the exact reason for the summation.

Interpretation of each term:

\[
\frac{\partial L}{\partial\hat y_k}
\]

asks how much the loss cares about probability \(k\), while:

\[
\frac{\partial\hat y_k}{\partial z_l}
\]

asks how much logit \(l\) affects probability \(k\).

Multiplying gives the effect of \(z_l\) on the loss through probability \(k\). Summing over all \(k\) combines all paths.

---

# 25. Derivation of \(\partial L/\partial z_l=\hat y_l-y_l\)

Substitute equation (84) into equation (85):

\[
\frac{\partial L}{\partial z_l}
=
\sum_k
\left(-\frac{y_k}{\hat y_k}\right)
\frac{\partial\hat y_k}{\partial z_l}.
\tag{86}
\]

The softmax derivative has two cases, so split the sum into:

1. the one term where \(k=l\),
2. all terms where \(k\neq l\).

Therefore:

\[
\boxed{
\frac{\partial L}{\partial z_l}
=
\left(-\frac{y_l}{\hat y_l}\right)
\frac{\partial\hat y_l}{\partial z_l}
+
\sum_{k\neq l}
\left(-\frac{y_k}{\hat y_k}\right)
\frac{\partial\hat y_k}{\partial z_l}
}
\tag{87}
\]

For the first term, equation (73) gives:

\[
\frac{\partial\hat y_l}{\partial z_l}
=
\hat y_l(1-\hat y_l).
\]

Therefore:

\[
\left(-\frac{y_l}{\hat y_l}\right)
\hat y_l(1-\hat y_l)
=
-y_l(1-\hat y_l)
\tag{88}
\]

and:

\[
=-y_l+y_l\hat y_l.
\tag{89}
\]

Now consider the \(k\neq l\) terms. Equation (79) gives:

\[
\frac{\partial\hat y_k}{\partial z_l}
=-\hat y_k\hat y_l.
\]

Therefore:

\[
\sum_{k\neq l}
\left(-\frac{y_k}{\hat y_k}\right)
(-\hat y_k\hat y_l)
\tag{90}
\]

The negatives cancel and \(\hat y_k\) cancels:

\[
=
\sum_{k\neq l}y_k\hat y_l.
\tag{91}
\]

Because \(\hat y_l\) does not depend on \(k\):

\[
=
\hat y_l\sum_{k\neq l}y_k.
\tag{92}
\]

From equation (9):

\[
\sum_k y_k=1,
\]

so:

\[
\sum_{k\neq l}y_k=1-y_l.
\tag{93}
\]

Thus the complete derivative is:

\[
\frac{\partial L}{\partial z_l}
=
-y_l+y_l\hat y_l+
\hat y_l(1-y_l).
\tag{94}
\]

Expand:

\[
=
-y_l+y_l\hat y_l+\hat y_l-y_l\hat y_l.
\tag{95}
\]

The two mixed terms cancel:

\[
+y_l\hat y_l-y_l\hat y_l=0.
\]

Therefore:

\[
\boxed{
\frac{\partial L}{\partial z_l}
=
\hat y_l-y_l.
}
\tag{96}
\]

Because this is true for every output index \(l\):

\[
\boxed{
\frac{\partial L}{\partial z}
=\hat y-y.
}
\tag{97}
\]

This is the famous simplification for softmax + cross-entropy.

---

# 26. Meaning of \(\delta^{(2)}\)

The symbol \(\delta^{(2)}\) is **just notation**.

It is not a new operation, variable type, or special function.

We define:

\[
\boxed{
\delta^{(2)}
\equiv
\frac{\partial L}{\partial z^{(2)}}.
}
\tag{98}
\]

Using equation (97):

\[
\boxed{
\delta^{(2)}=\hat y-y.
}
\tag{99}
\]

The superscript \((2)\) means “for layer 2.” It is **not an exponent**.

Similarly:

\[
\boxed{
\delta^{(1)}
\equiv
\frac{\partial L}{\partial z^{(1)}}.
}
\tag{100}
\]

The notation is useful because \(\partial L/\partial z^{(2)}\) appears repeatedly.

We could call it \(g^{(2)}\), but \(\delta^{(2)}\) is standard neural-network notation.

A useful interpretation is:

> \(\delta^{(l)}\) is the gradient of the loss with respect to the pre-activation of layer \(l\).

---

# 27. Why \(\delta^{(2)}\) is exactly what we need for the final layer

The final layer is:

\[
z^{(2)}=W^{(2)}a^{(1)}+b^{(2)}.
\]

The parameters \(W^{(2)}\) and \(b^{(2)}\) directly create \(z^{(2)}\).

Therefore, once we know:

\[
\frac{\partial L}{\partial z^{(2)}}
\]

we can immediately differentiate with respect to those parameters.

This is the general pattern:

\[
\boxed{
\text{parameter}
\rightarrow
\text{pre-activation}
\rightarrow
\text{rest of network}
}
\]

So first compute the gradient at the pre-activation, then compute the parameter gradient.

---

# 28. Gradient of the second-layer weights

Recall the scalar equation:

\[
z^{(2)}_k
=
\sum_jW^{(2)}_{kj}a^{(1)}_j+b^{(2)}_k.
\]

We want:

\[
\frac{\partial L}{\partial W^{(2)}_{kj}}.
\]

By the chain rule:

\[
\frac{\partial L}{\partial W^{(2)}_{kj}}
=
\frac{\partial L}{\partial z^{(2)}_k}
\frac{\partial z^{(2)}_k}{\partial W^{(2)}_{kj}}.
\tag{101}
\]

The first factor is:

\[
\delta^{(2)}_k.
\]

For the second factor:

\[
\frac{\partial z^{(2)}_k}{\partial W^{(2)}_{kj}}
=a^{(1)}_j.
\tag{102}
\]

Therefore:

\[
\boxed{
\frac{\partial L}{\partial W^{(2)}_{kj}}
=
\delta^{(2)}_k a^{(1)}_j.
}
\tag{103}
\]

This explains the individual gradient entry directly.

---

# 29. Why the second-layer weight gradient is an outer product

Equation (103) says every entry is:

\[
\delta^{(2)}_k a^{(1)}_j.
\]

So the entire matrix is:

\[
\boxed{
\frac{\partial L}{\partial W^{(2)}}
=
\delta^{(2)}(a^{(1)})^T.
}
\tag{104}
\]

Dimensions:

\[
(10\times1)(1\times300)=10\times300.
\]

That matches the shape of \(W^{(2)}\).

An outer product appears because each weight connects one hidden activation to one output pre-activation.

---

# 30. Gradient of the second-layer bias

Since:

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
\tag{105}
\]

For one example, the bias gradient is simply the layer-2 delta.

At this point we have everything needed to compute the final-layer gradients:

\[
\boxed{
\frac{\partial L}{\partial W^{(2)}}
=\delta^{(2)}(a^{(1)})^T
}
\tag{106}
\]

\[
\boxed{
\frac{\partial L}{\partial b^{(2)}}
=\delta^{(2)}.
}
\tag{107}
\]

---

# 31. Continue backward: from \(z^{(2)}\) to \(a^{(1)}\)

We are not done, because the first-layer parameters also affect the loss.

The dependency is:

\[
W^{(1)},b^{(1)}
\rightarrow z^{(1)}
\rightarrow a^{(1)}
\rightarrow z^{(2)}
\rightarrow L.
\]

To continue backward, first calculate:

\[
\boxed{
\frac{\partial L}{\partial a^{(1)}}.
}
\tag{108}
\]

From:

\[
z^{(2)}_k
=
\sum_jW^{(2)}_{kj}a^{(1)}_j+b^{(2)}_k,
\]

we have:

\[
\frac{\partial z^{(2)}_k}{\partial a^{(1)}_j}
=W^{(2)}_{kj}.
\tag{109}
\]

The chain rule says:

\[
\frac{\partial L}{\partial a^{(1)}_j}
=
\sum_k
\frac{\partial L}{\partial z^{(2)}_k}
\frac{\partial z^{(2)}_k}{\partial a^{(1)}_j}.
\tag{110}
\]

Therefore:

\[
\boxed{
\frac{\partial L}{\partial a^{(1)}_j}
=
\sum_k\delta^{(2)}_kW^{(2)}_{kj}.
}
\tag{111}
\]

In matrix form:

\[
\boxed{
\frac{\partial L}{\partial a^{(1)}}
=(W^{(2)})^T\delta^{(2)}.
}
\tag{112}
\]

Dimension check:

\[
(300\times10)(10\times1)=300\times1.
\]

---

# 32. Why does the transpose appear?

This is not a special neural-network trick.

Forward:

\[
z^{(2)}=W^{(2)}a^{(1)}
\]

with:

\[
W^{(2)}:\mathbb R^{300}\rightarrow\mathbb R^{10}.
\]

Backward we want a vector in \(\mathbb R^{300}\) from a vector in \(\mathbb R^{10}\), so the matrix must have shape:

\[
300\times10.
\]

That is exactly \((W^{(2)})^T\).

The real reason is equation (111):

\[
\frac{\partial L}{\partial a^{(1)}_j}
=
\sum_k\delta^{(2)}_kW^{(2)}_{kj}.
\]

Those indices are precisely the entries of \((W^{(2)})^T\delta^{(2)}\).

---

# 33. Continue backward through the sigmoid

We now know:

\[
\frac{\partial L}{\partial a^{(1)}}.
\]

But the first layer directly creates \(z^{(1)}\), not \(a^{(1)}\):

\[
W^{(1)},b^{(1)}
\rightarrow z^{(1)}
\rightarrow a^{(1)}.
\]

Therefore we next need:

\[
\boxed{
\frac{\partial L}{\partial z^{(1)}}.
}
\tag{113}
\]

For neuron \(j\):

\[
a^{(1)}_j=\sigma(z^{(1)}_j).
\]

By the chain rule:

\[
\frac{\partial L}{\partial z^{(1)}_j}
=
\frac{\partial L}{\partial a^{(1)}_j}
\frac{\partial a^{(1)}_j}{\partial z^{(1)}_j}.
\tag{114}
\]

From the sigmoid derivative:

\[
\frac{\partial a^{(1)}_j}{\partial z^{(1)}_j}
=
 a^{(1)}_j(1-a^{(1)}_j).
\tag{115}
\]

Therefore:

\[
\boxed{
\frac{\partial L}{\partial z^{(1)}_j}
=
\frac{\partial L}{\partial a^{(1)}_j}
 a^{(1)}_j(1-a^{(1)}_j).
}
\tag{116}
\]

Define:

\[
\boxed{
\delta^{(1)}
\equiv
\frac{\partial L}{\partial z^{(1)}}.
}
\tag{117}
\]

Then:

\[
\boxed{
\delta^{(1)}
=
\frac{\partial L}{\partial a^{(1)}}
\odot
 a^{(1)}\odot(1-a^{(1)}).
}
\tag{118}
\]

Using equation (112):

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
\tag{119}
\]

Here \(\odot\) means elementwise multiplication.

---

# 34. Why sigmoid creates an elementwise multiplication

The sigmoid is applied independently to each hidden neuron:

\[
a_j=\sigma(z_j).
\]

So:

\[
\frac{\partial a_j}{\partial z_r}=0
\qquad(j\neq r).
\tag{120}
\]

Its Jacobian is diagonal:

\[
J_\sigma
=
\operatorname{diag}\left(
\begin{bmatrix}
\sigma'(z_1)\\
\vdots\\
\sigma'(z_{300})
\end{bmatrix}
\right).
\tag{121}
\]

Multiplying a diagonal matrix by a vector is equivalent to elementwise multiplication. That is why equation (118) is written with \(\odot\) rather than a full 300×300 Jacobian.

---

# 35. Gradient of first-layer weights

Now we are at:

\[
W^{(1)},b^{(1)}
\rightarrow
z^{(1)}.
\]

We already know:

\[
\delta^{(1)}
=
\frac{\partial L}{\partial z^{(1)}}.
\]

The scalar equation is:

\[
z^{(1)}_j
=
\sum_iW^{(1)}_{ji}x_i+b^{(1)}_j.
\]

We want:

\[
\frac{\partial L}{\partial W^{(1)}_{ji}}.
\]

By the chain rule:

\[
\frac{\partial L}{\partial W^{(1)}_{ji}}
=
\frac{\partial L}{\partial z^{(1)}_j}
\frac{\partial z^{(1)}_j}{\partial W^{(1)}_{ji}}.
\tag{122}
\]

Since:

\[
\frac{\partial z^{(1)}_j}{\partial W^{(1)}_{ji}}=x_i,
\tag{123}
\]

we obtain:

\[
\boxed{
\frac{\partial L}{\partial W^{(1)}_{ji}}
=\delta^{(1)}_j x_i.
}
\tag{124}
\]

In matrix form:

\[
\boxed{
\frac{\partial L}{\partial W^{(1)}}
=
\delta^{(1)}x^T.
}
\tag{125}
\]

Dimensions:

\[
(300\times1)(1\times784)=300\times784.
\]

Exactly the shape of \(W^{(1)}\).

---

# 36. Gradient of first-layer bias

Since:

\[
z^{(1)}_j=\cdots+b^{(1)}_j,
\]

we have:

\[
\frac{\partial z^{(1)}_j}{\partial b^{(1)}_j}=1.
\]

Therefore:

\[
\boxed{
\frac{\partial L}{\partial b^{(1)}}
=\delta^{(1)}.
}
\tag{126}
\]

---

# 37. The entire backward pass as one chain

We can now connect everything.

## Start at the loss

\[
L
\]

## Through softmax + cross-entropy

\[
\boxed{
\delta^{(2)}
=
\frac{\partial L}{\partial z^{(2)}}
=
\hat y-y
}
\tag{127}
\]

## Use the output affine layer

\[
\boxed{
\frac{\partial L}{\partial W^{(2)}}
=
\delta^{(2)}(a^{(1)})^T
}
\tag{128}
\]

\[
\boxed{
\frac{\partial L}{\partial b^{(2)}}
=\delta^{(2)}
}
\tag{129}
\]

## Propagate into hidden activation

\[
\boxed{
\frac{\partial L}{\partial a^{(1)}}
=
(W^{(2)})^T\delta^{(2)}
}
\tag{130}
\]

## Cross the sigmoid

\[
\boxed{
\delta^{(1)}
=
(W^{(2)})^T\delta^{(2)}
\odot a^{(1)}\odot(1-a^{(1)})
}
\tag{131}
\]

## Use the first affine layer

\[
\boxed{
\frac{\partial L}{\partial W^{(1)}}
=
\delta^{(1)}x^T
}
\tag{132}
\]

\[
\boxed{
\frac{\partial L}{\partial b^{(1)}}
=\delta^{(1)}
}
\tag{133}
\]

Everything is now connected.

---

# 38. Parameter update

Once all four gradients are known, update the parameters using gradient descent.

For the second layer:

\[
\boxed{
W^{(2)}
\leftarrow
W^{(2)}-
\eta\frac{\partial L}{\partial W^{(2)}}
}
\tag{134}
\]

\[
\boxed{
b^{(2)}
\leftarrow
b^{(2)}-
\eta\frac{\partial L}{\partial b^{(2)}}
}
\tag{135}
\]

For the first layer:

\[
\boxed{
W^{(1)}
\leftarrow
W^{(1)}-
\eta\frac{\partial L}{\partial W^{(1)}}
}
\tag{136}
\]

\[
\boxed{
b^{(1)}
\leftarrow
b^{(1)}-
\eta\frac{\partial L}{\partial b^{(1)}}
}
\tag{137}
\]

Important: **we do not “update sigmoid” or “update softmax.”**

They are functions with no trainable parameters in this network. We differentiate **through** them; we update only weights and biases.

---

# 39. The complete computation graph

The most useful map to remember is:

```text
FORWARD

x
│
├── W¹, b¹
▼
z¹
│
├── sigmoid
▼
a¹
│
├── W², b²
▼
z²   (logits)
│
├── softmax
▼
ŷ    (probabilities)
│
├── cross-entropy with y
▼
L    (scalar loss)
```

The backward direction is:

```text
BACKWARD

L
│
├── through cross-entropy + softmax
▼
δ² = ∂L/∂z² = ŷ - y
│
├───────────────┐
│               │
▼               ▼
dW²            db²
│
│
└── W²ᵀ δ²
        │
        ▼
   ∂L/∂a¹
        │
        ├── multiply by sigmoid derivative
        ▼
   δ¹ = ∂L/∂z¹
        │
        ├───────────────┐
        │               │
        ▼               ▼
      dW¹             db¹
```

This graph is the organizational backbone of the entire derivation.

---

# 40. Why every intermediate quantity exists

Every intermediate has a purpose.

| Quantity | Definition | Why it exists |
|---|---|---|
| \(z^{(1)}\) | \(W^{(1)}x+b^{(1)}\) | Input to hidden sigmoid; directly produced by first-layer parameters |
| \(a^{(1)}\) | \(\sigma(z^{(1)})\) | Input to the second layer |
| \(z^{(2)}\) | \(W^{(2)}a^{(1)}+b^{(2)}\) | Input to softmax; directly produced by second-layer parameters |
| \(\hat y\) | softmax\((z^{(2)})\) | Predicted probability distribution |
| \(L\) | cross-entropy | Scalar objective to minimize |
| \(\delta^{(2)}\) | \(\partial L/\partial z^{(2)}\) | Needed for output-layer parameter gradients and for propagating backward |
| \(\partial L/\partial a^{(1)}\) | \((W^{(2)})^T\delta^{(2)}\) | Needed to cross the sigmoid |
| \(\delta^{(1)}\) | \(\partial L/\partial z^{(1)}\) | Needed for first-layer parameter gradients |
| \(dW^{(2)}\) | \(\delta^{(2)}(a^{(1)})^T\) | Update second-layer weights |
| \(db^{(2)}\) | \(\delta^{(2)}\) | Update second-layer biases |
| \(dW^{(1)}\) | \(\delta^{(1)}x^T\) | Update first-layer weights |
| \(db^{(1)}\) | \(\delta^{(1)}\) | Update first-layer biases |

The equations are not a disconnected list. Each quantity is introduced because it is required by the next equation in the backward chain.

---

# 41. Dimension bookkeeping

Dimensions provide a powerful consistency check.

## Forward

\[
W^{(1)}x:
(300\times784)(784\times1)=300\times1.
\]

Therefore:

\[
z^{(1)}\in\mathbb R^{300}.
\]

Then:

\[
W^{(2)}a^{(1)}:
(10\times300)(300\times1)=10\times1.
\]

Therefore:

\[
z^{(2)}\in\mathbb R^{10}.
\]

## Backward

\[
(W^{(2)})^T\delta^{(2)}:
(300\times10)(10\times1)=300\times1.
\]

So:

\[
\frac{\partial L}{\partial a^{(1)}}\in\mathbb R^{300}.
\]

Then:

\[
\delta^{(1)}x^T:
(300\times1)(1\times784)=300\times784.
\]

So:

\[
\frac{\partial L}{\partial W^{(1)}}
\in
\mathbb R^{300\times784}.
\]

Likewise:

\[
\delta^{(2)}(a^{(1)})^T:
(10\times1)(1\times300)=10\times300.
\]

So:

\[
\frac{\partial L}{\partial W^{(2)}}
\in
\mathbb R^{10\times300}.
\]

Every parameter gradient must have exactly the same shape as the parameter itself.

---

# 42. Why the weight gradient is always “delta times input”

This pattern is general for affine layers.

Consider:

\[
z=Wa+b.
\tag{138}
\]

Suppose we already know:

\[
\delta=\frac{\partial L}{\partial z}.
\]

A weight \(W_{ij}\) connects input \(a_j\) to output pre-activation \(z_i\):

\[
z_i=\sum_jW_{ij}a_j+b_i.
\]

Therefore:

\[
\frac{\partial z_i}{\partial W_{ij}}=a_j.
\]

Hence:

\[
\boxed{
\frac{\partial L}{\partial W_{ij}}
=
\delta_i a_j.
}
\tag{139}
\]

So:

\[
\boxed{
\frac{\partial L}{\partial W}
=
\delta a^T.
}
\tag{140}
\]

This is why outer products repeatedly appear in neural-network gradients.

The gradient says:

> output error signal \(\times\) input activation.

---

# 43. Why the bias gradient is simply the delta

Again for:

\[
z=Wa+b,
\]

we have:

\[
\frac{\partial z_i}{\partial b_i}=1.
\]

So:

\[
\boxed{
\frac{\partial L}{\partial b_i}=\delta_i.
}
\tag{141}
\]

Thus:

\[
\boxed{
\frac{\partial L}{\partial b}=\delta.
}
\tag{142}
\]

---

# 44. Why the input gradient also exists

Suppose:

\[
z=Wa+b.
\]

If we know:

\[
\delta=\frac{\partial L}{\partial z},
\]

then the gradient with respect to the input is:

\[
\boxed{
\frac{\partial L}{\partial a}=W^T\delta.
}
\tag{143}
\]

For the first layer, \(a=x\), so:

\[
\boxed{
\frac{\partial L}{\partial x}
=(W^{(1)})^T\delta^{(1)}.
}
\tag{144}
\]

We usually do not need this gradient merely to train the parameters, but it demonstrates that backpropagation can continue all the way to the input.

---

# 45. Scalar interpretation of a final-layer weight gradient

Take one particular output weight:

\[
W^{(2)}_{7,42}.
\]

It connects hidden unit 42 to output neuron 7.

Its gradient is:

\[
\boxed{
\frac{\partial L}{\partial W^{(2)}_{7,42}}
=
\delta^{(2)}_7a^{(1)}_{42}.
}
\tag{145}
\]

This has an intuitive interpretation:

- \(a^{(1)}_{42}\): how strongly the source hidden unit was active,
- \(\delta^{(2)}_7\): how much the loss cares about changing output neuron's pre-activation.

If the source activation is near zero, that particular connection receives little gradient.

---

# 46. What a delta really means

The notation:

\[
\delta^{(l)}=\frac{\partial L}{\partial z^{(l)}}
\]

has a very concrete interpretation:

> If I make a tiny increase in the pre-activation of neuron \(j\) at layer \(l\), how does the loss change?

For one component:

\[
\delta^{(l)}_j
=
\frac{\partial L}{\partial z^{(l)}_j}.
\]

If:

\[
\delta^{(l)}_j>0,
\]

then increasing \(z^{(l)}_j\) locally increases the loss.

If:

\[
\delta^{(l)}_j<0,
\]

then increasing \(z^{(l)}_j\) locally decreases the loss.

So the delta is a local error/learning signal attached to the pre-activation.

---

# 47. Why gradient descent subtracts the gradient

Suppose one parameter is \(w\) and:

\[
\frac{\partial L}{\partial w}=+3.
\]

That says increasing \(w\) increases the loss locally.

So we should move \(w\) in the opposite direction:

\[
w_{\text{new}}=w-\eta(3).
\]

If instead:

\[
\frac{\partial L}{\partial w}=-3,
\]

then increasing \(w\) decreases the loss, so:

\[
w_{\text{new}}=w-\eta(-3)=w+3\eta.
\]

Thus gradient descent follows:

\[
\boxed{
\theta_{\text{new}}
=
\theta-\eta\nabla_\theta L.
}
\tag{146}
\]

---

# 48. Complete algorithm for one training example

At this point we can write the entire algorithm in exact mathematical order.

## Forward

### Step 1

\[
\boxed{z^{(1)}=W^{(1)}x+b^{(1)}}
\tag{147}
\]

### Step 2

\[
\boxed{a^{(1)}=\sigma(z^{(1)})}
\tag{148}
\]

### Step 3

\[
\boxed{z^{(2)}=W^{(2)}a^{(1)}+b^{(2)}}
\tag{149}
\]

### Step 4

Compute stable softmax:

\[
 m=\max_k z^{(2)}_k,
\tag{150}
\]

\[
\boxed{
\hat y_k
=
\frac{e^{z^{(2)}_k-m}}
{\sum_r e^{z^{(2)}_r-m}}
}
\tag{151}
\]

### Step 5

\[
\boxed{
L=-\sum_k y_k\log\hat y_k
}
\tag{152}
\]

---

## Backward

### Step 6: output delta

\[
\boxed{
\delta^{(2)}=\hat y-y
}
\tag{153}
\]

### Step 7: output weight gradient

\[
\boxed{
\frac{\partial L}{\partial W^{(2)}}
=\delta^{(2)}(a^{(1)})^T
}
\tag{154}
\]

### Step 8: output bias gradient

\[
\boxed{
\frac{\partial L}{\partial b^{(2)}}
=\delta^{(2)}
}
\tag{155}
\]

### Step 9: propagate to hidden activation

\[
\boxed{
\frac{\partial L}{\partial a^{(1)}}
=(W^{(2)})^T\delta^{(2)}
}
\tag{156}
\]

### Step 10: hidden delta

\[
\boxed{
\delta^{(1)}
=
\frac{\partial L}{\partial a^{(1)}}
\odot a^{(1)}\odot(1-a^{(1)})
}
\tag{157}
\]

or directly:

\[
\boxed{
\delta^{(1)}
=
(W^{(2)})^T\delta^{(2)}
\odot a^{(1)}
\odot(1-a^{(1)})
}
\tag{158}
\]

### Step 11: first-layer weight gradient

\[
\boxed{
\frac{\partial L}{\partial W^{(1)}}
=\delta^{(1)}x^T
}
\tag{159}
\]

### Step 12: first-layer bias gradient

\[
\boxed{
\frac{\partial L}{\partial b^{(1)}}
=\delta^{(1)}
}
\tag{160}
\]

---

## Parameter update

### Step 13

\[
\boxed{
W^{(2)}
\leftarrow
W^{(2)}-
\eta\frac{\partial L}{\partial W^{(2)}}
}
\tag{161}
\]

### Step 14

\[
\boxed{
 b^{(2)}
\leftarrow
 b^{(2)}-
\eta\frac{\partial L}{\partial b^{(2)}}
}
\tag{162}
\]

### Step 15

\[
\boxed{
W^{(1)}
\leftarrow
W^{(1)}-
\eta\frac{\partial L}{\partial W^{(1)}}
}
\tag{163}
\]

### Step 16

\[
\boxed{
b^{(1)}
\leftarrow
b^{(1)}-
\eta\frac{\partial L}{\partial b^{(1)}}
}
\tag{164}
\]

That completes one training example.

---

# 49. Parameter count

The first layer has:

\[
784\times300=235200
\tag{165}
\]

weights and 300 biases:

\[
235200+300=235500.
\tag{166}
\]

The second layer has:

\[
300\times10=3000
\tag{167}
\]

weights and 10 biases:

\[
3000+10=3010.
\tag{168}
\]

Total parameters:

\[
235500+3010
=
\boxed{238510}.
\tag{169}
\]

So this network has 238,510 trainable parameters.

---

# 50. Mapping the mathematics to a C implementation

For a single example, the natural storage layout is conceptually:

```text
x       [784]

W1      [300][784]
b1      [300]
z1      [300]
a1      [300]

W2      [10][300]
b2      [10]
z2      [10]
yhat    [10]

d1      [300]
d2      [10]

dW1     [300][784]
db1     [300]
dW2     [10][300]
db2     [10]
```

You can use different naming, but these objects correspond directly to the equations.

---

# 51. Forward pass in scalar loops

Equation (147):

\[
z^{(1)}_j
=
\sum_iW^{(1)}_{ji}x_i+b^{(1)}_j.
\]

Maps almost literally to:

```text
for (j = 0; j < 300; j++) {
    z1[j] = b1[j];
    for (i = 0; i < 784; i++) {
        z1[j] += W1[j][i] * x[i];
    }
}
```

Then:

```text
for (j = 0; j < 300; j++) {
    a1[j] = sigmoid(z1[j]);
}
```

Equation (149):

\[
z^{(2)}_k
=
\sum_jW^{(2)}_{kj}a^{(1)}_j+b^{(2)}_k.
\]

Maps to:

```text
for (k = 0; k < 10; k++) {
    z2[k] = b2[k];
    for (j = 0; j < 300; j++) {
        z2[k] += W2[k][j] * a1[j];
    }
}
```

Then stable softmax:

```text
m = max(z2);

sum = 0.0;
for (k = 0; k < 10; k++) {
    yhat[k] = exp(z2[k] - m);
    sum += yhat[k];
}

for (k = 0; k < 10; k++) {
    yhat[k] /= sum;
}
```

---

# 52. Backward pass in scalar loops

Output delta:

```text
for (k = 0; k < 10; k++) {
    d2[k] = yhat[k] - y[k];
}
```

This is equation (153):

\[
d2=\hat y-y.
\]

Output-layer weight gradients:

\[
\frac{\partial L}{\partial W^{(2)}_{kj}}
=d2_k a1_j.
\]

So:

```text
for (k = 0; k < 10; k++) {
    for (j = 0; j < 300; j++) {
        dW2[k][j] = d2[k] * a1[j];
    }
}
```

Output bias gradients:

```text
for (k = 0; k < 10; k++) {
    db2[k] = d2[k];
}
```

Hidden gradient before sigmoid:

\[
\frac{\partial L}{\partial a^{(1)}_j}
=
\sum_kW^{(2)}_{kj}d2_k.
\]

So:

```text
for (j = 0; j < 300; j++) {
    da1[j] = 0.0;
    for (k = 0; k < 10; k++) {
        da1[j] += W2[k][j] * d2[k];
    }
}
```

Then cross sigmoid:

```text
for (j = 0; j < 300; j++) {
    d1[j] = da1[j] * a1[j] * (1.0 - a1[j]);
}
```

Finally:

```text
for (j = 0; j < 300; j++) {
    for (i = 0; i < 784; i++) {
        dW1[j][i] = d1[j] * x[i];
    }
}

for (j = 0; j < 300; j++) {
    db1[j] = d1[j];
}
```

The correspondence between mathematics and code is almost one-to-one.

---

# 53. What not to calculate explicitly in the C implementation

Mathematically, the softmax Jacobian exists:

\[
J=\operatorname{diag}(\hat y)-\hat y\hat y^T.
\]

But for ordinary softmax + cross-entropy training, you do **not** need to construct the full 10×10 Jacobian.

The entire chain-rule calculation simplifies to:

\[
\boxed{d2=\hat y-y.}
\]

Likewise, for sigmoid, its Jacobian is diagonal, so you do not need a 300×300 matrix.

You use the elementwise derivative:

\[
\sigma'(z)=a(1-a).
\]

This is why practical backpropagation is much cheaper than naively constructing every Jacobian.

---

# 54. A useful conceptual distinction: pre-activation, activation, gradient

These are three different things and should never be mentally collapsed into one.

## Pre-activation

\[
z.
\]

This is the weighted sum plus bias.

For example:

\[
z^{(1)}=W^{(1)}x+b^{(1)}.
\]

## Activation

\[
a.
\]

This is what comes out after the activation function.

For the hidden layer:

\[
a^{(1)}=\sigma(z^{(1)}).
\]

## Gradient / delta

\[
\delta=\frac{\partial L}{\partial z}.
\]

This is not an activation. It is a sensitivity of the loss to the pre-activation.

Keeping these three concepts separate makes backpropagation much easier to reason about.

---

# 55. The multivariable chain rule behind all of backpropagation

For ordinary scalar functions:

\[
y=f(x),
\qquad
L=g(y),
\]

we have:

\[
\frac{dL}{dx}
=
\frac{dL}{dy}
\frac{dy}{dx}.
\tag{170}
\]

For vector-valued functions, Jacobians replace scalar derivatives.

If:

\[
\mathbf y=f(\mathbf x),
\qquad
\mathbf z=g(\mathbf y),
\]

then locally:

\[
\frac{\partial \mathbf z}{\partial\mathbf x}
=
\frac{\partial\mathbf z}{\partial\mathbf y}
\frac{\partial\mathbf y}{\partial\mathbf x}.
\tag{171}
\]

In Jacobian notation:

\[
\boxed{
J_{g\circ f}(x)=J_g(f(x))J_f(x).
}
\tag{172}
\]

Backpropagation exploits this structure in reverse, usually avoiding explicit construction of large Jacobians.

---

# 56. Why the softmax outputs are coupled

Softmax satisfies:

\[
\sum_k\hat y_k=1.
\]

Therefore its outputs cannot vary independently.

Increasing one logit changes its corresponding numerator and also changes the common denominator.

Consequently:

\[
\frac{\partial\hat y_k}{\partial z_l}\neq0
\quad(k\neq l).
\]

By contrast, if:

\[
a_i=\sigma(x_i),
\]

then:

\[
\frac{\partial a_i}{\partial x_j}=0
\quad(i\neq j).
\]

So the sigmoid Jacobian is diagonal while the softmax Jacobian is not.

This difference is a direct consequence of the denominator coupling in softmax.

---

# 57. The two important derivative patterns

There are two recurring patterns you should recognize immediately.

## Pattern A: elementwise nonlinearity

For:

\[
a=\phi(z)
\]

applied elementwise:

\[
\boxed{
\frac{\partial L}{\partial z}
=
\frac{\partial L}{\partial a}
\odot\phi'(z).
}
\tag{173}
\]

For sigmoid:

\[
\boxed{
\frac{\partial L}{\partial z}
=
\frac{\partial L}{\partial a}
\odot a(1-a).
}
\tag{174}
\]

## Pattern B: affine layer

For:

\[
z=Wa+b,
\]

if:

\[
\delta=\frac{\partial L}{\partial z},
\]

then:

\[
\boxed{
\frac{\partial L}{\partial W}=\delta a^T
}
\tag{175}
\]

\[
\boxed{
\frac{\partial L}{\partial b}=\delta
}
\tag{176}
\]

\[
\boxed{
\frac{\partial L}{\partial a}=W^T\delta.
}
\tag{177}
\]

These three equations plus the output softmax + cross-entropy simplification are the core reusable patterns.

---

# 58. A complete “dependency-first” view

Instead of memorizing formulas, trace the dependency graph.

## To get \(dW^{(2)}\), what do we need?

We know:

\[
W^{(2)}\rightarrow z^{(2)}.
\]

Therefore we need:

\[
\frac{\partial L}{\partial z^{(2)}}.
\]

But:

\[
z^{(2)}\rightarrow\hat y\rightarrow L.
\]

So we work backward through softmax and cross-entropy:

\[
L
\rightarrow\hat y
\rightarrow z^{(2)}
\]

giving:

\[
\delta^{(2)}=\hat y-y.
\]

Then:

\[
 dW^{(2)}=
\delta^{(2)}(a^{(1)})^T.
\]

---

## To get \(dW^{(1)}\), what do we need?

We know:

\[
W^{(1)}\rightarrow z^{(1)}.
\]

Therefore we need:

\[
\frac{\partial L}{\partial z^{(1)}}.
\]

But:

\[
z^{(1)}\rightarrow a^{(1)}\rightarrow z^{(2)}\rightarrow\hat y\rightarrow L.
\]

So we must continue backward:

\[
L
\rightarrow z^{(2)}
\rightarrow a^{(1)}
\rightarrow z^{(1)}.
\]

First:

\[
\frac{\partial L}{\partial a^{(1)}}
=(W^{(2)})^T\delta^{(2)}.
\]

Then through sigmoid:

\[
\delta^{(1)}
=
\frac{\partial L}{\partial a^{(1)}}
\odot a^{(1)}(1-a^{(1)}).
\]

Then:

\[
dW^{(1)}=\delta^{(1)}x^T.
\]

This is why the backward pass has the order that it does.

---

# 59. The “one line per stage” summary

If you want a compact reference after understanding everything:

## Forward

\[
\boxed{z_1=W_1x+b_1}
\]

\[
\boxed{a_1=\sigma(z_1)}
\]

\[
\boxed{z_2=W_2a_1+b_2}
\]

\[
\boxed{\hat y=\operatorname{softmax}(z_2)}
\]

\[
\boxed{L=-\sum_i y_i\log\hat y_i}
\]

## Backward

\[
\boxed{\delta_2=\hat y-y}
\]

\[
\boxed{dW_2=\delta_2a_1^T}
\]

\[
\boxed{db_2=\delta_2}
\]

\[
\boxed{da_1=W_2^T\delta_2}
\]

\[
\boxed{\delta_1=da_1\odot a_1\odot(1-a_1)}
\]

\[
\boxed{dW_1=\delta_1x^T}
\]

\[
\boxed{db_1=\delta_1}
\]

## Update

\[
\boxed{W\leftarrow W-\eta dW}
\]

\[
\boxed{b\leftarrow b-\eta db}
\]

---

# 60. Common conceptual mistakes

## Mistake 1: thinking \(\delta^{(2)}\) is a new function

It is only a shorthand:

\[
\delta^{(2)}=\frac{\partial L}{\partial z^{(2)}}.
\]

---

## Mistake 2: thinking the superscript means “squared”

\[
\delta^{(2)}
\]

means “delta for layer 2,” not \(\delta^2\).

---

## Mistake 3: saying we “update softmax” or “update sigmoid”

No. We update the trainable parameters \(W\) and \(b\).

We differentiate through the activation functions.

---

## Mistake 4: thinking \(k\) and \(l\) are special constants

They are dummy indices.

\[
k=\text{output index},
\qquad
l=\text{logit being differentiated with respect to}.
\]

They could have been named \(i,j\), \(p,q\), etc.

---

## Mistake 5: forgetting that softmax couples outputs

For softmax:

\[
\frac{\partial\hat y_k}{\partial z_l}
\neq0
\quad(k\neq l).
\]

The common denominator couples all outputs.

---

## Mistake 6: using the unstable softmax expression directly

Use:

\[
\hat y_k
=
\frac{e^{z_k-m}}{\sum_j e^{z_j-m}},
\qquad
m=\max_j z_j.
\]

---

## Mistake 7: constructing the full softmax Jacobian during ordinary training

You usually do not need to. Softmax + cross-entropy simplifies to:

\[
\delta^{(2)}=\hat y-y.
\]

---

## Mistake 8: mixing up \(z\), \(a\), and \(\delta\)

Remember:

\[
z=\text{pre-activation},
\]

\[
a=\text{activation},
\]

\[
\delta=\frac{\partial L}{\partial z}=\text{gradient at pre-activation}.
\]

---

# 61. A deeper interpretation of softmax + cross-entropy

The output delta:

\[
\delta^{(2)}=\hat y-y
\]

has a clean interpretation.

For each class:

\[
\delta_k=\text{predicted probability}-\text{target probability}.
\]

For the correct class in a one-hot target:

\[
\delta_c=\hat y_c-1<0
\]

unless the prediction is exactly 1.

For an incorrect class \(k\):

\[
\delta_k=\hat y_k>0.
\]

Thus the gradient pushes the correct class logit upward and the incorrect class logits downward when gradient descent is applied.

This is why the expression is so intuitive after all of the calculus is done.

---

# 62. A local-perturbation interpretation of all gradients

Suppose a parameter changes by a tiny amount \(d\theta\).

Locally:

\[
 dL
\approx
\frac{\partial L}{\partial\theta}d\theta.
\tag{178}
\]

For many parameters:

\[
dL
\approx
\nabla_\theta L^T d\theta.
\tag{179}
\]

This is the deeper meaning of the gradient: it tells you the local sensitivity of the loss to every parameter direction.

Gradient descent chooses:

\[
d\theta=-\eta\nabla_\theta L,
\]

which locally gives:

\[
dL
\approx
-\eta\nabla_\theta L^T\nabla_\theta L
=
-\eta\|\nabla_\theta L\|^2
\le0.
\tag{180}
\]

This shows, in the first-order approximation, why moving opposite the gradient lowers the loss unless the gradient is zero.

---

# 63. One complete chain written only with scalar dependencies

For a particular first-layer weight \(W^{(1)}_{ji}\), the complete path to the loss is:

\[
W^{(1)}_{ji}
\rightarrow
z^{(1)}_j
\rightarrow
 a^{(1)}_j
\rightarrow
 z^{(2)}_1,\ldots,z^{(2)}_{10}
\rightarrow
\hat y_1,\ldots,\hat y_{10}
\rightarrow
L.
\]

There are multiple downstream paths after \(a^{(1)}_j\), because that hidden unit contributes to every output logit.

This is why:

\[
\frac{\partial L}{\partial a^{(1)}_j}
=
\sum_k
\frac{\partial L}{\partial z^{(2)}_k}
\frac{\partial z^{(2)}_k}{\partial a^{(1)}_j}.
\]

That sum is literally “add the effects through every path.”

Then sigmoid contributes one local derivative:

\[
\frac{\partial a^{(1)}_j}{\partial z^{(1)}_j}
=a^{(1)}_j(1-a^{(1)}_j).
\]

Finally the affine layer contributes:

\[
\frac{\partial z^{(1)}_j}{\partial W^{(1)}_{ji}}=x_i.
\]

Therefore:

\[
\boxed{
\frac{\partial L}{\partial W^{(1)}_{ji}}
=
\left[
\sum_kW^{(2)}_{kj}\delta^{(2)}_k
\right]
 a^{(1)}_j(1-a^{(1)}_j)x_i.
}
\tag{181}
\]

This is the fully expanded scalar expression for a first-layer weight gradient.

The compact matrix equations are simply a cleaner way of calculating the same thing.

---

# 64. What backpropagation is *not*

Backpropagation is not:

- a mysterious neural-network-specific force,
- a special optimization algorithm,
- a replacement for the chain rule,
- a heuristic approximation.

Backpropagation **is** an efficient organization of derivatives using the chain rule and reuse of shared intermediate gradients.

The network is a composition of functions.

The derivative of a composition follows the chain rule.

Backprop calculates those derivatives efficiently from the output toward the input.

---

# 65. One-layer abstraction that generalizes to deeper networks

For a generic layer:

\[
z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)}.
\tag{182}
\]

Suppose the layer has activation:

\[
a^{(l)}=\phi^{(l)}(z^{(l)}).
\tag{183}
\]

Then once you know:

\[
\delta^{(l)}
=
\frac{\partial L}{\partial z^{(l)}},
\]

you immediately have:

\[
\boxed{
\frac{\partial L}{\partial W^{(l)}}
=
\delta^{(l)}(a^{(l-1)})^T
}
\tag{184}
\]

\[
\boxed{
\frac{\partial L}{\partial b^{(l)}}
=\delta^{(l)}
}
\tag{185}
\]

and if you need to propagate further backward:

\[
\boxed{
\frac{\partial L}{\partial a^{(l-1)}}
=(W^{(l)})^T\delta^{(l)}.
}
\tag{186}
\]

If the activation is elementwise:

\[
\boxed{
\delta^{(l-1)}
=
\left[(W^{(l)})^T\delta^{(l)}\right]
\odot
\phi^{(l-1)\prime}(z^{(l-1)}).
}
\tag{187}
\]

Our 784→300→10 network is just this generic pattern instantiated twice, with sigmoid in the hidden layer and softmax + cross-entropy at the output.

---

# 66. The complete mental model

The most durable way to remember the entire subject is not as a list of formulas, but as the following sequence.

## Forward

1. **Affine transformation:**
   \[
   z=Wa+b
   \]

2. **Activation:**
   \[
   a=\phi(z)
   \]

3. Repeat until you produce the output.

4. **Loss:** compare prediction with target.

## Backward

1. Start from the loss.

2. Ask:
   > What is the gradient with respect to the immediately preceding quantity?

3. Multiply by the local derivative.

4. Continue one operation to the left.

5. When you reach an affine layer, the gradients naturally become:
   \[
   dW=\delta a^T,
   \]
   \[
   db=\delta,
   \]
   \[
   da=W^T\delta.
   \]

6. When you cross an elementwise activation, multiply by its derivative elementwise.

7. When you have all parameter gradients, apply:
   \[
   \theta\leftarrow\theta-\eta\nabla_\theta L.
   \]

That is backpropagation.

---

# 67. Final reference sheet

## Network

\[
784\rightarrow300\rightarrow10
\]

## Shapes

\[
x:784
\]

\[
W^{(1)}:300\times784
\]

\[
b^{(1)}:300
\]

\[
a^{(1)}:300
\]

\[
W^{(2)}:10\times300
\]

\[
b^{(2)}:10
\]

\[
z^{(2)}:10
\]

\[
\hat y:10
\]

\[
y:10
\]

## Forward equations

\[
\boxed{z^{(1)}=W^{(1)}x+b^{(1)}}
\]

\[
\boxed{a^{(1)}=\sigma(z^{(1)})}
\]

\[
\boxed{z^{(2)}=W^{(2)}a^{(1)}+b^{(2)}}
\]

\[
\boxed{\hat y_k=\frac{e^{z^{(2)}_k}}{\sum_r e^{z^{(2)}_r}}}
\]

Stable implementation:

\[
\boxed{\hat y_k=\frac{e^{z^{(2)}_k-m}}{\sum_r e^{z^{(2)}_r-m}},\quad m=\max_r z^{(2)}_r}
\]

\[
\boxed{L=-\sum_k y_k\log\hat y_k}
\]

## Basic derivatives

\[
\boxed{\sigma'(z)=\sigma(z)(1-\sigma(z))}
\]

\[
\boxed{
\frac{\partial\hat y_k}{\partial z_l}
=
\begin{cases}
\hat y_k(1-\hat y_k),&k=l,\\
-\hat y_k\hat y_l,&k\neq l
\end{cases}}
\]

## Output backward pass

\[
\boxed{\delta^{(2)}=\frac{\partial L}{\partial z^{(2)}}=\hat y-y}
\]

\[
\boxed{dW^{(2)}=\delta^{(2)}(a^{(1)})^T}
\]

\[
\boxed{db^{(2)}=\delta^{(2)}}
\]

\[
\boxed{da^{(1)}=(W^{(2)})^T\delta^{(2)}}
\]

## Hidden backward pass

\[
\boxed{\delta^{(1)}=da^{(1)}\odot a^{(1)}\odot(1-a^{(1)})}
\]

\[
\boxed{dW^{(1)}=\delta^{(1)}x^T}
\]

\[
\boxed{db^{(1)}=\delta^{(1)}}
\]

## Update

\[
\boxed{W^{(l)}\leftarrow W^{(l)}-\eta dW^{(l)}}
\]

\[
\boxed{b^{(l)}\leftarrow b^{(l)}-\eta db^{(l)}}
\]

---

# 68. Final “if I forget everything” map

If you come back to this years later, start here.

### Forward

```text
x
 ↓
W1x + b1
 ↓
z1
 ↓
sigmoid
 ↓
a1
 ↓
W2a1 + b2
 ↓
z2
 ↓
softmax
 ↓
yhat
 ↓
cross-entropy with y
 ↓
L
```

### Backward

```text
L
 ↓
softmax + cross-entropy
 ↓
δ2 = yhat - y
 ↓
├── dW2 = δ2 a1ᵀ
├── db2 = δ2
└── W2ᵀ δ2
         ↓
       da1
         ↓
   × sigmoid'(z1)
         ↓
   δ1 = da1 ⊙ a1 ⊙ (1-a1)
         ↓
├── dW1 = δ1 xᵀ
└── db1 = δ1
```

And then:

```text
W1 ← W1 - η dW1
b1 ← b1 - η db1
W2 ← W2 - η dW2
b2 ← b2 - η db2
```

The single idea tying the whole document together is:

\[
\boxed{
\textbf{Backpropagation = the chain rule applied backward through the computational graph.}
}
\]

Every derivative you calculated exists because it is the next link required to move the gradient one step closer to the parameter you ultimately want to update.
