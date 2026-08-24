## The Sigmoid Derivative & The Vanishing Gradient Problem

The derivative of the sigmoid function $\sigma(x)$ can be expressed elegantly in terms of the function itself:

$$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$

While mathematically efficient, this property is the root cause of the **Vanishing Gradient Problem** in deep neural networks.

---

### 1. Mathematical Breakdown

There are two primary ways to derive this formula:

#### Method A: The Chain Rule (Expanded)
Starting with the function written as a negative exponent:
$$\sigma(x) = (1 + e^{-x})^{-1}$$

1. **Outer Derivative (Power Rule):** 
   $$\frac{d}{dx}[(1 + e^{-x})^{-1}] = -1(1 + e^{-x})^{-2} \cdot \frac{d}{dx}[1 + e^{-x}]$$
2. **Inner Derivative (Chain Rule):** The derivative of $(1 + e^{-x})$ is $-e^{-x}$.
   $$\sigma'(x) = -1(1 + e^{-x})^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1 + e^{-x})^2}$$
3. **Algebraic Split:**
   $$\sigma'(x) = \left( \frac{1}{1 + e^{-x}} \right) \cdot \left( \frac{e^{-x}}{1 + e^{-x}} \right)$$
4. **Substitution:** Because $\frac{e^{-x}}{1 + e^{-x}} = 1 - \sigma(x)$, we get:
   $$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$

#### Method B: The Quotient Rule
Using $\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$ where $f = 1$ and $g = 1 + e^{-x}$:
$$\sigma'(x) = \frac{(0) \cdot (1 + e^{-x}) - (1) \cdot (-e^{-x})}{(1 + e^{-x})^2} = \frac{e^{-x}}{(1 + e^{-x})^2}$$

---

### 2. The Vanishing Gradient Problem

The mathematical structure of $\sigma'(x)$ creates a bottleneck during backpropagation in deep networks:

* **Low Maximum Value:** The maximum possible value of $\sigma'(x)$ is only **0.25** (occurring at $x = 0$). As $x$ moves away from 0, the derivative rapidly approaches 0.
* **The Multiplicative Bottleneck:** During backpropagation, the Chain Rule multiplies gradients layer by layer. For a 4-layer network, updating the earliest weights requires calculating:
  $$\text{Gradient} \propto \sigma'(z_4) \cdot \sigma'(z_3) \cdot \sigma'(z_2) \cdot \sigma'(z_1)$$
* **Diminishing Returns:** Even in a perfect scenario where every neuron outputs its maximum derivative ($0.25$):
  * **1 Layer Back:** $0.25$
  * **2 Layers Back:** $0.25 \times 0.25 = 0.0625$
  * **3 Layers Back:** $0.25 \times 0.25 \times 0.25 = 0.0156$
  * **4 Layers Back:** $0.25 \times 0.25 \times 0.25 \times 0.25 = 0.0039$

By the time the gradient flows back to the earliest layers, it has effectively vanished. As a result, the front layers learn incredibly slowly or stop updating entirely, preventing the network from learning basic visual features or abstract concepts.

---

### 3. The Solution: ReLU

To solve this, modern neural networks use **ReLU (Rectified Linear Unit)** in hidden layers:
* **Function:** $f(x) = \max(0, x)$
* **Derivative:** Exactly **1** for all positive inputs ($x > 0$).

Because $1 \times 1 \times 1 \times 1 = 1$, gradients can flow backwards through hundreds of layers without losing strength.
