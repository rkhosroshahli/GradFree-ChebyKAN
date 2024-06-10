# ChebyKAN

Gradient-free Kolmogorov-Arnold Networks (KAN) using Chebyshev polynomials instead of B-splines. Due to scarcity of dimensionality, we reduce the trainable parameters in optimization using Block technique insipired from <https://ieeexplore.ieee.org/abstract/document/10254079> paper.

This is inspired by Kolmogorov-Arnold Networks <https://arxiv.org/abs/2404.19756v2>, which uses B-splines to approximate functions. B-splines are poor in performance and not very intuitive to use. @SynodicMonth and others tried to replace B-splines with Chebyshev polynomials.

[Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials) are orthogonal polynomials defined on the interval [-1, 1]. They are very good at approximating functions and can be calculated recursively.

A simple (and naive) implementation of ChebyKANLayer is provided in `chebyKANLayer_unoptimized.py`. Its reserved for a brief understanding.

Thanks to @SynodicMonth to develop this beautiful project in an interpretable way.
Thanks to @JanRocketMan for proving ChebyKAN = Linear + custom activation function. (see issue #3 for more information)
Thanks @iiisak and @K-H-Ismail for providing an optimized version by replace recurrent definition with trigonometric definition and vectorization. The optimized version is in `ChebyKANLayer.py`.

# Usage

Just copy `ChebyKANLayer.py` to your project and import it.

```python
from ChebyKANLayer import ChebyKANLayer
```

# Example

Construct a ChebyKAN for MNIST

```python
class MNISTChebyKAN(nn.Module):
    def __init__(self):
        super(MNISTChebyKAN, self).__init__()
        self.chebykan1 = ChebyKANLayer(28*28, 32, 4)
        self.ln1 = nn.LayerNorm(32) # To avoid gradient vanishing caused by tanh
        self.chebykan2 = ChebyKANLayer(32, 16, 4)
        self.ln2 = nn.LayerNorm(16)
        self.chebykan3 = ChebyKANLayer(16, 10, 4)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the images
        x = self.chebykan1(x)
        x = self.ln1(x)
        x = self.chebykan2(x)
        x = self.ln2(x)
        x = self.chebykan3(x)
        return x
```

**Note:** Since Chebyshev polynomials are defined on the interval [-1, 1], we need to use tanh to keep the input in that range. We also use LayerNorm to avoid gradient vanishing caused by tanh. Removing LayerNorm will cause the network really hard to train.

Have a look at `GradFree-Cheby-KAN-MNIST.ipynb` for gradien-free optimization and `Cheby-KAN-MNIST.ipynb` for gradient-based optimization.
