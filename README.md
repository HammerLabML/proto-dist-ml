# Prototype-based Machine Learning on Distance Data

Copyright (C) 2019 - Benjamin Paassen  
Machine Learning Research Group  
Center of Excellence Cognitive Interaction Technology (CITEC)  
Bielefeld University

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.

## Introduction

This [scikit-learn][scikit] compatible, Python3 library provides several algorithms
to learn prototype models on distance data. At this time, this library features
the following algorithms:

* Relational Neural Gas ([Hammer and Hasenfuss, 2007][Ham2007]) for clustering, and
* Median Generalized Learning Vector Quantization ([Nebel, Hammer, Frohberg, and Villmann, 2015][Neb2015])
    for classification.

Refer to the Quickstart Guide for a note on how to use these models and
refer to the Background section for more details on the algorithms.

Note that this library follows the 

If you intend to use this library in academic work, please cite the respective
reference paper.

## Installation

This package is available on `pypi` as `proto_dist_ml`. You can install
it via

```
pip install proto_dist_ml
```

## QuickStart Guide

For an example we recommend to take a look at the demo in the notebook
`demo.ipynb`. In general, all models in this library follow the [scikit-learn][scikit]
convention, i.e. you need to perform the following steps:

1. Instantiate your model, e.g. via `model = proto_dist_ml.rng.RNG(K)` where
    `K` is the number of prototypes.
2. Fit your model to training data, e.g. via `model.fit(D)`, where `D` is the
    matrix of pairwise distances between your training data points.
3. Perform a prediction for test data, e.g. via `model.predict(D)`, where `D`
    is the matrix of distances from the test to the training data points.

## Background

The basic idea of prototype models for distance data is that every data point
is assigned to the closest prototype as defined by a given distance. Prototypes
are represented either as data points itself (in case of median approaches) or
as convex combinations of data points, i.e. the $`k`$th prototype $`w_k`$ is
defined as

```math
\vec w_k = \sum_{i=1}^m \alpha_{k, i} \cdot \vec x_i
\qquad \text{where } \sum_{i=1}^m \alpha_{k, i} = 1
\text{ and } \alpha_{k, i} \geq 0 \quad \forall i
```

where $`\vec x_1, \ldots, \vec x_m`$ are the training data points and where
$`\alpha_{k, 1}, \ldots, \alpha_{k, m}`$ are our convex coefficients.

Intuitively, this definition only makes sense for vectorial data. However, as
it turns out, we do not need to explicitly refer to the data vectors, because
the distance between any data point $`\vec x`$ and any prototype $`\vec w_k`$
can be re-written as follows:

```math
||\vec x - \vec w_k||^2 = \sum_{j=1}^m \alpha_{k, j} \cdot ||\vec x - \vec x_j||^2
- \frac{1}{2} \sum_{j=1}^m \sum_{j'=1}^m \alpha_{k, j} \cdot \alpha_{k, j'} \cdot ||\vec x_j - \vec x_{j'}||^2
```

In other words, we only need the distances from $`\vec x`$ to the training data
and the convex coefficients to determine the distance between $`\vec x`$ and
$`\vec w_k`$. This can be expressed even more neatly by re-writing as follows:

```math
d_k^2 = {\vec \alpha_k}^T \cdot \vec d^2
- \frac{1}{2} {\vec \alpha_k}^T \cdot D^2 \cdot \vec \alpha_k
```

The main challenge for distance-based prototype learning is now to optimize
the coefficients $`\alpha_{k, i}`$ according to some meaningful loss function.
The loss function and its optimization differs between the algorithms. In more
detail, we take the following approaches.

### Relational Neural Gas

<!-- TODO -->

### Median Generalized Learning Vector Quantization

<!-- TODO -->

## Contents

This library contains the following files.

* `demo.ipynb` : A demo script illustrating how to use this library.
* `LICENSE.md` : A copy of the GPLv3 license.
* `proto_dist_ml/rng.py` : An implementation of relational neural gas.
* `README.md` : This file.
* `rng_test.py` : A set of unit tests for `rng.py`.

## Licensing

This library is licensed under the [GNU General Public License Version 3][GPLv3].

## Dependencies

This library depends on [NumPy][np] for matrix operations, on [scikit-learn][scikit]
for the base interfaces and on [SciPy][scipy] for optimization.

## Literature

* Hammer, B. & Hasenfuss, A. (2007). Relational Neural Gas. Proceedings of the
    30th Annual German Conference on AI (KI 2007), 190-204. doi:[10.1007/978-3-540-74565-5_16](https://doi.org/10.1007/978-3-540-74565-5_16). [Link][Ham2007]
* Nebel, D., Hammer, B., Frohberg, K., & Villmann, T. (2015). Median variants
    of learning vector quantization for learning of dissimilarity data.
    Neurocomputing, 169, 295-305. doi:[10.1016/j.neucom.2014.12.096][Neb2015]

<!-- References -->

[scikit]: https://scikit-learn.org/stable/ "Scikit-learn homepage"
[np]: http://numpy.org/ "Numpy homepage"
[scipy]: https://scipy.org/ "SciPy homepage"
[GPLv3]: https://www.gnu.org/licenses/gpl-3.0.en.html "The GNU General Public License Version 3"
[Ham2007]:https://www.researchgate.net/publication/221562215_Relational_Neural_Gas "Hammer, B. & Hasenfuss, A. (2007). Relational Neural Gas. Proceedings of the 30th Annual German Conference on AI (KI 2007), 190-204. doi:10.1007/978-3-540-74565-5_16"
[Neb2015]:https://doi.org/10.1016/j.neucom.2014.12.096 "Nebel, D., Hammer, B., Frohberg, K., & Villmann, T. (2015). Median variants of learning vector quantization for learning of dissimilarity data. Neurocomputing, 169, 295-305. doi:10.1016/j.neucom.2014.12.096"
