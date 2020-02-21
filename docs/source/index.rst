.. ProtoDistML documentation master file, created by
   sphinx-quickstart on Thu Feb 20 22:04:49 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Prototype-based Machine Learning on Distance Data
=================================================

This `scikit-learn`_ compatible, Python3 library provides several algorithms
to learn prototype models on distance data. At this time, this library features
the following algorithms:

* Relational Neural Gas (`Hammer and Hasenfuss, 2007`_) for clustering,
* Relational Generalized Learning Vector Quantization (`Hammer, Hofmann, Schleif, and Zhu, 2014`_) for classification, and
* Median Generalized Learning Vector Quantization (`Nebel, Hammer, Frohberg, and Villmann, 2015`_) for classification.

If you intend to use this library in academic work, please cite the respective
reference paper.

Please consult the `project website <https://gitlab.ub.uni-bielefeld.de/bpaassen/proto-dist-ml>`_
for more detailed information about the project.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   rng
   rglvq
   mglvq


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _scikit-learn: https://scikit-learn.org/stable/
.. _Hammer and Hasenfuss, 2007: https://www.researchgate.net/publication/221562215_Relational_Neural_Gas
.. _Hammer, Hofmann, Schleif, and Zhu, 2014: http://www.techfak.uni-bielefeld.de/~fschleif/pdf/nc_diss_2014.pdf
.. _Nebel, Hammer, Frohberg, and Villmann, 2015: https://doi.org/10.1016/j.neucom.2014.12.096
