FisherVectorEmbedder
====================

The Fisher Vector solves both problems of the BoW model. First, it encodes
higher-order statistics, such as the first and optionally second-order
differences, instead of just counting the occurrences of visual words like BoW.
This method is derived from the Fisher kernel framework, which describes a
sample set's deviation from an average distribution. Secondly, the distribution
of the local descriptors, unlike BoW and VLAD, is modeled by a Gaussian Mixture
Model. This mitigates the hard assignment problem introduced by the K-Means
algorithm, since each descriptor is assigned to multiple Gaussian components
with different probabilities.

Computation
-----------

Fisher kernel framework
~~~~~~~~~~~~~~~~~~~~~~~

Given a set of :math:`T` local descriptors
:math:`X = \{x_t; t = 1, \ldots, T\}` extracted from an image, it is assumed
that the generation process of :math:`X` can be modeled by an image-independent
probability density function :math:`u_{\lambda}` with parameters
:math:`\lambda` [Jégou et al., 2012]. The gradient vector
:math:`G^{X}_{\lambda}` is obtained by computing the gradient of the
log-likelihood of the sample set :math:`X` with respect to the parameters
:math:`\lambda`:

.. math::

   G^{X}_{\lambda} = \frac{1}{T} \nabla_{\lambda} \log u_{\lambda}(X)

where :math:`G^{X}_{\lambda}` describes the contribution of the parameters to
the generation process [Perronnin & Dance, 2010].

The Fisher kernel is then defined as:

.. math::

   K(X, Y) = (G^{X}_{\lambda})^T F_{\lambda}^{-1} G^{Y}_{\lambda}

where :math:`F_{\lambda}` is the Fisher information matrix, defined by:

.. math::

   F_{\lambda} = \mathbb{E}_{x \sim u_{\lambda}} \left[ \nabla_{\lambda} \log u_{\lambda}(x) \nabla_{\lambda} \log u_{\lambda}(x)^T \right]

:math:`\mathcal{G}^{X}_{\lambda}` is the Fisher Vector after applying the
Cholesky decomposition on :math:`F_{\lambda}^{-1} = L_{\lambda}^T L_{\lambda}`,
and is computed as:

.. math::

   \mathcal{G}_i^X = L_{\lambda} G^{X}_{\lambda}

Fisher Vector computation
~~~~~~~~~~~~~~~~~~~~~~~~~

As discussed, the Fisher Vector encodes each descriptor to multiple Gaussian
components (also called "soft assignment"). The probability of a descriptor
:math:`x_t` belonging to the :math:`i`-th Gaussian is computed with the
Gaussian Mixture Model.

The Gaussian Mixture Model is chosen for
:math:`u_{\lambda}(x) = \sum_{i=1}^{K} w_i u_i(x)`, where
:math:`w_i, \mu_i, \Sigma_i` are the mixture weights, mean vectors, and
variance matrices of the Gaussian :math:`u_i`. The Fisher Vector is then
computed as:

.. math::

   \gamma_t(i) = \frac{w_i u_i(x_t)}{\sum_{j=1}^{K} w_j u_j(x_t)}

.. math::

   \mathcal{G}_i^X = \frac{1}{T \sqrt{w_i}} \sum_{t=1}^{T} \gamma_t(i)\, \sigma_i^{-1} (x_t - \mu_i)

where:

- :math:`\gamma_t(i)` is the soft assignment of descriptor :math:`x_t` to the
  :math:`i`-th Gaussian.
- :math:`w_i`, :math:`\mu_i`, and :math:`\Sigma_i` are the mixture weight, mean
  vector, and covariance matrix of the :math:`i`-th Gaussian component.

The final Fisher Vector :math:`G^{X}_{\lambda}` is the concatenation of the
vectors :math:`G^{X}_{i}` for :math:`i = 1, \ldots, K`, resulting in a
:math:`K \times d`-dimensional vector. This vector captures both the occurrence
and distributional properties of the local descriptors.

The resulting vector has shape ``(2 * K * D + K,)``, where ``K`` is the number
of GMM components and ``D`` is the local descriptor dimension (after optional
PCA).

Usage
-----

.. code-block:: python

   from pyvisim.classic import FisherVectorEmbedder

   fisher = FisherVectorEmbedder(
       n_components=256,                # number of mixture components
       gmm_params={"rng": 0},           # forwarded to the GMM
       pca_params={"n_components": 64}, # optional, omit for no PCA
   )
   fisher.learn(images)                 # fits the PCA (if any) then the GMM

   embedding = fisher.embed(image)      # Embed image into a Fisher Vector

   # Cosine similarity between two images
   similarity = fisher.similarity_score(image1, image2)

   fisher.save_to_disk("fisher.embedder")   # Save the embedder to disk

   # Load the embedder from disk
   fisher = FisherVectorEmbedder.load_from_disk("fisher.embedder")

GMM parameters (``gmm_params``)
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Meaning
   * - ``n_init``
     - ``1``
     - Number of k-means++ seeded EM runs. The run with the highest final
       log-likelihood is kept. Raise it for better, more stable vocabularies.
   * - ``max_iter``
     - ``100``
     - Maximum number of EM iterations per run.
   * - ``tol``
     - ``1e-3``
     - Convergence threshold: a run stops when the change of the mean
       per-sample log-likelihood between iterations falls below it.
   * - ``reg_covar``
     - ``1e-6``
     - Non-negative regularisation added to (and floored on) the per-feature
       variances, keeping them strictly positive when a component collapses or
       dies.
   * - ``rng``
     - ``None``
     - Seed (``int``) or :class:`numpy.random.Generator` for reproducible
       fitting.

PCA parameters (``pca_params``)
-------------------------------

See :doc:`PCA <../pca/pca>`.

References
----------

- H. Jégou et al. "Aggregating Local Image Descriptors into Compact Codes". In:
  IEEE Transactions on Pattern Analysis and Machine Intelligence 34.9 (2012),
  pp. 1704-1716. doi: 10.1109/TPAMI.2011.235.

API reference
-------------

.. autoclass:: pyvisim.classic.FisherVectorEmbedder
   :members:
   :inherited-members:
   :show-inheritance:
