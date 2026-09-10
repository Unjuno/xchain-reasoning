# SPDX-License-Identifier: Apache-2.0
"""Observation-only Gaussian model diagnostics. No query truth enters this module."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve


@dataclass(frozen=True)
class Candidate:
    gate: float
    observed_precision: np.ndarray
    logdet: float
    output_coef: np.ndarray
    exact_output_coef: np.ndarray


def build_candidates(adjacency, query, tau, noise_variance, gates, steps=16):
    """Noise variances are supplied measurements, not learned from query labels."""
    a = np.asarray(adjacency, dtype=np.float64)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all()
            or not np.allclose(a, a.T) or np.any(np.diag(a) != 0)):
        raise ValueError('Require finite symmetric zero-diagonal adjacency.')
    n = len(a)
    if not isinstance(query, (int, np.integer)) or not 0 <= query < n:
        raise ValueError('Invalid query.')
    noise = np.asarray(noise_variance, dtype=np.float64)
    if (noise.shape != (n-1,) or not np.isfinite(noise).all()
            or np.any(noise <= 0) or not np.isfinite(tau) or tau <= 0 or steps < 1):
        raise ValueError('Invalid noise, tau or steps.')
    obs = np.delete(np.arange(n), query)
    diagonal = tau + np.abs(a).sum(1)
    outer = a.copy(); outer[query] = 0; outer[:, query] = 0
    near = a - outer
    prec = np.zeros(n); prec[obs] = 1/noise
    posterior_diagonal = diagonal + prec
    forcing = np.zeros((n, n-1))
    forcing[obs, np.arange(n-1)] = prec[obs] / posterior_diagonal[obs]
    result = []
    for gate in gates:
        if not np.isfinite(gate) or not 0 <= gate <= 1:
            raise ValueError('Each gate must be in [0,1].')
        candidate_a = near + gate * outer
        prior = np.diag(diagonal) - candidate_a
        sigma = cho_solve(cho_factor(prior, lower=True), np.eye(n))
        covariance = sigma[np.ix_(obs, obs)] + np.diag(noise)
        cf = cho_factor(covariance, lower=True)
        precision = cho_solve(cf, np.eye(n-1))
        logdet = float(2*np.log(np.diag(cf[0])).sum())
        transition = candidate_a / posterior_diagonal[:, None]
        coef = np.zeros_like(forcing)
        for _ in range(steps):
            coef = transition @ coef + forcing
        exact = precision @ sigma[obs, query]
        result.append(Candidate(float(gate), precision, logdet, coef[query].copy(), exact))
    return result


def diagnostic_scores(observations, candidates):
    """LOO NLPD, marginal NLPD (constants dropped), LOO squared error.

    All 48 observed nodes are held out one at a time in the Gaussian identity.
    The query node is never an observation. Scores may use all observed values;
    the prediction of a held-out node excludes that node's value.
    """
    y = np.asarray(observations, dtype=np.float64)
    if y.ndim != 2 or not np.isfinite(y).all() or not candidates:
        raise ValueError('Finite 2D observations and nonempty candidates required.')
    loo, evidence, mse = [], [], []
    for candidate in candidates:
        h = candidate.observed_precision
        if h.shape != (y.shape[1], y.shape[1]):
            raise ValueError('Observation dimension mismatch.')
        d = np.diag(h)
        projected = y @ h.T
        loo.append(.5*np.mean(projected**2/d - np.log(d), axis=1))
        evidence.append(.5*(np.sum(projected*y, axis=1)+candidate.logdet)/y.shape[1])
        mse.append(np.mean((projected/d)**2, axis=1))
    return {'loo_nlpd': np.stack(loo, axis=1),
            'evidence': np.stack(evidence, axis=1),
            'loo_mse': np.stack(mse, axis=1)}


def select_predictions(observations, candidates, scores):
    """Choose a gate independently for each field. Ties use candidate order."""
    index = np.argmin(scores, axis=1)
    coefficients = np.stack([c.output_coef for c in candidates])
    pred = np.einsum('ij,ij->i', observations, coefficients[index])
    return pred, index


def corrupt_relations(adjacency, query, kind, rng):
    """Return a perturbed graph; query-incident edges always remain correct.

    coherent50 applies a vertex gauge away from query and its neighbors. Every
    cycle remains balanced, although the relation to the true field is wrong.
    """
    a = np.asarray(adjacency).copy()
    edges = np.argwhere(np.triu(a != 0, 1))
    eligible = edges[np.all(edges != query, axis=1)]
    if kind == 'intact':
        return a
    if kind in ('flip15', 'flip50'):
        count = round((.15 if kind == 'flip15' else .5)*len(eligible))
        chosen = rng.permutation(len(eligible))[:count]
        for i, j in eligible[chosen]:
            a[i, j] *= -1; a[j, i] *= -1
    elif kind == 'coherent50':
        allowed = np.flatnonzero((a[query] == 0) & (np.arange(len(a)) != query))
        signs = np.ones(len(a))
        signs[rng.permutation(allowed)[:round(len(allowed)/2)]] = -1
        a *= signs[:, None]*signs[None, :]
    else:
        raise ValueError('Unknown corruption kind.')
    return a
