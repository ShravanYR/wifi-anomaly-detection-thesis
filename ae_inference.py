#!/usr/bin/env python3
"""
ae_inference.py
================
Pure Python 3 standard-library Autoencoder inference engine.
NO third-party packages required — no numpy, no torch, no sklearn.

This is intentional, not a shortcut: the trained models are tiny
(idle: ~120 parameters, active: ~3,500 parameters), this AP has no
internet access to install anything anyway, and at a 10-second polling
interval raw Python loop speed is not remotely a bottleneck.

WHAT THIS REPRODUCES, EXACTLY:
    For each Linear layer:      y = x @ W^T + b
    For each BatchNorm1d layer (eval mode, i.e. using running stats):
                                 y = (x - running_mean) / sqrt(running_var + eps)
                                     * gamma + beta
    ReLU:                       y = max(0, x)
    Dropout:                    skipped entirely (it is an identity at eval time)

    The StandardScaler step before the network is:
                                 y = (x - mean) / scale

    The final anomaly score is the mean squared error between the
    scaled input and the network's reconstruction of it — identical to
    the .reconstruction_error() method in the original PyTorch class.

USAGE (as a library):
    from ae_inference import AEModel
    model = AEModel.load('ae_weights_export.json', 'active')
    mse = model.score([0.5, 0.5, ...])   # list of 15 raw feature values, in order

USAGE (as a quick self-test from the command line):
    python3 ae_inference.py ae_weights_export.json
"""

import json
import math
import sys


def linear(x, weight, bias):
    """x: list[float], weight: list[list[float]] shape [out,in], bias: list[float]"""
    return [
        sum(w * xi for w, xi in zip(row, x)) + b
        for row, b in zip(weight, bias)
    ]


def batchnorm(x, layer):
    eps = layer['eps']
    return [
        (xi - m) / math.sqrt(v + eps) * g + b
        for xi, m, v, g, b in zip(
            x, layer['running_mean'], layer['running_var'],
            layer['gamma'], layer['beta']
        )
    ]


def relu(x):
    return [max(0.0, v) for v in x]


def run_sequential(x, layers):
    """Apply a list of exported layers in order, following each Linear's
    'activation' tag exactly as export_model_weights.py assigned it."""
    i = 0
    while i < len(layers):
        layer = layers[i]
        if layer['type'] == 'linear':
            x = linear(x, layer['weight'], layer['bias'])
            if layer['activation'] == 'relu':
                x = relu(x)
            elif layer['activation'] == 'relu_after_bn':
                # next layer is the batchnorm; apply it then ReLU
                i += 1
                x = batchnorm(x, layers[i])
                x = relu(x)
            # 'none' -> raw linear output, nothing further
        i += 1
    return x


def scale_input(x, scaler):
    return [(xi - m) / s for xi, m, s in zip(x, scaler['mean'], scaler['scale'])]


class AEModel:
    def __init__(self, encoder, decoder, scaler, threshold, feature_order):
        self.encoder = encoder
        self.decoder = decoder
        self.scaler = scaler
        self.threshold = threshold
        self.feature_order = feature_order   # e.g. ['rssi', 'tx_throughput_mbps', ...]

    @classmethod
    def load(cls, json_path, which):
        """which: 'idle' or 'active'"""
        with open(json_path) as f:
            data = json.load(f)
        return cls(
            encoder=data[f'{which}_model']['encoder'],
            decoder=data[f'{which}_model']['decoder'],
            scaler=data[f'{which}_scaler'],
            threshold=data[f'{which}_threshold'],
            feature_order=data[f'{which}_feature_order'],
        )

    def score_from_dict(self, feature_dict):
        """
        Build the feature vector in the EXACT order the model was trained on,
        directly from a {feature_name: value} dict — this is the safe way to
        call scoring, since it can never silently misorder features.
        """
        raw_features = [feature_dict[name] for name in self.feature_order]
        return self.score(raw_features)

    def reconstruction_error(self, raw_features):
        """raw_features: list[float] in the exact feature order used at training time."""
        x_scaled = scale_input(raw_features, self.scaler)
        latent = run_sequential(x_scaled, self.encoder)
        reconstructed = run_sequential(latent, self.decoder)
        mse = sum((a - b) ** 2 for a, b in zip(x_scaled, reconstructed)) / len(x_scaled)
        return mse

    def score(self, raw_features):
        """Returns (mse, is_anomaly: bool)."""
        mse = self.reconstruction_error(raw_features)
        return mse, mse > self.threshold


# ── Self-test entry point ──────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 ae_inference.py <ae_weights_export.json>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path) as f:
        data = json.load(f)

    idle_dim = len(data['idle_scaler']['mean'])
    active_dim = len(data['active_scaler']['mean'])

    idle_model = AEModel.load(path, 'idle')
    active_model = AEModel.load(path, 'active')

    test_idle = [0.0] * idle_dim
    test_active = [0.0] * active_dim

    mse_idle, anom_idle = idle_model.score(test_idle)
    mse_active, anom_active = active_model.score(test_active)

    print(f"Idle model   ({idle_dim}D)  : MSE={mse_idle:.6f}  threshold={idle_model.threshold}  anomaly={anom_idle}")
    print(f"Active model ({active_dim}D): MSE={mse_active:.6f}  threshold={active_model.threshold}  anomaly={anom_active}")
    print("\nSelf-test ran without errors — pure stdlib engine is working.")
