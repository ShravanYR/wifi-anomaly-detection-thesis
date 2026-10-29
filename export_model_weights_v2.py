#!/usr/bin/env python3
"""
export_model_weights.py
========================
Run this ONCE on your WORKSTATION (where PyTorch is already installed).
It converts the trained Autoencoder .pt files + scaler .pkl files into a
single plain JSON file containing nothing but numbers — no PyTorch object,
no pickle, no class definition. That JSON is the ONLY thing that needs to
be copied to the AP.

WHY THIS IS GENERIC (works for the idle model AND the active model without
any hardcoded architecture):
    nn.Dropout has no learnable parameters, so it never appears in
    state_dict() at all. That means every Sequential's state_dict only
    ever contains Linear and BatchNorm1d entries, in the exact order they
    run. We can tell them apart purely by key name:
        - a BatchNorm1d layer has a 'running_mean' key
        - a Linear layer does not
    And we know from the Autoencoder class definition (see
    autoencoder_training.py) that the activation rule is always:
        - Linear followed by a BatchNorm  -> apply Linear, BatchNorm, ReLU
        - the LAST Linear in "encoder"    -> apply Linear, ReLU   (no BN)
        - the LAST Linear in "decoder"    -> apply Linear only    (no activation)
    So the exporter only needs to walk state_dict() in order and apply
    that one rule — no per-model dimension hardcoding required.

USAGE:
    python export_model_weights.py

INPUT (expected in models_active/ directory):
    models_active/ae_idle_model.pt
    models_active/ae_active_model.pt
    models_active/ae_scaler_idle.pkl
    models_active/ae_scaler_active.pkl
    models_active/ae_metadata.json

OUTPUT:
    ae_weights_export.json   <- copy ONLY this file to the AP
"""

import os
import json
import pickle
import torch


def export_sequential(state_dict, prefix):
    """
    Walk a Sequential's parameters (encoder.* or decoder.*) in order and
    build a list of layer dicts: {type: 'linear'|'batchnorm', ...weights...}
    plus an 'activation' field telling the inference engine what to apply
    after this layer.
    """
    # Group keys by their numeric Sequential index, e.g. "encoder.0.weight" -> index 0
    indices = sorted({
        int(k.split('.')[1]) for k in state_dict if k.startswith(prefix + '.')
    })

    layers = []
    for idx in indices:
        base = f"{prefix}.{idx}."
        keys_here = [k for k in state_dict if k.startswith(base)]
        is_batchnorm = any(k.endswith('running_mean') for k in keys_here)

        if is_batchnorm:
            layers.append({
                'type'        : 'batchnorm',
                'gamma'       : state_dict[base + 'weight'].tolist(),
                'beta'        : state_dict[base + 'bias'].tolist(),
                'running_mean': state_dict[base + 'running_mean'].tolist(),
                'running_var' : state_dict[base + 'running_var'].tolist(),
                'eps'         : 1e-5,   # PyTorch nn.BatchNorm1d default, never changed in training
            })
        elif (base + 'weight') in state_dict:
            layers.append({
                'type'  : 'linear',
                'weight': state_dict[base + 'weight'].tolist(),  # shape [out, in]
                'bias'  : state_dict[base + 'bias'].tolist(),
            })
        # Dropout / ReLU have no parameters, so they produce no keys and are skipped here.
        # The activation that follows each Linear is determined below, not from state_dict.

    # Apply the activation rule from the Autoencoder class definition:
    # every Linear immediately followed by a batchnorm entry gets ReLU after the batchnorm;
    # a Linear with NO following batchnorm gets ReLU if this is the encoder's final layer,
    # or no activation at all if this is the decoder's final layer.
    for i, layer in enumerate(layers):
        if layer['type'] != 'linear':
            continue
        followed_by_bn = (i + 1 < len(layers) and layers[i + 1]['type'] == 'batchnorm')
        if followed_by_bn:
            layer['activation'] = 'relu_after_bn'   # handled by the BN entry that follows
        elif prefix == 'encoder':
            layer['activation'] = 'relu'            # encoder's final Linear -> ReLU
        else:
            layer['activation'] = 'none'            # decoder's final Linear -> raw output

    return layers


def export_model(pt_path):
    state_dict = torch.load(pt_path, map_location='cpu')
    return {
        'encoder': export_sequential(state_dict, 'encoder'),
        'decoder': export_sequential(state_dict, 'decoder'),
    }


def export_scaler(pkl_path, feature_order):
    """
    Apply a PER-FEATURE minimum floor to scale_, not one blind global value.

    Why per-feature: tx_throughput_mbps naturally ranges in the tens-to-
    hundreds (Mbps) -- a training scale near 0 there means the feature was
    almost perfectly constant in training, and any real live value will
    produce an absurdly inflated, uninformative z-score. But retry_rate or
    tx_success_rate are 0-1 ratios by design -- a small natural scale there
    (e.g. 0.02) is normal and IS the signal (it means small real changes in
    retry rate matter a lot). Flooring those up to the same large value as
    a throughput feature would make them nearly invisible to the model --
    exactly backwards from what we want.

    FEATURE_FLOORS below reflects each feature's own natural unit. Any
    feature not explicitly listed falls back to a conservative default.
    """
    FEATURE_FLOORS = {
        # rate-style counters (packets/sec, bytes/sec) -- small but real units
        'tx_packet_rate'    : 0.05,
        'rx_packet_rate'    : 0.05,
        'tx_byte_rate'      : 10.0,
        'rx_byte_rate'      : 10.0,
        # 0-1 ratios -- naturally small scale is correct, keep floor tiny
        'tx_success_rate'   : 0.01,
        'retry_rate'        : 0.01,
        'tx_failed_rate'    : 0.005,
        'low_mcs_ratio'     : 0.01,
        'mid_mcs_ratio'     : 0.01,
        'high_mcs_ratio'    : 0.01,
        # tens-to-hundreds scale features -- this is where the real bug was
        'rssi'              : 1.0,
        'tx_throughput_mbps': 2.0,
        'rx_throughput_mbps': 2.0,
        # small bounded-range scores
        'mcs_weighted_score': 0.1,
        'mcs_entropy'       : 0.02,
    }
    DEFAULT_FLOOR = 0.05   # safety net for any feature not explicitly listed

    with open(pkl_path, 'rb') as f:
        scaler = pickle.load(f)

    scale = []
    clipped = []
    for name, s in zip(feature_order, scaler.scale_):
        floor = FEATURE_FLOORS.get(name, DEFAULT_FLOOR)
        s = float(s)
        if s < floor:
            clipped.append((name, s, floor))
            s = floor
        scale.append(s)

    if clipped:
        print(f"  [WARN] {pkl_path}: {len(clipped)} feature(s) clipped to their floor:")
        for name, old, floor in clipped:
            print(f"           {name:20s}  {old:.6f} -> {floor}")

    return {
        'mean' : scaler.mean_.tolist(),
        'scale': scale,
    }


def main():
    out = {}

    idle_pt = 'models_active/ae_idle_model.pt'
    active_pt = 'models_active/ae_active_model.pt'
    idle_scaler_pkl = 'models_active/ae_scaler_idle.pkl'
    active_scaler_pkl = 'models_active/ae_scaler_active.pkl'
    meta_path = 'models_active/ae_metadata.json'

    # These MUST exactly match the order used in autoencoder_training.py's
    # IDLE_FEATURES / ACTIVE_FEATURES lists — embedding them here means the
    # live script on the AP reads the order from the model file itself
    # instead of relying on a second hand-typed copy that could drift.
    #
    # CORRECTED: idle features are [rssi, tx_success_rate, retry_rate] —
    # NOT [rssi, tx_throughput_mbps, rx_throughput_mbps]. During idle,
    # tx_success_rate and retry_rate are almost always exactly 0 (no
    # packets means 0/0 filled as 0), which is correct and expected —
    # it is not a sign of a broken feature.
    IDLE_FEATURE_ORDER = ['rssi', 'tx_success_rate', 'retry_rate']
    ACTIVE_FEATURE_ORDER = [
        'tx_packet_rate', 'rx_packet_rate', 'tx_byte_rate', 'rx_byte_rate',
        'tx_success_rate', 'retry_rate', 'tx_failed_rate', 'rssi',
        'tx_throughput_mbps', 'rx_throughput_mbps', 'low_mcs_ratio',
        'mid_mcs_ratio', 'high_mcs_ratio', 'mcs_weighted_score', 'mcs_entropy',
    ]

    print("[INFO] Exporting idle model...")
    out['idle_model'] = export_model(idle_pt)
    out['idle_scaler'] = export_scaler(idle_scaler_pkl, IDLE_FEATURE_ORDER)
    out['idle_feature_order'] = IDLE_FEATURE_ORDER

    print("[INFO] Exporting active model...")
    out['active_model'] = export_model(active_pt)
    out['active_scaler'] = export_scaler(active_scaler_pkl, ACTIVE_FEATURE_ORDER)
    out['active_feature_order'] = ACTIVE_FEATURE_ORDER

    # Sanity check: scaler dimension must match feature list length,
    # or something is already inconsistent before this ever reaches the AP.
    assert len(out['idle_scaler']['mean']) == len(IDLE_FEATURE_ORDER), \
        "Idle scaler dimension does not match IDLE_FEATURE_ORDER length!"
    assert len(out['active_scaler']['mean']) == len(ACTIVE_FEATURE_ORDER), \
        "Active scaler dimension does not match ACTIVE_FEATURE_ORDER length!"
    print("[INFO] Feature order embedded and verified against scaler dimensions.")

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        out['idle_threshold']   = meta.get('idle_threshold')
        out['active_threshold'] = meta.get('active_threshold')
        print(f"[INFO] Thresholds loaded: idle={out['idle_threshold']}, "
              f"active={out['active_threshold']}")
    else:
        print(f"[WARN] {meta_path} not found — thresholds not included. "
              f"Add 'idle_threshold' and 'active_threshold' to the output JSON manually.")

    out_path = 'ae_weights_export.json'
    with open(out_path, 'w') as f:
        json.dump(out, f)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\n[INFO] Done. {out_path} written ({size_kb:.1f} KB).")
    print(f"[INFO] This is the ONLY file you need to copy to the AP.")


if __name__ == '__main__':
    main()
