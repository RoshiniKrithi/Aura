"""Optimizers Module.

WHY THIS MODULE EXISTS:
    PyTorch parameters require gradient-based updates using optimization algorithms like AdamW.
    To adhere to first-principles learning, custom optimizers (e.g. AdamW with decoupled weight decay)
    are implemented directly in PyTorch tensors within this module.

HOW FUTURE MODULES WILL PLUG IN:
    - Phase 13 (Training Loop): Trainer instantiates optimizers from this module, separating parameters
      that receive weight decay (weights) from parameters that do not (biases, LayerNorm weights).
"""
