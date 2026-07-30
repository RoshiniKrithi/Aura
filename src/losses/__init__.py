"""Losses Module.

WHY THIS MODULE EXISTS:
    Computes optimization objective losses for language model training. Houses custom loss implementations
    such as Cross-Entropy Loss with label smoothing, target token masking, and focal loss variants.

HOW FUTURE MODULES WILL PLUG IN:
    - Phase 13 (Training Loop) & Phase 16 (Fine-Tuning): The trainer calls loss functions from this module
      to compute scalar loss tensors for backward pass backpropagation (`loss.backward()`).
"""
