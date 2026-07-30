"""Schedulers Module.

WHY THIS MODULE EXISTS:
    Dynamically adjusts learning rate during training. Implements learning rate schedules
    such as Linear Warmup followed by Cosine Decay down to a specified minimum learning rate.

HOW FUTURE MODULES WILL PLUG IN:
    - Phase 13 (Training Loop): The trainer steps the scheduler after every optimizer step
      to update `learning_rate` dynamically.
"""
