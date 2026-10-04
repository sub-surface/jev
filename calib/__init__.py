"""calib — small, tested core for calibrated-decision experiments.

scoring      proper scoring rules as torch losses (masked, mixed cardinality)
metrics      numpy evaluation: NLL, Brier, ECE variants (debiased, smooth), AURC, bootstrap CIs
recalibrate  temperature scaling, isotonic confidence maps
escalation   System-1 → System-2 routing curves: compute saved at matched quality
"""
