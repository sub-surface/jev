# Cloud compute log

Every cloud run gets one row here, written before the run (plan) and completed after (actual). The budget was granted by the owner per request.

| Date | Experiment | Hardware | Plan (napkin) | Hard cap | Actual | Notes |
|---|---|---|---|---|---|---|
| 2026-10-04 | E007 amortised inference sweep | 1× Modal H100 80GB, one container | 4 models × 220 s + about 70 s overhead ≈ 950 s ≈ $1.04 | timeout 1300 s = $1.43 (granted: $1.50) | 901 s wall including a 45 s CPU image build, so about 860 s on the GPU ≈ **$0.95** | Data pool resident on the GPU, so the GPU was not starved. The smallest model ran at about 52 steps/s, about 7M tokens/s. |

**Pre-teardown spend (archive), for the record:** $5.02 logged in `archive/contracts/02-rlcd-decision/logs/budget_log.jsonl`.
