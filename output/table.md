# Final CNN–TCN benchmark summary

Cells report mean (minimum/maximum) across five trained networks. Bold indicates the highest mean within each holdout stratum and metric. Head F1 is not applicable to signal-only networks.

## Held-out animals

| Input | Objective | Correlation | Inhalation F1 from signal | Inhalation F1 from head |
|---|---|---:|---:|---:|
| Gray | Signal only | 0.8641 (0.8576/0.8722) | 0.7654 (0.7449/0.7920) | N/A |
| Gray | Signal + event detection | 0.8626 (0.8534/0.8759) | 0.7658 (0.7422/0.7877) | 0.9150 (0.9015/0.9306) |
| Gray + diff | Signal only | 0.9024 (0.8953/0.9058) | 0.8101 (0.7986/0.8242) | N/A |
| Gray + diff | Signal + event detection | 0.8998 (0.8927/0.9049) | 0.8125 (0.7999/0.8362) | 0.9382 (0.9319/0.9444) |
| Gray + flow | Signal only | 0.9013 (0.8992/0.9044) | 0.8394 (0.8320/0.8465) | N/A |
| Gray + flow | Signal + event detection | 0.9001 (0.8960/0.9045) | 0.8469 (0.8399/0.8512) | 0.9464 (0.9415/0.9522) |
| Gray + diff + flow | Signal only | 0.9072 (0.9043/0.9106) | 0.8438 (0.8359/0.8503) | N/A |
| Gray + diff + flow | Signal + event detection | **0.9084 (0.9056/0.9130)** | **0.8548 (0.8435/0.8656)** | **0.9562 (0.9511/0.9590)** |

## Held-out sessions

| Input | Objective | Correlation | Inhalation F1 from signal | Inhalation F1 from head |
|---|---|---:|---:|---:|
| Gray | Signal only | 0.8782 (0.8640/0.8862) | 0.7970 (0.7915/0.8052) | N/A |
| Gray | Signal + event detection | 0.8820 (0.8771/0.8873) | 0.8024 (0.7906/0.8099) | 0.9202 (0.9173/0.9282) |
| Gray + diff | Signal only | 0.9141 (0.9074/0.9214) | 0.8548 (0.8479/0.8622) | N/A |
| Gray + diff | Signal + event detection | 0.9101 (0.9067/0.9120) | 0.8702 (0.8644/0.8743) | 0.9403 (0.9383/0.9431) |
| Gray + flow | Signal only | 0.9099 (0.9076/0.9142) | 0.8719 (0.8665/0.8774) | N/A |
| Gray + flow | Signal + event detection | 0.9113 (0.9084/0.9141) | 0.8837 (0.8778/0.8891) | 0.9503 (0.9475/0.9529) |
| Gray + diff + flow | Signal only | 0.9151 (0.9113/0.9199) | 0.8787 (0.8755/0.8819) | N/A |
| Gray + diff + flow | Signal + event detection | **0.9168 (0.9134/0.9198)** | **0.8901 (0.8861/0.8947)** | **0.9515 (0.9483/0.9552)** |
