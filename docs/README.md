# Dataset family docs

cpssbench groups loaders by cyber-physical modality. Each family has different semantics and data challenges:

| Family | Doc | Datasets | Distinct challenge |
| --- | --- | --- | --- |
| CAN | [CAN.md](CAN.md) | SynCAN, ROAD (+ planned X-CANIDS) | Intermittent bus gaps → `filling` + observation `mask` for Transformer-style training |
| V2X | [V2X.md](V2X.md) | MisbehaviorX (`vasp` / `veremi`) | Manual curated traces; dense windows; no CAN mask API |

Start from the [main README](../README.md) for install and the shared MNIST-style `(window, label)` contract.
