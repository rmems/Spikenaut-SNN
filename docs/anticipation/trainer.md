# Julia forecasting readout trainer

The pilot trainer consumes `anticipation-prepared-v1` from the ETL. Frame indices
are zero-based. Every valid frame drives the network; only listed examples
contribute supervised loss or predictions. Input normalization and target
normalization are frozen training statistics from the prepared artifact.

Run against the existing read-only local Julia environments:

```sh
JULIA_LOAD_PATH=/path/to/SynapticDistill.jl:/path/to/SynapticDistill.jl/scripts:@stdlib \
  julia +1.12.7 --startup-file=no tools/anticipation/test_train.jl
JULIA_LOAD_PATH=/path/to/SynapticDistill.jl:/path/to/SynapticDistill.jl/scripts:@stdlib \
  julia +1.12.7 --startup-file=no tools/anticipation/train.jl prepared.json output/snn 1200
```

The checked-in `Project.toml` also supports a separately instantiated environment.
Develop SynapticDistill into that environment using Julia's `Pkg.develop(path=...)`
with your local checkout and instantiate before the timed experiment. Dependency
installation and the deterministic fixture belong to experiment preparation.

## Network and learning

Each seed (123, 456, 789) initializes a shared 16×5 input matrix and 4×16 readout
matrix for uniform and mixed arms. Input weights are Gaussian with standard
deviation `0.6/sqrt(5)`. Readout weights initially have standard deviation 0.02.
Hidden weights stay fixed. Membrane time constants are 0.5 seconds in the uniform
arm and four neurons each at 0.1, 0.5, 2, and 5 seconds in the mixed arm.

At every 0.1-second tick, each neuron integrates
`v = exp(-dt/tau)*v + (1-exp(-dt/tau))*(1.2 + W_input*x)`.
A neuron emits a binary spike when `v >= 1`, then subtracts the threshold. There
is no recurrence, learned hidden bias, input clipping, or stimulus-schedule input.
The fixed 1.2 drive gives the standardized signed inputs a baseline firing regime.
These details are included in each checkpoint.

Readout features are normalized exponential spike traces with a 0.5-second time
constant: `pre = exp(-0.1/0.5)*pre + spikes`,
`features = (1-exp(-0.1/0.5))*pre`. The four physical-unit outputs are reconstructed
from the training target means and standard deviations. The output ordering is
`temperature_delta_1s_c`, `power_delta_1s_w`, `temperature_delta_5s_c`,
`power_delta_5s_w`.

Training calls SynapticDistill's actual `update_ottt!` with a four-by-time logits
matrix and squared-error loss, pairing every target with its own trace. It asserts
that the library reports time-resolved operation. Batches contain up to 64 ticks.
Masked warmup ticks retain state but contribute no loss. Mask compensation and
the library's time averaging together yield mean squared error over the eligible
examples and four outputs, without averaging over time twice. The learning rate
is 0.01. A 16-tick tail is split into 15+1 because the library rejects ambiguous
square spike matrices; no ticks are invented or dropped.

Membrane and trace state reset for every session, segment, invalid frame, and
frame-index discontinuity, and at each epoch. Evaluation runs with independent
local state and never updates weights. Future targets only enter the supervised
loss; held-out target values never enter state evolution or predictions.

## Deadline and checkpoint selection

The six runs advance in round-robin epoch order, with a maximum of 20 epochs.
Each completed epoch is scored only on validation: session-weighted mean of the
two five-second absolute errors divided by their training target standard
deviations. Only improvements replace the saved checkpoint. No test errors
participate in selection. The selected checkpoint is reloaded for held-out output.

The one shared monotonic wall-clock budget starts before package imports. The first 85%
is available to training; 15% is reserved for reloaded validation/test predictions.
Deadline checks occur at batch boundaries. An already-running Julia operation or
artifact write is not forcibly interrupted, so the summary records actual elapsed
time. No additional epochs are granted after the deadline. An interrupted epoch
cannot replace the selected checkpoint.

`summary.json` records all six planned runs with `complete`, `budget_limited`,
`unfinished`, or `unfinished_evaluation` status. `budget_limited` means a selected
checkpoint and held-out predictions exist but fewer than 20 epochs completed;
`unfinished_evaluation` means a checkpoint exists but held-out evaluation did not
finish. These are incomplete comparisons for campaign reporting.

Each arm directory contains `learning_curves.json`, and, when available,
`checkpoint.json` and `predictions.json`. Curves contain online training MSE
and validation MSE in standardized units alongside the validation primary score.
Training MSE uses the changing readout as it learns within that epoch.
Prediction records contain session ID,
frame index, split, and four physical-unit predictions. Diagnostics give per-neuron
spikes and firing rates, silent neurons, and time-resolved training evidence in
the learning curves. Checkpoints preserve the feature map, normalization, target
ordering, source provenance, and prepared-file SHA-256.
