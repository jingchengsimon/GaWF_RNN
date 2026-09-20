# Clean-room BRIMs provenance

This directory contains no source copied from the upstream BRIMs repository.
The project implementation lives in `utils/training/recurrent_cores/brims.py`
and was written from the architecture and equations in Mittal et al. (2020),
*Learning to Combine Top-Down and Bottom-Up Signals in Recurrent Neural
Networks with Attention over Modules* (arXiv:2006.16981).

The official repository at https://github.com/sarthmit/BRIMs was inspected at
commit `f8af67e863ea751b45b70cc7a7b91fb277beb329` only to identify the published
MNIST command structure (`2` layers, `num_blocks 6 3`, `topk 4 2`) and its
public interface. It has no license grant, so none of its source is vendored or
copied. The clean-room implementation keeps the paper-defined bottom-up/current
and top-down/previous-timestep attention, sparse module activation, independent
module recurrence, and within-layer communication.

The clean-room implementation also fixes the attention dimensions to those observed in the
published MNIST core: input attention uses 4 heads with `d_k=64` and a concatenated value width
of four times the module width; within-layer communication uses 4 heads with `d_k=d_v=32`.
