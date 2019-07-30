# Model Representation Tool

MRT, short for Model Representation Tool, is a deterministic quantization framework developed by Cortex that enables model inference in the limited-resource and strictly deterministic environment of blockchain, ushering in a new generation of AI smart contracts. MRT is designed to convert floating point models supported by nnvm into fixed-point models executable on the CVM while preventing significant loss of precision. 

The quantization method reduces the output number field of all layers of the model to INT8 or INT32 to simulate the floating-point network and converts the operators involved in the floating-point operation into integer operators using fuse and rewrite. Quantization ensures no overflow and guarantees the deterministic outcome of the model execution.

Currently, we only support models trained by MXnet framework.

## Converting

1. Use the MXNet framework to train the floating-point model.
2. Use MRT to convert your floating-point model to fixed-point model executable on the cvm.

