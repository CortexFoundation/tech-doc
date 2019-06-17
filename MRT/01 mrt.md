# Model Representation Tool

MRT, stands for Model Representation Tool, is a deterministic quantization framework where model inference can be enabled on the limited-resource and strictly deterministic environment of blockchain, enabling a new domain of smart contracts with machine learning models. MRT is used to connect to the AI framework API provided by the CVM. 

MRT is designed to convert the floating point model supported by nnvm into a fixed-point model that is executable on CVM, and to ensure sufficient loss of precision. The quantization method reduces the output number field of all layers of the model to INT8 or INT32 to simulate the floating-point network, and converts the operators involved in the floating-point operation into integer operators using fuse and rewrite. Quantization ensures no overflow and guarantees the deterministic outcome of the model execution.

