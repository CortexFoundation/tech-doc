# Cortex Virtual Machine (CVM)
The Cortex Virtual Machine (CVM), is ported from the Ethereum Virtual Machine (EVM) with added support for AI inference. The CVM is backward-compatible with EVM and capable of running both traditional smart contracts and AI smart contracts. 

The CVM has two main modules:  deterministic inference engine and inference instruction sets. 

The deterministic inference engine is called Synapse or the CVM Executor. It guarantees the consistency of AI inference results in heterogeneous computing environments, without significantly compromising performance and accuracy. Synapse proposes a model-based fixed-point execution framework and a corresponding deterministic machine learning operator library. AI developers can train and quantize their models using MRT to be executable on CVM. 

The instruction sets allow smart contracts to include and execute AI models. AI models can be called in contracts through instruction sets, including Infer (code: 0xc0), InferArray (code: 0xc1). 

