# Model Representation Tool

MRT, short for Model Representation Tool, is a deterministic quantization framework developed by Cortex that enables model inference in the limited-resource and strictly deterministic environment of blockchain, ushering in a new generation of AI smart contracts. MRT is designed to convert floating point models supported by nnvm into fixed-point models executable on the CVM while preventing significant loss of precision. 

The quantization method reduces the output number field of all layers of the model to INT8 or INT32 to simulate the floating-point network and converts the operators involved in the floating-point operation into integer operators using fuse and rewrite. Quantization ensures no overflow and guarantees the deterministic outcome of the model execution.

Currently we only support models trained by MXnet framework.

## Converting

1. Use the MXNet framework to train the floating-point model.
2. Use MRT to convert your floating-point model to fixed-point model executable on the cvm.

## Uploading

There are 3 phases to upload ml models on the Cortex blockchain:

###Upload Phase

mI models are treated as a special type of smart contract on the Cortex blockchain. Creators need to send a special transaction with a function call to advanced the upload progress. Each transaction will increase the file upload progress by 512K bytes, consuming the corresponding storage quota.

###Preparation Phase

After the completion of the upload phase, the file preparation phase is entered. This phase lasts for 100 blocks (about 25 minutes). 

###Mature Phase

After 100 blocks, the prepared files enter the mature phase. The model is saved on the storage layer of the Cortex blockchain.

## Broadcasting

After mature phase, the model owner is responsible for broadcasting the file to the network to reach the entire distributed file system; otherwise, the network consensus will reject relevant contract calls.  