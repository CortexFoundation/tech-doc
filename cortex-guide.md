# Cortex Theseus

[TOC]

## Introduction

The inspiration of CortexLabs starts from the development of virtual machine in Ethereum project, which can execute simple contract logic. And further more, can it execute the AI on blockchain? The absolute answer is YES! That is the **AI on Blockchain** @CortexLabs.

Firstly, welcome to visit our offical website: https://www.cortexlabs.ai.

And then , we will give an total review of our foundation structure. All the whole projects of CortexLabs are arranged in the below image. You can have a quick glance at the CortexLabs' global mind map, and then find the corresponding sections in the documentation from the picture.

![XMind](imgs/CortexTheseus.png)

## Documentation

The main introduction to CortexLabs project is located at [cortex documentation](README.md).

CortexLabs chain-node GitHub project is named [**CortexTheseus**](https://github.com/CortexFoundation/CortexTheseus).

## Quick Demo

This is an preview demo to show what  CortexLabs is doing with AI, edge devices, and blockchain. Refers to [Apple Demo](apple-demo.md) to learn more.

![demo](cvm/demo/demo.gif)

## Start Up

The Cortex Theseus project start-up documentation is located in this [link](clients.md).

## Cortex Release

We compiled and organized many necessary binaries for people who do not want to access the source code.  the Cortex Release project is located in this [link](https://github.com/CortexFoundation/Cortex_Release).

## Blockchain

### Mining

Mining documentation, [mining](mining.md).

- Cuckoo Cycle Algorithm

  Cortex Theseus Client uses [cuckoo cycle](https://github.com/CortexFoundation/PoolMiner/blob/dev/README.md) as mining algorithm.

- Pool Miner

  We have published the mining pool source code in [GitHub](https://github.com/CortexFoundation/PoolMiner).

### Account Balance

Cortex Theseus inherits numerous traits from the [Ethereum](https://github.com/ethereum/go-ethereum) project (which is inspired by [Bitcoin](https://github.com/bitcoin/bitcoin)). Hence a wallet address in the blockchain have the concept of balance that can use to transfer  CTXC in between accounts.

### Currency: CTXC

CTXC is the native token used within Cortex. Refer to this [link](ctxc.md) for more details.

### Endorphin / Gas

Payment for transaction packed by miner in blockchain is calledEndorphin, or Gas as called in Ethereum. Refer to this [link](endorphin.md) for more details.

### JSON RPC

Cortex Full Node(Cortex Theseus) can supply JSON RPC interface, its functionality is similar to the [RPC of Ethereum](https://github.com/ethereum/wiki/wiki/JSON-RPC) except that the prefix `eth_` should be replaced with `ctxc_` if method contains. 

You should enable the JSON RPC in the Cortex Theseus Node with commands:

```bash
--rpc --rpcapi ctxc,web,admin,txpool --rpcaddr 0.0.0.0 --rpcport 38888
```

The official JSON RPC documentation is still in progress. You can have a glance through this [link](http://ec2-18-191-10-249.us-east-2.compute.amazonaws.com:5000/).

### Node Peers

We have collected all the running nodes peers from global MainNet network. 

Refer to this [link](https://github.com/CortexFoundation/discv4-dns-lists) for more detial .

## Torrent File Storage

Cortex Theseus has a distributed file system storing the model and input data. The storage layer is named Torrent FS. 

Rrefer to this [link](storage-layer.md) for more detail.

### Model & Input

- Available Models on Chain

  We have supply some pre-quantized model on the Cortex blockchain including image classification and NLP category. More details [here](model-list.md).

- Model & Input Upload

  Model & Input upload process require a few steps. Refer to this [link](model-data-upload.md) if you are interested in uploading models and on-chain AI contract call.

## Cortex Virtual Machine

Cortex Virtual Machine is inherited from EVM in Ethereum project with added **important and powerful** features. Exposition of CVM is described [here](cvm.md).

### AI Contract

- Contract Deployment refers to the [link](ai-contracts.md).
- AI Dapps refer to the [link](ai-dapps.md).
- Contract Editor: Remix refer to the [link](cortex-remix.md).
- Solidity Project refers to the [link](https://github.com/CortexFoundation/ctxc-solc).

### Synapse / CVM Executor

Synapse is a go-lang level wrapper of cvm-runtime. We supply portable usage for GPU-lack devices like remote inference without native AI inference. 
- Local Inference (To be continued)
- Remote Inference (To be continued)

## CVM Runtime

CVM Runtime is a deterministic AI framework like TensorFlow, Caffe, MxNet, etc. We have sperated the project from CortexTheseus as an independent repository and have completed many features AI developers want such as high-level python interface, graph model support, etc.

### Documentation

- operator's OPS refer to the [link](https://github.com/CortexFoundation/cvm-runtime/blob/wlt/docs/cvm/ops.md).

### Github Project

The cvm-runtime github url is https://github.com/CortexFoundation/cvm-runtime/.

**Notice**: Many new features under development is in branch wlt, that will be merged into master in the future.

### Formalization

We have researched the operator formalization of cvm-runtime, using `Z3-Prover` to verify the operator's deterministic process logic. All the verification source code and records have been uploaded in the [GitHub link](https://github.com/CortexFoundation/z3_prover).

## Model Representation Tools

MRT is a representation toolset for transforming the floating AI model into a full integer and non-data-flow model that CVM Runtime can accept. More information can be found in this [link](mrt.md).

### Documentation

- Install and Introduction, [link](https://github.com/CortexFoundation/cvm-runtime/blob/wlt/docs/mrt/README.md).
- MNist train and quantize tutorial,  [link](https://github.com/CortexFoundation/cvm-runtime/blob/wlt/docs/mrt/mnist_tutorial.md).
- Pre-quantized Model Information, [link](https://github.com/CortexFoundation/cvm-runtime/blob/wlt/docs/mrt/model.md).
- MRT Start and API, [link](https://github.com/CortexFoundation/cvm-runtime/blob/wlt/docs/mrt/mrt.md)

### Github Project

The MRT code is intergral in the [cvm-runtime](https://github.com/CortexFoundation/cvm-runtime/tree/wlt) project. Mainly located at the directory `python/mrt`.
