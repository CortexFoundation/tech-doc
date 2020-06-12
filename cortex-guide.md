# Cortex Theseus

[TOC]

## Introduction

The inspiration of CortexLabs starts from the development of virtual machine in Ethereum project, which can execute simple contract logic. And further more, can it execute the AI on blockchain? The absolute answer is YES! That is the **AI on Blockchain** @CortexLabs.

Firstly, welcome to visit our offical website: https://www.cortexlabs.ai.

And then , we will give an total review of our foundation structure. All the whole projects of CortexLabs are arranged in the below image. You can have a quick glance at the CortexLabs' global mind map, and then find the corresponding sections in the documentation from the picture.

![XMind](imgs/CortexTheseus.png)

## Documentation

The main introduction to @CortexLabs project is located at [cortex documentation](README.md).

CortexLabs chain-node GitHub project is named [**CortexTheseus**](https://github.com/CortexFoundation/CortexTheseus).

## Quick Demo

This is an simple demo for quick learning that what @CortexLabs can do with AI, edge devices and blockchain. More Information about the video refers to [Apple Demo](apple-demo.md).

![demo](cvm/demo/demo.gif)

## Start Up

The Cortex Theseus project start-up documentation is located at the [link](clients.md).

## Cortex Release

Since our mainnet published on the time plan from roadmap, we compiled and organized many neccessary binaries for people do not want to access the source code.  the Cortex Release project is located in the [link](https://github.com/CortexFoundation/Cortex_Release).

## Block Chain

### Mining

Mining documentation refers to [mining](mining.md).

- Cuckoo Cycle Algorithm

  Cortex Theseus Client use the [cuckoo cycle](https://github.com/CortexFoundation/PoolMiner/blob/dev/README.md) as mining algorithm.

- Pool Miner

  We have published the mining pool source code in [GitHub](https://github.com/CortexFoundation/PoolMiner).

### Account Balance

Cortex Theseus inherits from the [Ethereum](https://github.com/ethereum/go-ethereum) project which starts from [Bitcoin](https://github.com/bitcoin/bitcoin). Hence a wallet address in chain have the balance concept and could transfer the CTXC between accounts.

### Currency: CTXC

CTXC is the native token used within Cortex. More details refer to [link](ctxc.md).

### Endorphin / Gas

Payment for transaction packed to miner in blockchain is named Endorphin, or Gas as called in Ethereum. More details refer to [link](endorphin.md).

### JSON RPC

Cortex Full Node(Cortex Theseus) can supply JSON RPC interface, it's same as the [RPC of Ethereum](https://github.com/ethereum/wiki/wiki/JSON-RPC) generally except that the prefix `eth_` should be replaced with `ctxc_` if method contains. 

You should enable the JSON RPC in the Cortex Theseus Node with commands:

```bash
--rpc --rpcapi ctxc,web,admin,txpool --rpcaddr 0.0.0.0 --rpcport 38888
```

The official JSON RPC documentation is still in progress and you could have a glance at the [link](http://ec2-18-191-10-249.us-east-2.compute.amazonaws.com:5000/).

### Node Peers

We have collected all the running nodes peers from global MainNet network. 

More information refers to the [link](https://github.com/CortexFoundation/discv4-dns-lists).

## Torrent File Storage

Cortex Theseus have designed a distributed file system storing the model and input data. The storage layer is named Torrent FS. 

More details refer to the [link](storage-layer.md).

### Model & Input

- Available Models on Chain

  We have supply some pre-quantized model in chain including from image classification to NLP category. More details refer to [link](model-list.md).

- Model & Input Upload

  Model & Input upload process have many steps to go through. You may look at the [link](model-data-upload.md) if you are interested in model supply and on-chain AI contract call.

## Cortex Virtual Machine

Cortex Virtual Machine is inherited from EVM in Ethereum project and add some new **important and powerful** features. Exposition of CVM is described in [here](cvm.md).

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

CVM Runtime is a deterministic AI framework same as TensorFlow, Caffe, MxNet, etc. We have sperated the project from CortexTheseus as an independent item and have completed many features AI developers want such as high-level python interface, graph model support, etc.

### Documentation

- operator's OPS refer to the [link](https://github.com/CortexFoundation/cvm-runtime/blob/wlt/docs/cvm/ops.md).

### Github Project

The cvm-runtime github url is https://github.com/CortexFoundation/cvm-runtime/.

**Notice**: Many new features are developed in branch wlt, it would be merged into master in future.

### Formalization

We have researched the operator formalization of cvm-runtime, using `Z3-Prover` to verify the operator's deterministic process logic. All the verification source code and record have been uploaded in the [GitHub Link](https://github.com/CortexFoundation/z3_prover).

## Model Representation Tools

MRT is a representation tool set for transforming the floating AI model into a full of integer and non data-flow model that CVM Runtime can accept. More information can refer to the [link](mrt.md).

### Documentation

- Install and Introduction refer to the [link](https://github.com/CortexFoundation/cvm-runtime/blob/wlt/docs/mrt/README.md).
- Mnist train and quantize tutorial refer to the [link](https://github.com/CortexFoundation/cvm-runtime/blob/wlt/docs/mrt/mnist_tutorial.md).
- Pre-quantized Model Information refers to the [link](https://github.com/CortexFoundation/cvm-runtime/blob/wlt/docs/mrt/model.md).
- MRT Start and API refer to the [link](https://github.com/CortexFoundation/cvm-runtime/blob/wlt/docs/mrt/mrt.md)

### Github Project

The MRT code is intergral in the [cvm-runtime](https://github.com/CortexFoundation/cvm-runtime/tree/wlt) project. Mainly located at the directory `python/mrt`.



