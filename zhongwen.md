MRT

mrt全名model representation tool，是cortex的模型定点化转换工具。mrt用于对接cortex虚拟机cvm提供的AI框架API，旨在将nnvm支持的浮点模型转换为cvm可执行的全整形定点化模型，并保证足够少的精度损失。量化方法主要是将模型所有layer的输出数域放缩到INT8或者INT32范围，来模拟浮点网络，通过fuse和rewrite等方法，将涉及到浮点操作的算子转换为全整形算子；另一方面，量化网络确保整形范围操作的非溢出，保证模型执行的确定性。

目前我们mrt只支持转换成定点化模型的功能，现在流程是使用mxnet框架训练好浮点模型，然后使用我们的mrt转换成cvm可执行的定点化模型。

CVM

CVM 全称 Cortex Virtual Machine， 由原 Ethereum 项目的 EVM 移植过来，在此基础上添加了对 Cortex 链上 AI 推断功能的支持。该功能主要分为两个模块，一方面是在 CVM 中支持了 Inference 指令，包含 Infer(code: 0xc0), InferArray(code: 0xc1)，前端如remix编译合约时同步增加一条infer函数来封装 CVM 接口，实现链上 AI 推断功能；另一方面，Synapse 作为定点化 AI 推断引擎，又名 CVM Executor，可以在异构计算环境下保证 AI 推断结果的一致性。Synapse 提出了模型定点化执行框架，并提出了相对应的确定性机器学习算子库，有兴趣的 AI 开发人员可以基于目前提出的框架训练或者量化。为保证解耦性，Synapse 是作为基于 CVM 的 device 形式存在的，是一个单独的模块，可以将其看作是 CVM 将需要Inference的模型和数据放入一块缓存，以中断的形式调用 Synapse 接口。



Cortex Remix

Cortex Remix 是 Cortex 智能合约编程语言 Solidity IDE, 是基于 Remix-IDE 开发的智能合约 IDE，支持 AI 智能合约的编译与部署，支持对transaction的 debug

Cortex Remix 主要包含两个功能模块，一是 AI 智能合约的编译模块，二是 AI 智能合约的部署模块。编译模块支持 AI 智能合约的编译，优化，可获取编译生成的 abi, bytecode 等信息；合约部署模块在 Cortex Wallet 的支持下，能够将合约部署至 Cortex 区块链网络，从而实现对链上 AI model 的调用



Storage

链上AI推断依赖于AI模型和推断数据文件，这部分数据的存储对传统区块链的协议传输层提出了很大的挑战。Cortex使用了基于DHT（分布式哈希表）的分布式文件系统作为存储层的解决方案，以减轻网络负载压力和降低网络传输开销。存储配额在Cortex链上被视作一种资源，每个区块被挖掘后将提供64K字节的存储配额。用户对存储配额的使用将按照交易费用自由竞价。

AI模型或者推断数据在链上被视为一种特殊的智能合约，创建者需要向合约发送空交易以推进上传的进度，每笔交易将提高文件上传的完成度512K字节，这将消耗对应的存储配额。文件上传完成后进入准备期，这个时间将持续100个区块，约25分钟。文件准备期结束后进入成熟期，可供AI推断合约调用。

AI模型的合约调用会给予AI模型所有者奖励。上传者有义务对文件在网络中进行广播，以分发到整个分布式文件系统，否则相关的合约调用将被网络共识拒绝。



Miner:

1. 算法：Cuckoo cycle ( cuckaroo30 )
2. 算法介绍： https://github.com/CortexFoundation/PoolMiner/blob/master/README.md
3. 硬件要求：Nvidia gpu, 可用显存 >= 10.7G, 1080ti, 2080ti, titan V…
4. 系统 : linux ubuntu 18.04
5. Cuda版本: 9.2+ ;  驱动版本: 396+
6. 编译源码需要：go1.10+,  gcc 5.4+
7. 安装步骤
   1. git clone git@github.com:CortexFoundation/PoolMiner.git
   2. make
   3. 运行：./build/bin/cortex_miner -pool_uri=192.168.50.5:8008 -devices=1,2
      a. pool_uri: 矿池地址 ip:port
      b. -devices: gpu编号





- Quota general: 64k per block (model uploading space)
- Uploading network bandwidth: 1MB/s
- Model mature: 100 blocks
- Model size limit: 1GB
- TPS: 25.4
- Pre-allocation: 149792458
- Total reward for mining: 150000000
- Total supply: 299792458
- Reward: 2.5 per block (half every 4 years)  =  8409600



