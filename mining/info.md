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





block interval   15s

tps   25.4



quota general   64k per block (model uploading space)

uploading network bandwidth   1MB/s

model mature   100 blocks

model size limit   1GB



pre alloc   149792458

total reward for mining   150000000

total   299792458

reward   2.5 per block (half every 4 years)  =  8409600



consensus pow cuckoo cycle