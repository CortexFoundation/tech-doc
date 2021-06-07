# **Ubuntu**

## 基础环境 (NVIDIA GeForce GTX 1080 Ti is needed)

#### `cmake 3.11.0+`

```shell
wget https://cmake.org/files/v3.11/cmake-3.11.0-rc4-Linux-x86_64.tar.gz
tar zxvf cmake-3.11.0-rc4-Linux-x86_64.tar.gz
sudo mv cmake-3.11.0-rc4-Linux-x86_64  /opt/cmake-3.11
sudo ln -sf /opt/cmake-3.11/bin/*  /usr/bin/
```

#### **`go 1.16+`**

```shell
wget https://dl.google.com/go/go1.16.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.16.linux-amd64.tar.gz
echo 'export PATH="$PATH:/usr/local/go/bin"' >> ~/.bashrc
source ~/.bashrc
```

#### **`gcc/g++ 5.4+`**

```shell
sudo apt install gcc
sudo apt install g++
```

#### **`cuda-9.2`**

[下载地址](https://developer.nvidia.com/cuda-toolkit-archive)

```shell
export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64/:/usr/local/cuda-9.2/lib64/stubs:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda-9.2/lib64/:/usr/local/cuda-9.2/lib64/stubs:$LIBRARY_PATH
```

![image-20210325151317036](/Users/yangchunxueya/Library/Application Support/typora-user-images/image-20210325151317036.png)

#### **nvidia驱动下载**

[下载地址](https://www.nvidia.cn/Download/index.aspx?lang=cn)

根据显卡类型进行下载安装。



## 代码

#### 编译代码

```shell
git clone --recursive https://github.com/CortexFoundation/CortexTheseus.git
cd CortexTheseus
make clean && make -j$(nproc)

make mine
```

#### 创建账户

```shell
./build/bin/cortex account new
```

> 记录地址，下面会使用到

#### 开始运行

```shell
./build/bin/cortex --mine --miner.threads=1 --miner.coinbase="{you account}" --miner.cuda
```

