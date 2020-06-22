# Write & Upload AI Models to Cortex

# Introduction

Cortex is currently the _only_ public blockchain that allows on-chain execution of machine learning models. Every step of the inference is transparent to the public, verified by blockchain consensus.

A few vocabulary essential for the rest of the tutorial:

**CVM-Runtime**:

An open-source deterministic machine learning framework written in C++. During runtime, the CVM (Cortex Virtual Machine) executes your models via CVM-Runtime. It is comparable to other frameworks such as MXNet, TensorFlow etc. with the important distinction that it is deterministic.

**MRT**:

MRT (written in Python) is one of the most important parts of CVM-Runtime. It quantizes and prepares your model for on-chain execution by CVM-Runtime. At the risk of oversimplification, MRT compresses and converts your model from floating point to integer only so that its execution is efficient and deterministic across different devices. The original research was endorsed by the official team of Amazon MXNet and the details can be read [here](https://medium.com/apache-mxnet/quantizing-neural-network-models-in-mxnet-for-strict-consistency-on-blockchain-b5c950674866).

In this tutorial, we will write a simple handwritten digit recognition model, convert it via MRT, and upload it to the Cortex blockchain so that it can be executed on-chain.

We will work through be 4 main stages

(1) Install CVM-Runtime (including MRT) and other dependencies.

(2) Train a model using the MXNet framework.

(3) Quantize the model trained in stage (2) using MRT.

(4) Upload the model

**Prerequisites:**

- A machine with the Linux Operating System <sup>[1]</sup>. The official implementation is currently only in Linux, but if you're passionate about implementing CVM-Runtime in a different operating system, pull requests are more than welcomed!

- A machine with GPU and CUDA installed properly for your Linux version.

- Willingness to debug for yourself. While we seek to be as comprehensive as possible in this tutorial, we cannot cover all the idiosyncrasies of different machines and environments.

If you encounter any problems during the course, feel free to reach out to our core dev team via our [Telegram](https://t.me/CortexOfficialEN) or [Twitter](https://twitter.com/CTXCBlockchain/). We seek to constantly improve our documentation.

Let's get started!

# Tutorial

## Stage I: Install CVM-runtime and Other Dependencies

### 1. Git clone the CVM-runtime repository

```bash
git clone -b wlt https://github.com/CortexFoundation/cvm-runtime.git

cd cvm-runtime
```

### 2. Configure for compilation

In `config.cmake`, set the `ENABLE_CUDA` variable to `ON` on line 6.

![config](imgs/config.png)

### 3. Compile

You may need to install `make` if you have not already.

```bash
make -j8 lib
```

If you encounter this error below

![cmake](imgs/cmake.png)

you should install `cmake`. Try the following commands:

```bash
sudo apt-get update
sudo apt-get install cmake
```

Now type `g++` to see if you have `g++` installed. If not, `sudo apt-get install g++`

You might need to switch to a machine with GPU and CUDA installed if you don't have one.

# Train Your Model

## Install MXNET

Now go to [MXNET's website](https://mxnet.apache.org/get_started/?platform=linux&language=python&processor=gpu&environ=pip&) to
install the GPU version of MXNET suited for your CUDA. The install command should look something like

`$pip3 install mxnet-cu102`

Use `$nvcc --version` to find out your CUDA version; make sure that it matches the number after "cu", in this case it is version 10.

Now run

```
pip3 install gluoncv

make dep
```

### Mnist Training

Execute the following command:

```bash
python3 tests/mrt/train_mnist.py
```

Trained models are stored in `~/mrt_model`.

# Quantize Your Model

To prepare the model for the Cortex blockchain, we need to quantize it. Cortex's original research has led to a tool called MRT that readily helps us quantize ML models for deterministic inference on the blockchain. If you're interested in how it works under the hood, check out this article (link) released on the blog of Amazon's MXNet team, who has officially endorsed the MRT.

Execute the following command:

```bash
python3 python/mrt/main2.py python/mrt/model_zoo/mnist.ini
```

All the pre-quantized model configuration file is stored in `python/mrt/model_zoo`, and the file `config.example.ini` expositions all the key meanings and value.

# Upload quantized model

Now the model is fully quantized, we're ready to upload it! Let's go the [Cerebro Explorer](https://cerebro.cortexlabs.ai/) .

In the menu bar at the top, find "upload" under "AI Contract"

![cerebroMenu](imgs/cerebroMenu.png)

`mnist_.json` and `mnist_.params` are your models, stored in `~/mrt_model`.

# Inference the model

We will try to call this model that you just trained from a smart contract to recognize handwritten digits!
<br />
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
Questions:

1. relationship between cvm-runtime, cvm and mrt?
2. can we train the model before compiling cvm-rumtime?

# Footnotes

[1] If you don't have access to a Linux system locally, you can open a Linux EC2 Instance on AWS and connect it to your editor via ssh. There will be a more detailed bonus tutorial on how to set this up - for now, here are the general steps to setting up a Linux system on AWS. Play around with it and use google if you get stuck. You're always welcomed to ask questions in our Telegram group.

> (1) Go to AWS console, set up an EC20 Ubuntu (Ubuntu is one of the most user-friendly Linux systems) Instance.

> (2) Start a folder named `.ssh`, put in your key pair and start a text file named `config`

> (3) Open the command palette in Visual Studio code (`command + P` in Mac), type in
> `> Remote-SSH: Connect to Host`
> Then choose `Add New SSH Host`.

> Type in your address `ubuntu@your-aws-instance-public-ip`

> Substitute in your own public ip here. Note that each time you restart your instance, the public ip changes, so you need to reconfigure upon each restart.

> Type in the absolute path to your configuration file. Your path should end with something like `/.ssh/config`
