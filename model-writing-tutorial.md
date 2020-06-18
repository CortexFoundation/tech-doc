# AI Models for Cortex Tutorial

In this tutorial, we will write a simple handwritten digit recognition model and convert it via MRT for uploading to the Cortex blockchain.

There will be 5 main steps

(1) We will first train a model with the MXNET framework.

(2) Install MRT dependencies

(3) Write a .ini configuration file to ready MRT for converting our model

(4) Run MRT to convert the model

(5) Upload the model

### Prerequisites:

- A Linux System (if you don't have access to a Linux system locally, you can open a Linux EC2 Instance on AWS and connect it to your editor via ssh. There will be a more detailed bonus tutorial on how to set this up - for now, here are the general steps to setting up a Linux system on AWS. Play around with it and use google if you get stuck. You're always welcomed to ask questions in our Telegram group.

(1) Go to AWS console, set up an EC20 Ubuntu (Ubuntu is one of the most user-friendly Linux systems) Instance.

(2) Start a folder named `.ssh`, put in your key pair and start a text file named `config`

(3) Open the command palette in Visual Studio code (`command + P` in Mac), type in
`> Remote-SSH: Connect to Host`
Then choose `Add New SSH Host`.

Type in your address `ubuntu@your-aws-instance-public-ip`

Substitute in your own public ip here. Note that each time you restart your instance, the public ip changes, so you need to reconfigure upon each restart.

Type in the absolute path to your configuration file. Your path should end with something like `/.ssh/config`

- A machine with GPU and CUDA installed properly for your Linux version.

# Mnist Training & Quantization

## 1. Configure for compilation

Set the `ENABLE_CUDA` variable `ON` in `config.cmake` line 6.

![config](imgs/config.png)

## 2. Compile MRT

You may need to install `make` if you have not already.

```bash
export PYTHONPATH=/home/ubuntu/cvm-runtime/python:${PYTHONPATH}

export LD_LIBRARY_PATH=/home/ubuntu/cvm-runtime/build:${LD_LIBRARY_PATH}

make -j8 lib
```

If you encounter this error while running `make -j8 lib`

![cmake](imgs/cmake.png)

Run `sudo apt-get update`

Then `sudo apt-get install cmake` to install `cmake`.

Now type `g++` to see if you have `g++` installed. If not, `sudo apt-get install g++`

You might need to switch to a machine with GPU and CUDA installed if you don't have one.

## Install MRT

Now go to MXNET's website (https://mxnet.apache.org/get_started/?platform=linux&language=python&processor=gpu&environ=pip&) to
install the GPU version of MXNET suited for your CUDA. The install command should look something like `$pip3 install mxnet-cu102`

Use `$nvcc --version` to find out your CUDA version.

```
pip3 install gluoncv

make python(dep? make python will cause "no module named mrt")
```

### Mnist Training

Execute the following command:

```bash
python3 tests/mrt/train_mnist.py
```

Training model is stored in `~/mrt_model`.

### Mnist Quantization

To prepare the model for the Cortex blockchain, we need to quantize it. Cortex's original research has led to a tool called MRT that readily helps us quantize ML models for deterministic inference on the blockchain. If you're interested in how it works under the hood, check out this article (link) released on the blog of Amazon's MXNet team, who has officially endorsed the MRT.

Execute the following command:

```bash
python3 python/mrt/main2.py python/mrt/model_zoo/mnist.ini
```

All the pre-quantized model configuration file is stored in `python/mrt/model_zoo`, and the file `config.example.ini` expositions all the key meanings and value.

## Upload quantized model

Now the model is fully quantized, we're ready to upload it!

## Inference the model
