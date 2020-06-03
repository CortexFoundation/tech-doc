# AI Models for Cortex Tutorial

In this tutorial, we will write a simple handwritten digit recognition model and convert it via MRT for uploading to the Cortex blockchain.

There will be 5 main steps

(1) We will first train a model with the MXNET framework.

(2) Install MRT dependencies

(3) Write a .ini configuration file to ready MRT for converting our model

(4) Run MRT to convert the model

(5) Upload the model

### Prerequisites:

A Linux System (if you don't have access to a Linux system locally, you can open a Linux EC2 Instance on AWS and connect it to your editor via ssh. There will be a more detailed bonus tutorial on how to set this up - for now, here are the general steps to setting up a Linux system on AWS. Play around with it and use google if you get stuck. You're always welcomed to ask questions in our Telegram group.

(1) Go to AWS console, set up an EC20 Ubuntu (Ubuntu is one of the most user-friendly Linux systems) Instance.

(2) Start a folder named `.ssh`, put in your key pair and start a text file named `config`

(3) Open the command palette in Visual Studio code (`command + P` in Mac), type in
`> Remote-SSH: Connect to Host`
Then choose `Add New SSH Host`.

Type in your address `ubuntu@your-aws-instance-public-ip`

Substitute in your own public ip here.

Type in the absolute path to your configuration file. Your path should end with something like `/.ssh/config`

# Mnist Training & Quantization

### CVM-Runtime Project Compilation

1. Config the configuration for compilation

   a) Check file `config.cmake` exists in your project root, or execute the following command:

   ```bash
   cp cmake/config.cmake .
   ```

   b) set the `ENABLE_CUDA` variable `ON` in `config.cmake` line 6.

   ![config](imgs/config.png)

2. Compile with following command

You may need to install `make` if you have not already.

```bash
make -j8 lib
make python
```

If you encounter this error

![cmake](imgs/cmake.png)

You can run `sudo apt-get install cmake` to install `cmake`. You may need to run `sudo update apt-get` first to make sure `cmake` can install successfully.

### Mnist Training

Execute the following command:

```bash
python3 tests/python/train_mnist.py
```

Training model is stored in `~/mrt_model`.

### Mnist Quantization

Execute the following command:

```bash
python3 python/mrt/main2.py python/mrt/model_zoo/mnist.ini
```

All the pre-quantized model configuration file is stored in `python/mrt/model_zoo`, and the file `config.example.ini` expositions all the key meanings and value.
