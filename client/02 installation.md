Installation
------------

Before we begin installing Cortex, we need to get some dependencies for your system.

### Install NVIDIA Driver

Remove old driver

```bash
sudo apt-get remove --purge nvidia*
sudo apt-get install build-essential freeglut3-dev libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev
sudo nano /etc/modprobe.d/blacklist-nouveau.conf
```

Paste the following content into the text editor and save `blacklist-nouveau.conf`,

    blacklist nouveau
    options nouveau modeset=0

Download the driver - Option 1 - via Nvidia.com

```bash
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/410.93/NVIDIA-Linux-x86_64-410.93.run
chmod +x NVIDIA-Linux-x86_64-410.93.run
```

Download the driver - Option 2 - via PPA repository

```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update 
```


Press Ctrl + Alt + F1 to enter tty1 console,

```bash
sudo update-initramfs -u
sudo service lightdm stop
sudo ./NVIDIA-Linux-x86_64-410.93.run –no-opengl-files
(do not apply xorg config here!!!!)
```
Then

```bash
sudo service lightdm start
```

Press Ctrl + Alt + F7 to go back tty7 interface.

### Install CUDA (without NVIDIA Driver)

Option 1 - Install CUDA 10.0 (Without NVIDIA Driver)

```bash
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/ local_installers/cuda_10.0.130_410.48_linux
mv cuda_10.0.130_410.48_linux cuda_10.0.130_410.48_linux.run
sudo sh cuda_10.0.130_410.48_linux.run
sudo ldconfig /usr/local/cuda/lib64
(IMPORTANT: don't install driver here!!!)

// Add two lines to ~/.bashrc
echo 'export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```

Option 2 - Install CUDA 9.2 (Without NVIDIA Driver)

```bash
wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda_9.2.148_396.37_linux
mv cuda_9.2.148_396.37_linux cuda_9.2.148_396.37_linux.run
sudo sh cuda_9.2.148_396.37_linux.run
sudo ldconfig /usr/local/cuda/lib64
(IMPORTANT: don't install driver here!!!)

// Add two lines to ~/.bashrc
echo 'export PATH=/usr/local/cuda-0.2/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```


### Install Go 

Optnion 1 - Install via Google.com

```bash
wget https://dl.google.com/go/go1.11.5.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.11.5.linux-amd64.tar.gz
echo 'export PATH="$PATH:/usr/local/go/bin"' >> ~/.bashrc
source ~/.bashrc
```

Optnion 2 - Install via package manager

```bash
sudo apt install golang-1.11
```

Make sure you have installed correctly

```bash
go version
```

### Building the source

Clone the source (Need permission)

```bash
git clone git@github.com:CortexFoundation/CortexTheseus.git --branch dev-cerebro
(with git accessable key)
git clone http://github.com/CortexFoundation/CortexTheseus --branch dev-cerebro
(with git accessable account)

cd CortexTheseus
```

Once the dependencies are installed, run

```bash
cd infernet && make clean
cd .. && make clean
make -j cortex
```

Save the executable file

```bash
sudo mkdir -p /serving/cortex-core/bin
sudo chmod 777 /serving/cortex-core/bin
cp build/bin/cortex /serving/cortex-core/bin/cortex
```

The compiled binary files are located in the ./build/bin

```bash
./build/bin/cortex
```


Run the fullnode
----------------

Deploy with script

```bash
wget https://raw.githubusercontent.com/lizhencortex/cortex-deploy/master/deploy.sh
chmod +x deploy.sh
sudo ./deploy.sh
```

### Fullnode executables directly

```bash
./cortex --port 37566 --rpc --rpccorsdomain '*' --rpcport 30089 --rpcaddr 127.0.0.1 --rpcapi web3,eth,ctx,miner,net,txpool --verbosity 4 --storage --cerebro --gcmode archive --rpcaddr 127.0.0.1
```

### Fullnode executables via supervisor

#### Create bash script

```bash
rm /serving/cortex-core/bin/cortex.sh
sudo nano /serving/cortex-core/bin/cortex.sh
```

Create /serving/cortex-core/bin/cortex.sh

```bash
#!/bin/bash
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
/serving/cortex-core/bin/cortex --port 37566 --rpc --rpccorsdomain '*' --rpcport 30089 --rpcaddr 127.0.0.1 --rpcapi web3,eth,ctx,miner,net,txpool --verbosity 4 --storage --cerebro --gcmode archive --rpcaddr 127.0.0.1
```

Make the script executable

```bash
sudo chmod +x /serving/cortex-core/bin/cortex.sh
```

#### config

    [program:cortexnode]
    directory=/serving/cortex-core/bin/
    command=bash /serving/cortex-core/bin/cortex.sh
    autostart=true
    autorestart=true
    startsecs=5
    stderr_logfile=/tmp/cortex_fullnode_stderr.log
    stdout_logfile=/tmp/cortex_fullnode_stdout.log

#### check running status

```bash
sudo supervisorctl tail cortexnode stdout
```