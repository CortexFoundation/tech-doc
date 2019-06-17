User Guide
==========

About
-----

The Cortex repository is a fork of [Go Ethereum](https://github.com/ethereum/go-ethereum>) which contains protocol changes to support ml model inference. This implements the Cortex network, which maintains a separate ledger from the Ethereum network, for several reasons, the most immediate of which are AI inference support and the consensus protocol is different.

Requirements
---------------

To ensure that your Cortex client runs gracfully, please check your system meets the following requirements:

- System: Linux Ubuntu 18.04+
- RAM: 8GB
- GPU: Nvidia GPU with 10.7GB VRAM (1080ti, 2080ti, titan V, etc.)
- Space: 25GB  (*the size of the block chain increases over time*)
- CUDA version: 9.2+
- CUDA driver: 396+
- Compiler: Go 1.10+, GCC 5.4+


note: Currently we only officially support Linux (Ubuntu), but we are actively investigating development for other operating systems and platforms(e.g. macOS, Ubuntu, Windows, Fedora). 

Building Cortex clients requires a Go (version 1.10 or later), a C compiler, and a CUDA (version 9.2 or later). We will guide you through the Installation section.

