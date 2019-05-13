User Guide
==========

About
-----

The Cortex repository is a fork of [Go Ethereum](https://github.com/ethereum/go-ethereum>) which contains protocol changes to support [Snynapse Engine](). This implements the Cortex network, which maintains a separate ledger from the Ethereum network, for several reasons, the most immediate of which are AI inference support and the consensus protocol is different.

Requirements
---------------

To ensure that your Cortex client runs gracfully, please check your system meets the following requirements:

	| :fa:`linux` ``64-bit`` Linux OS
	| :fa:`microchip` ``64-bit`` Processor
	| :fa:`database` ``8GB`` of free RAM
	| :fa:`hdd-o` ``25GB`` of free Disk (*the size of the block chain increases over time*)


.. note:: Currently we only officially support Linux (Debian), but we are actively investigating development for other operating systems and platforms(e.g. macOS, Ubuntu, Windows, Fedora). 

Building Cortex clients requires a Go (version 1.7 or later), a C compiler, and a CUDA (version 9.2 or later). We will guide you through the Installation section.