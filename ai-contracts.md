# AI Contracts

The idea of smart contracts is not originated by Cortex development team but by cryptographer Nick Szabo in 1994. Ethereum later implemented a nearly Turing-complete language on its blockchain and popularized the usage of smart contracts. Cortex adds machine learning operators to allow various interactions within the smart contracts that is executable on CVM without losing performance and precision. 

## Solidity

Cortex officially supports Solidity programming language to develop AI contracts and complie to CVM bytcode. 

  - [Solidity Docs](https://solidity.readthedocs.org/en/latest/) - Solidity is the Smart Contract language developed by Ethereum, which compiles to CVM (Cortex Virtual Machine) opcodes.
  - [Cortex-Remix](https://cerebro.cortexlabs.ai/remix) -  a browser-based compiler and IDE using programming language Solidity.

## Calling On-Chain Model

Function inferArray(model, input, output)

- Parameters

  1. model - address of the model
  2. Input - storage array of uint256
  3. output - array of uint256, receive infer output

- Returns
  None

  

## Fractional Units for Cortex

Refer to [CTXC](ctxc.md)

## A Simple Smart Contract Example

This contract illustrate a simple implementation of calling and inferring a model on the Cortex blockchain. 

```javascript
pragma solidity ^0.4.18;
contract AIContract {
    function post_process_yolo(address model, address input) public returns (uint256) {
    uint256[] memory output = new uint256[](uint256((1 * 28 + 31) >> 5));
    inferArray(model, input, output);
    return output[0];
  }
}
```

The first line tells the source code is written for Solidty version 0.4.18 or newer. The `pragma` command are used by the compiler to check compatability, while the caret symbol (^) means any *minor* version above 0.4.18, e.g., 0.4.19, but not 0.5.0.

In Solidity, the data types of both the function parameters and output need to be specified. In this case, the function `post_process_yolo` takes `address model`, `address input`, and returns an integer. The line `uint256[] input_data;` declares a state variable `input_data` of type `uint256`. It can be accessed throughout the contract `AIContract`. The line `address model` declares a static variable called `model` of type `address` with variable of `0x0000000000000000000000000000000000001003`. 

The line below specifies integer array of 256 bits `data_input` with length `(1 * 3 * 416 * 416 + 31)` moved by 5 digits for. It is to ensure enough length to read data stored on the Cortex blockchain. Similarly, ` uint256[] memory output` also specifies integer array of 256 bits with length `(1 * 28 + 31)` moved by 5 digits that is stored in the memory.

`inferArray` calls the on-chain model with paramemters model, input and output. When executed, the CVM will call the data and model on the Cortex blockchain, make inference, and return array of output. In this case, the contract `AIContract` returns the first value of `output` array.

