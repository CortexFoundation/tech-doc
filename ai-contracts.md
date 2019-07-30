# AI Contracts

The idea of smart contracts is not originated by Cortex development team but by cryptographer Nick Szabo in 1994. Ethereum later implemented a nearly Turing-complete language on its blockchain and popularized the usage of smart contracts. Cortex adds machine learning operators to allow various interactions within the smart contracts that is executable on CVM without losing performance and precision. 

## Solidity

Cortex officially supports Solidity programming language to develop AI contracts and complie to CVM bytcode. 

  - [Solidity Docs](https://solidity.readthedocs.org/en/latest/) - Solidity is the Smart Contract language developed by Ethereum, which compiles to CVM (Cortex Virtual Machine) codes.
  - [Cortex-Remix](https://cerebro.cortexlabs.ai/remix) -  a browser-based compiler and IDE using programming language Solidity.

## Calling On-Chain Model

Function infer(model, input, infer_output)

- Parameters

  1. model - address of the model
  2. input - address of the data
  3. infer_output - array of uint256, receive infer output

- Returns
  None

Function inferArray(model, input_data, infer_output)

- Parameters

  1. model - address of the model
  2. input_data - storage array of uint256
  3. infer_output - array of uint256, receive infer output

- Returns
  None


## Fractional Units for Cortex

Refer to [CTXC](ctxc.md)

## A Simple Smart Contract Example

This contract illustrate a simple implementation of calling and inferring a model on the Cortex blockchain. 

```js
pragma solidity ^0.4.18;
contract AIContract {
  uint256[] input_data;
  uint256[] infer_output = new uint256[](uint256((1 * 10 + 31) >> 5));
  
  constructor() public {
      input_data = new uint256[]((1 * 3 * 32 * 32 + 31) >> 5);
  }
  
  function Infer(address model, address input) public returns (uint256) {
    // feed data in input to model and store the output in infer_output
    infer(model, input, infer_output);
    return infer_output[0];
  }
  
  function InferArray(address model) public returns (uint256) {
    // feed data in input_data to model and store the output in infer_output
    inferArray(model, input_data, infer_output);
    return infer_output[0];
  }
}
```

The first line tells the source code is written for Solidty version 0.4.18 or newer. The `pragma` command are used by the compiler to check compatability, while the caret symbol (^) means any *minor* version above 0.4.18, e.g., 0.4.19, but not 0.5.0.

In Solidity, the data types of both the function parameters and output need to be specified. In the first case, the function `infer` takes `address model`, `address input`, `infer_output`, and returns the infer result stored in infer_output[0]. While `inferArray` take uint256 storage array as input data for the model. It is to ensure enough length to read data stored on the Cortex blockchain. The size of uint256 is equal to 32 int8. 

`infer` and `inferArray` calls the on-chain model with paramemters model, input and output. When executed, the CVM will call the data and model on the Cortex blockchain, make inference, and set the result in the output, namely the first element of `output` array.

