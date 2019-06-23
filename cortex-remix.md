#Cortex Remix

Cortex Remix is a browser-based compiler and IDE using programming language Solidity. It is based on [Remix IDE](https://github.com/ethereum/remix-ide). It supports the compilation and deployment of AI smart contracts and debugging transactions.

Developing on Cortex closely resembles writing Solidity contracts on Ethtereum, but with added instruction sets. The instruction sets allows contracts to interact with AI models and data on Cortex and make inference.

Cortex Remix mainly consists of two functional modules: compilation and deployment. 

## Compilation

The compilation module supports compilation and optimization of AI smart contracts. Complied abi, bytecode, and additional information are also displayed in this module.

## Deployment

The deployment module can help deploy AI smart contracts to the Cortex network with the support of Cortex Wallet, allowing for on-chain inference.

## Calling On-Chain Model

Function inferArray(model, input, output)

- Parameters
	1. model - address of the model
	2. Input - storage array of uint256
	3. output - array of uint256, receive infer output
- Returns
	None
##Example
```javascript
pragma solidity ^0.4.18;
contract AIContract {
  uint256[] input_data;
    function post_process_yolo() public returns (uint256) {
    address model = 0x0000000000000000000000000000000000001003;
    input_data = new uint256[]((1 * 3 * 416 * 416 + 31) >> 5);
    uint256[] memory output = new uint256[](uint256((1 * 28 + 31) >> 5));
    inferArray(model, input_data, output);
    return output[0];
  }
}
```



## Fractional Units for Cortex

Refer to [CTXC](ctxc.md)