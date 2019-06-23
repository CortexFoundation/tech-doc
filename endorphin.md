# Endorphin

Endorphin is the execution fee that senders of transactions need to pay for every operation made on Cortex. Every transaction is required to include an endorphin limit and a fee that it is willing to pay per Endorphin. Miners have the choice of including the transaction and collecting the fee or not. 

In contrast to the traditional blockchain, in which the reward for each packaging block is paid directly to the miners, a portion of the reward goes to the model provider to motivate developers to submit richer and better models. Endorphins invoking the contract, are not only allocated to miners who help package the block on the full node, but also used to pay the model providers. 

##Mechanism

The proportion of fees charged will be determined by the market gaming price, similar to the Ethereum Gas mechanism. AI smart contracts having a higher max Endorphins price have a higher priority to be executed. A single execution of a contract will cause a fee of Endorphins Limit * max Endorphins Price. Endorphin is a measurement of the number of computing resources spent on a hardware level within a virtual machine when bringing a data model into a contract during inference. Generally speaking, the cost of the Endorphin is proportional to the size of the model. Cortex also sets an upper bound of 8GB on the parameter size of the model, corresponding to up to about 2 billion Float32 parameters.

## Estimating transaction costs

The total ether cost of a transaction is based on 2 factors:

`endorphinUsed` is the total gas that is consumed by the transaction

`endorphinPrice` price (in ether) of one unit of gas specified in the transaction

**Total cost = endorphinUsed \* endorphinPrice**

