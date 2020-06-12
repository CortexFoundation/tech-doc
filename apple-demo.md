
# Apple Demo in Jetson Nano Car

*Newcomers may be confused about what real-life application can CortexLabs bring to the table. This repository provides a quick glance into what we are doing at Cortex, as indicated on our [official website](www.cortexlabs.ai) introduced: AI on Blockchain.*

## A Compressed GIF Version

![demo](cvm/demo/demo.gif)



The gif above is a demo Car running Jetson Nano by NVIDIA. The inference framework used is our very own [CVM Runtime](https://github.com/CortexFoundation/cvm-runtime) by CortexLabs.

The full video has been uploaded on YouTube via CortexLabs official account. Click [Here](https://youtu.be/88c-446s9JE) to watch.

## Video Guide

Below is the written steps shown in the video:

1. Execute deterministic AI framework, **cvm-runtime**, in an off-chain edge device that can make Ai decision in realtime. In this case, the Jetson Nano Car.
2. Upload the image data to Cortex blockchain and call the on-chain AI contract to ensure results are consistent across the Jetson Nano Car and on-chain AI contract.
The purpose is to:
   - Strictly reproduce real-time AI decisions in future requests for analysis or audit.
   - Add the confidence level of AI inference by using a decentralization and credible blockchain record will. Any doubt of the last AI decision will focus on the AI inference procedure and model creator, instead of the inference process.

## Implementation Details

*The demo is tested in Cortex Dolores TestNet.*

### Real-time AI Inference In Jetson Nano Car

Please clone our cvm-runtime project in GitHub and compile. Refer to the project link for more detail:

[cvm-runtime in GitHub](https://github.com/CortexFoundation/cvm-runtime)

#### Developer API

We have created a few interfaces in cvm-runtime, both python and c++ are available to invoke. We will use the python API that is shown in the video demo for demonstration. The python API is located in `{ProjectRoot}/python`.

##### cvm.runtime

The package has three main inference interface: 

| Name            | Parameters                     | Function                           |
| --------------- | ------------------------------ | ---------------------------------- |
| CVMAPILoadModel | (json_str, param_bytes[, ctx]) | Load model and return reference    |
| CVMAPIFreeModel | (model_reference)              | Free the loaded model by reference |
| CVMAPIInference | (model_reference, input_data)  | inference input via model          |

A simple code usage likes

```python
# symbol_path and params_path is model stored in disk path.
json_str, param_bytes = cvm.utils.load_model(symbol_path, params_path)
model_reference = cvm.runtime.CVMAPILoadModel(json_str, param_bytes, ctx=cvm.gpu())
# input_data is cvm.nd.NDArray format, generally create in cvm.nd.array(numpy_data)
result = cvm.runtime.CVMAPIInference(model_reference, input_data)
# do something process the inference result ...
# free the loaded model
cvm.runtime.CVMAPIFreeModel(model_reference)
```

### Upload Image to Blockchain

Image upload ton the blockchain requires many procedures. All the lower and intrinsic representation format is transacted with different parameters. The default raw transaction JSON RPC format is:

#### ctxc_sendRawTransaction

*Creates new message call transaction or a contract creation for signed transactions.*

##### Parameter Format

`Object` - The transaction object

- `from`: `DATA`, 20 Bytes - The address the transaction is sent from.
- `to`: `DATA`, 20 Bytes - (optional when creating new contract) The address the transaction is directed to.
- `gas`: `QUANTITY` - (optional) Integer of the gas provided for the transaction execution. It will return unused gas.
- `gasPrice`: `QUANTITY` - (optional, default: To-Be-Determined) Integer of the gasPrice used for each paid gas
- `value`: `QUANTITY` - (optional) Integer of the value sent with this transaction.
- `data`: `DATA` - The compiled code of a contract OR the hash of the invoked method signature and encoded parameters.
- `nonce`: `QUANTITY` - (optional) Integer of a nonce. This allows to overwrite your own pending transactions that use the same nonce.

1. `DATA`, The rlp encoded data of `object` above. 

##### Returns

`DATA`, 32 Bytes - the transaction hash, or the zero hash if the transaction is not yet available.

Use `ctxc_getTransactionReceipt` to get the contract address, after the transaction ismined, you can create a contract.

After acknowledging the native transaction rules, it's time to introduce the detailed upload procedure.

#### 1. Create torrent for image data

You should save the image data in folder called `data`, create the torrent file and organize at the same level. Directory structure should look like this:

```
root_directory/
├── data
└── torrent
```

#### 2. Announce image uploading onto blockchain

First, we should tell the blockchain that we want to upload an image, and supply the image information to the blockchain via transaction before download. The most important thing is the torrent info hash. After blockchain node received the uploaded transaction, it will start to find the torrent corresponding with the info hash from tracker and DHT network. 

**Be aware** Do not forget to seed the `root_directory` directory list above in your local environment. It's important for chain-node to find the right torrent, or else the chain-node will not be able to find it then refuse to download the image you want to upload. You can use a torrent seeding server such as *qbittorrent*, *libtorrent* etc to serve the `root_directory`.

Image is called as `Input` in blockchain, same as contract deployment that sends a transaction of key `to` is null. However, `Input` deployment has a unique hex prefix indicator: `0002` before the contract data.

The `Input` meta is a structure of `Comments, InfoHash, RawSize, Shape `, and then one uses the RLP method to encode the list above. 

So an announce image upload trasaction may looks like this:

- `method`: ctxc_sendRawTransaction
- `parameters`: 
  - `to`: none
  - `value`: 0
  - `data`: "0x0002" + RLPEncode([Comments, InfoHash, RawSize, Shape])

#### 3. Wait for seeding block

After announcing the uploaded image on the blockchain, we have set some seeding blocks left for chain-node to download the torrent from DHT network. The default seeding block number in MainNet and TestNet is 6, that is you have to wait for six blocks(about 90 sec) mined before starting the next step.

#### 4. Push upload progress

`Input` uploading consumes CTXC. You should transfer to the `Input` contract address with a specific gas price: 277777. One of the specific transaction should push your upload progress on-chain with 512K size. The calculated formula of the transaction number to be sent is $\lceil RawSize / 512K \rceil$.

#### 5. Wait for mature block

Once the push process is done, the mature block is set to wait for the chain-node to download. The MainNet mature block number is 100, but the TestNet is set to 1 to speed up the testing. 

### AI Contract on Blockchain Inference

If you are new to contract deployment, use this [link](ai-contracts.md) to learn how to deploy a contract.

Here is the core contract inference code in the video: 

```javascript
    function bet(address _input, address _receiveAddress) payable public {
        require(msg.value == betPerGame, "Invalid bet!"); // constant betting amount
        uint256[] memory output = new uint256[](uint256(1));
        infer(defaultModelAddress, _input, output);

        // Note:
        // this part is hardcoded for this specific model
        uint256 result = output[0] >> 16;
        uint256 mask = 0xff;

        // grabing the second 16 bits (index 1)
        int8 index1 = int8((result & (mask << (28 * 8))) >> (28 * 8));
        // grabing the first 16 bits (index 0)
        int8 temp = int8((result & (mask << (29 * 8))) >> (29 * 8));
        if(temp > index1) {
            emit BettingResult(false, 0, index1, temp); // or make reward negative
            return;
        }
        for(int8 i = 27; i >= 0; --i) {
            temp = int8((result & (mask << uint(i * 8))) >> uint(i * 8));
            if(temp > index1) {
                emit BettingResult(false, 0, index1, temp); // or make reward negative
                return;
            }
        }
        uint256 rewardVal = msg.value * 2; // 2x odds
        _receiveAddress.transfer(rewardVal);
        emit BettingResult(true, rewardVal, index1, temp);
    }
```

The `bet` function will transfer 0.02 CTXC to you if on-chain AI recognize an image as apples as well as the deterministic AI framework in the Jetson Nano Car, you'll receive the reward CTXC after calling the contract.
