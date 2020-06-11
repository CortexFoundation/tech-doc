# Apple Demo in Jetson Nano Car

*One newer to CortexLabs may be confused about the project application in real environment , and this is a preview demo guiding you having a quick glance at the special point as our [official website](www.cortexlabs.ai) introduced: AI on Blockchain.*

## A Compressed GIF Version

![demo](cvm/demo/demo.gif)



The gif above is a demo running in Jetson Nano Car produced by NVIDIA. And the inference framework is [CVM Runtime](https://github.com/CortexFoundation/cvm-runtime) developed by CortexLabs.

The full video has been uploaded in YouTube via CortexLabs official account. Click [Here](https://youtu.be/88c-446s9JE) to watch.

## Video Guide

The procedure showed in the video contains:

1. Execute the deterministic AI framework: **cvm-runtime** in Jetson Nano Car, it's off-chain and deterministic. That is  edge devices can execute real-time AI decision with our AI framework.
2. And then upload the image data to blockchain and call on-chain AI contract call since the result must be consistent with native execute progress. The benefits have
   - Real-time AI decision can be strictly reproduction in future requests for analisis.
   - A decentralization and crediable blockchain record will largely add the confidence level of AI inference. Any doubt about the last AI decision will only focus on the AI inference procedure and model supplier, instead of the credibility of inference itself.

## Implementation Details

*The demo is tested in Cortex Dolores TestNet.*

### Real-time AI Inference In Jetson Nano Car

Please clone our cvm-runtime project in GitHub and compile. More details refer to the project link:

[cvm-runtime in GitHub](https://github.com/CortexFoundation/cvm-runtime)

#### Developer API

We have supplied some interface in cvm-runtime, both python and c++ are available to invoke. And we will introduce the python API for the video demo. The python API is located in `{ProjectRoot}/python`.

##### cvm.runtime

The package expose three main inference interface: 

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

Image upload has many procedures, and all tha lower and instrinstic representation format is transaction with different parameters. The default raw transaction JSON RPC format is:

#### ctxc_sendRawTransaction

*Creates new message call transaction or a contract creation for signed transactions.*

##### Parameter Format

`Object` - The transaction object

- `from`: `DATA`, 20 Bytes - The address the transaction is send from.
- `to`: `DATA`, 20 Bytes - (optional when creating new contract) The address the transaction is directed to.
- `gas`: `QUANTITY` - (optional) Integer of the gas provided for the transaction execution. It will return unused gas.
- `gasPrice`: `QUANTITY` - (optional, default: To-Be-Determined) Integer of the gasPrice used for each paid gas
- `value`: `QUANTITY` - (optional) Integer of the value sent with this transaction.
- `data`: `DATA` - The compiled code of a contract OR the hash of the invoked method signature and encoded parameters.
- `nonce`: `QUANTITY` - (optional) Integer of a nonce. This allows to overwrite your own pending transactions that use the same nonce.

1. `DATA`, The rlp encoded data of `object` above. 

##### Returns

`DATA`, 32 Bytes - the transaction hash, or the zero hash if the transaction is not yet available.

Use `ctxc_getTransactionReceipt` to get the contract address, after the transaction was mined, when you created a contract.

After acknowledging the native transaction rules, it's time to introduce the detail upload procedure.

#### 1. Create torrent for image data

You should save the image data into disk named `data`, create the torrent file and organize at the same level. Directory structure looks like this:

```
root_directory/
├── data
└── torrent
```

#### 2. Announce image uploading onto blockchain

First, we should tell the blockchain that I want to upload an image, and supply the image information to blockchain via transaction before download. The most important thing is the torrent info hash. After blockchain node received the uploaded transaction, it will start to find the torrent corresponding with the info hash from tracker and DHT network. 

**Be attention**, do not forget to seed the `root_directory` directory list above in you local environment. It's important for chain-node to find the right torrent, or chain-node cannot find it and refused to download the image you want to upload. You can use some tools as torrent seeding server such as *qbittorrent*, *libtorrent* etc to serve the `root_directory`.

Image is called as `Input` in blockchain, same as contract deployment that is send an transaction of key `to` is null. Howerver, `Input` deployment has an unique hex prefix indicator: `0002` before the contract data.

The `Input` meta is a structure of `Comments, InfoHash, RawSize, Shape `, and then one use the RLP method to encode the list above. 

So a normal announce image upload trasaction may looks like this:

- `method`: ctxc_sendRawTransaction
- `parameters`: 
  - `to`: none
  - `value`: 0
  - `data`: "0x0002" + RLPEncode([Comments, InfoHash, RawSize, Shape])

#### 3. Wait for seeding block

After announcing the uploaded image onto blockchain, we have set some seeding blocks left for chain-node to download the torrent from DHT network. The default seeding block number in MainNet and TestNet is 6, that is you have to wait six blocks(about 90 sec) mined before starting next step.

#### 4. Push upload progress

`Input` uploading consumes CTXC. You should transfer to the `Input` contract address with specific gas price: 277777. One the specfic transaction should push your upload progress on chain 512K size. The calculate formula of transaction number to be sent is $\lceil RawSize / 512K \rceil$.

#### 5. Wait for mature block

Once push progress done, the mature block is set to wait for chain-node to download. The MainNet mature block number is 100 and the TestNet is 1 for quick test. 

### AI Contract on Blockchain Inference

If you are newer to contract deployment, may you click the [link](ai-contracts.md) to review how to deploy contract.

Here is the contract core inference code in videos: 

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

The `bet` function will transfer you 0.02 CTXC if on-chain AI recognize an image as apples, and given that you have inferenced as apples in car, you'll receive the reward CTXC after calling contract.