## Model and Data Uploading

There are 3 phases to upload ml models/data on the storage layer of the Cortex blockchain (see [storage layer](storage-layer.md) for more detail):

### Upload Phase

mI models and input data are treated as a special type of smart contract on the Cortex blockchain. Creators need to send a special transaction with a function call to advanced the upload progress. Each transaction will increase the file upload progress by 512K bytes, consuming the corresponding storage quota.

### Preparation Phase

After the completion of the upload phase, the file preparation phase is entered. This phase lasts for 100 blocks (about 25 minutes). 

### Mature Phase

After 100 blocks, the prepared files enter the mature phase. The model/data is saved on the storage layer of the Cortex blockchain.

## Broadcasting

After mature phase, the owner is responsible for broadcasting the file to the network to reach the entire distributed file system; otherwise, the network consensus will reject relevant contract calls.  

