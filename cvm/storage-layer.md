# Storage Layer

Cortex uses a distributed file system based on DHT (Distributed Hash Table) as the storage layer for model files and data files.

##AI & Data Upload

The AI model or data is treated as a special smart contract on the Cortex blockchain. The creator needs to send a transaction to the contract to advance the progress of the upload. The model/data is available after the upload progress is completed. Calling the contract will reward the contract owner. The uploader needs to broadcast the file on the network for distribution to the entire storage layer.

