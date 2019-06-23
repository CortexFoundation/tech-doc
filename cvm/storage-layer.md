# Storage Layer

Cortex uses a distributed file system based on DHT (Distributed Hash Table) as a storage layer solution to reduce network load and network transmission cost. Storage quota is treated as a resource on the Cortex chain. Each mined block provides a 64K byte storage quota. Users freely bid on the use of storage quota with transaction fees.

##AI & Data Upload

AI models and inference data are treated as a special type of smart contract on the Cortex chain. Creators need to send an empty transaction to the contract to advance the upload progress. Each transaction will increase the file upload progress by 512K bytes, consuming the corresponding storage quota. 

After the file upload phase is completed, the file preparation phase is entered. This phase lasts for 100 blocks (about 25 minutes) and at the end of it, the prepared files enter the mature phase and can be used by AI inference contracts.

The owners of AI models are rewarded when they are called in smart contracts. The uploader is responsible for broadcasting the file to the network in order to reach the entire distributed file system; otherwise, relevant contract calls will be rejected by the network consensus.

