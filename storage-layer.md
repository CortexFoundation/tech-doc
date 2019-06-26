# Storage Layer

Cortex uses a distributed file system based on DHT (Distributed Hash Table) as a storage layer solution to reduce network load and network transmission cost. Storage quota is treated as a resource on the Cortex chain. Each mined block provides a 64K byte storage quota. Users freely bid on the use of storage quota with transaction fees.

A DHT is a form of a distributed database that can store and retrieve information associated with a key in a network of peer nodes that can join and leave the network at any time. The nodes coordinate among themselves to balance and store data in the network without any central coordinating party. DHT is both fault tolerant and resilient when key/value pairs are replicated.

The idea of DHT is that each node either stores this key itself, or it is acquainted. It knows some other computer which is closer to this key in terms of the distance on the circle. That way, if a request comes to any node in the network for a key, it either can find this key inside it's own storage or redirect the request to another node which is closer to this key. Thenthat node will either store the key or direct the code to the next node, which is even closer to that key. In finite number of iterations the request will come to the node that will actually stores the key. 









