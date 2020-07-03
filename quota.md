# Quota

Quota is a kind of resource like gas. While gas is used to limit the computation consume for transactions in a block, quota is used to limit the space consume. And Quota increase by block mining and cost by uploading files to blockchain.

You can find `Quota` and `Quota_used` in the block header. `Quota` describes the total quota in the blockchain. `Quota_used` describes the total used quota in the blockchain.



## Following rules:

1. **The Quota of each block increases by NewQuota.**

   CurrentBlock.Quota == ParentBlock.Quota + NewQuota

   NewQuota is set according to the network ID (BLOCK_QUOTA/Bernard_BLOCK_QUOTA/Dolores_BLOCK_QUOTA)

2. **The QuotaUsed of each block is less or equal than the Quota in the block**

   Block.QuotaUsed <= Block.Quota

3. **The increased QuotaUsed is computed by function ApplyTransaction**

