# Target:
1. Replace The complex Mining Algorithm on CUDA with 
Another Classic Mining Algorithm on CPU.
2. The Mining Process Should Be Configurable, with
flags to control the behaviour of mining of generating blocks.
3. The Module is better compatible with latest master branch,
and should be plugable.

# Mechanism:

## Cuckoo:
> POW, Proof of AI work for [CortexTheseus](https://github.com/CortexFoundation/CortexTheseus).
> Cuckoo controls flow of generating solutions, verifying solutions, veirifying blocks, accumalate reward, and finalizing blocks. 
> Based on implementation of [cvm-runtime](https://github.com/CortexFoundation/cvm-runtime), cuckoo calls plugins of CUDA env to generate and verify solutions.

## SHA3 Origin:
> Zero-ahead nonce counting hashing is a classic mechanism for Crypto-Currency Mining, also a realization of Pow (power of work), which supports BlockChain Technique developing.
Partly refer to [bitcoin-algorithm-explained](https://www.mycryptopedia.com/bitcoin-algorithm-explained/), SHA3 requires header hash, nonce, difficulty and timestamp to do next hashing.


- Version – The version number of the Bitcoin software.
- Previous block hash – A reference to the hash of the previous block that was included on the blockchain.
- Merkle Root – A representative hash of all transactions that are included in the candidate block.
- Timestamp – A piece of information that references the time that the block was created.
- Target – The target hash threshold, this block’s header hash must be less than or equal to the target hash that has been set by the network.
- Nonce – The variable that is used in the proof of work mining process.

## Difficulty Adjustment Algorithm
> https://github.com/cortex/EIPs/issues/100.
 algorithm:
 diff = (parent_diff +
         (parent_diff / 2048 * max((2 if len(parent.uncles) else 1) - ((timestamp - parent.timestamp) // 9), -99))
   ) + 2^(periodCount - 2) 
> the estimated interval is [9,18], the average is 13.5
> using block generating timestamp as factors of difficulty adjustment, which can control
the block generating interval. The Difficulty is higher with shorter interval.

# Dev Doc

## SHA3 Solution(CPU)
> Simple sha3 solution, Relace seal_miner "Gen Solution"
> header_hash len is 32*u8, nonce len is 2*u32,
> including sha3 solution Verify, Replace CuckooVerifyHeader()

Function Definition1:
`func (cuckoo *Cuckoo) GenSha3Solution(hash []byte, nonce uint64) (r uint32, sols [][]uint32)`

Function Definition2:
`func (cuckoo *Cuckoo) CuckooVerifyHeader_SHA3(hash []byte, nonce uint64, sol *types.BlockSolution, targetDiff *big.Int) bool`


## Difficulty Adjustment
 calcDifficultySHA3 the difficulty adjustment algorithm. It returns
 the difficulty that a new block should have when created at time given the
 parent block's time and difficulty. The calculation uses the Byzantium rules,
 tuning x growing faster when difficulty growing positively
 The estimated interval is (interval + 2*interval)/2

1. replace cuckoo CalcDifficulty with calcDifficultySHA3.

2. implement calcDifficultySHA3 as: 
Function Deifinition:
`func calcDifficultySHA3(time uint64, parent *types.Header, interval *big.Int) *big.Int`

3. test whether difficulty successfully controls the blocking generating interval

## Command Flags 
Using Mining Or Cuckoo Flags to control Block generating interval and Mining Behaviour (with or without Txs)

### Block Interval
`cuckoo.blockinterval`: 
> --cuckoo.blockinterval 10s
a flag embedded into cuckoo config, and infect the calcDifficultySHA3 to adjust
difficulty as the estimated average interval, which should be `1.5 * cuckoo.blockinterval`

### CarryTx
`miner.carrytx`:
> --miner.carrytx
a flag embedded into miner config, and infect the worker to only commit work with newTxs

## Testing

### go test
> sh3sol_test.go

### with some cortex node scripts:
#### create new account
```bash
 ./build/bin/cortex --dolores --datadir .cortex2/dolores account new
```
#### list accounts
```bash
 ./build/bin/cortex --dolores --datadir .cortex2/dolores account list
```
#### start mining with specific account, with interval 7.5s, should carrytx
```bash
./build/bin/cortex --dolores --storage.mode lazy --datadir .cortex2/dolores --rpc --rpcapi ctxc,web3,admin,txpool --rpcaddr 0.0.0.0 --rpcport 38545 --mine --miner.coinbase 7b775e3bc393db28d515ff6ad7bdf3713018935f --cuckoo.blockinterval 5s --allow-insecure-unlock --unlock 7b775e3bc393db28d515ff6ad7bdf3713018935f --miner.carrytx
```
#### send tx with js console attach
```bash
./build/bin/cortex attach --dolores --datadir .cortex2/
```
```js
ctxc.sendTransaction({
 "from": "0x7b775e3bc393db28d515ff6ad7bdf3713018935f",
 "to": "0xf51d56e3e9647a23e52c8da2ae26900835ee1015",
 "value": 100e2,
 "gas": 21000,
 "gasPrice": 3,
 "nonce": 0})
```


### Web3 Txs Testing
```bash
yarn add web3
node index.ts
```

```typescript
// index.ts In Node.js use: const Web3 = require('web3');
const Web3 = require('web3');

const web3_ = new Web3(new Web3.providers.HttpProvider("http://localhost:38545"));

var version = web3_.version
console.log('@@@web3', web3_)
console.log('web3 version', version);

// cortex is compatible with eth
// get node info
web3_.eth.getNodeInfo().then(
    console.log
);

// cortex is compatible with eth
// get chain id, normally is 43
web3_.eth.getChainId().then(
    console.log
);

// get accounts address
web3_.eth.getAccounts().then(
    console.log
)

// get block Number
web3_.eth.getBlockNumber().then(
    console.log
)

// send Transaciton
web3_.eth.sendTransaction(
    {
     "from": "0x7b775e3bc393db28d515ff6ad7bdf3713018935f",
     "to": "0xf51d56e3e9647a23e52c8da2ae26900835ee1015",
     "value": 100e2,
     "gas": 21000,
     "gasPrice": 3,
     "nonce": 1
    }
).then((rlt)=>{
        console.log(rlt);
        // get Transaciton after success
        web3_.eth.getTransaction(rlt
        ).then(
            console.log
        );

        web3_.eth.getTransactionCount(
        ).then(
            console.log
        );
    }
).catch(
    console.warn
);
```

