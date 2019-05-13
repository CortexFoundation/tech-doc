### Executables

------------------

#### Command

use the parameter directly.

|parameter| description |
|----|----|
|help|Print Cortex binary help statements.|
|init|Init Cortex node from a genesis configuration file|

#### Options

use "--" prefix with the parameter.

|       parameter     | description |
|--------------------- | -------------|
| cerebro | Setting --cerebro flag to enable connection to Cerebro testnet. |
|| Notice: When this flag is enabled, datadir will be located at $HOME/.cortex/cerebro/ by default, which different from $HOME/.cortex/ in the previous version, be noticed to set datadir if restart from the previous version with default directory.|
| storage | [Necessary] |
||Setting this flag enables synchronization of Cortex storage layer, a standard fullnode with inference engine must set this.|
| storage.dir|Set the directory of Cortex storage layer. By default, $HOME/.cortex/storage the directory is the storage dir.|
|storage.tracker|To alleviate the latency of finding the torrent file in P2P network, you can set torrent tracker manually. By default, a pre-defined tracker is available.|
|datadir|Set Cortex binary data directory. By default, $HOME/.cortex/ is data directory.|
|networkid|Set Cortex blockchain network id, Cerebro testnet is 42 by default.|
|port|Set Cortex binary listening port.|
|bootnodes| [Options]|
||Set bootnodes of Cortex blockchain.|
|verbosity|Set logging level to print, by default is 3, range in [1, 5], which represent Error, Warn, Info, Debug, Trace.|