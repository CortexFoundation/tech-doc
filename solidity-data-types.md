# Data Types

 Solidity is a statically typed language, which means that the type of each variable (state and local) needs to be specified. Solidity provides several elementary types which can be combined to form complex types.

Some of the basic data types offered in Solidity:

- Boolean (bool)

  Boolean value, true or false, with logical operators ! (not), && (and), || (or), == (equal), and != (not equal).

- Integer (int, uint)

  Signed (int) and unsigned (uint) integers, declared in increments of 8 bits from int8 to uint256. Without a size suffix, 256-bit quantities are used, to match the word size of the EVM.

- Fixed point (fixed, ufixed)

  Fixed-point numbers, declared with (`u`)`fixed*M*x*N*` where *M* is the size in bits (increments of 8 up to 256) and *N* is the number of decimals after the point (up to 18); e.g., ufixed32x2.

- Address

  A 20-byte Cortex address. The address object has many helpful member functions, the main ones being balance (returns the account balance) and `transfer` (transfers ether to the account).

- Byte array (fixed)

  Fixed-size arrays of bytes, declared with bytes1 up to bytes32.

- Byte array (dynamic)

  Variable-sized arrays of bytes, declared with bytes or string.

- Enum

  User-defined type for enumerating discrete values: enum NAME {LABEL1, LABEL 2, ...}.

- Arrays

  An array of any type, either fixed or dynamic: uint32[][5] is a fixed-size array of five dynamic arrays of unsigned integers.

- Struct

  User-defined data containers for grouping variables: `struct NAME {TYPE1 VARIABLE1; TYPE2 VARIABLE2; ...}`.

- Mapping

  Hash lookup tables for *key* => *value* pairs: mapping(KEY_TYPE => VALUE_TYPE) NAME.

In addition to these data types, Solidity also offers a variety of value literals that can be used to calculate different units:

- Time units

  The units seconds, minutes, hours, and days can be used as suffixes, converting to multiples of the base unit seconds.

- CTXC units

  The [units](ctxc.md) can be used as suffixes, converting to multiples of the base unit endorphin.

## Reference

- [Solidity Docs](https://solidity.readthedocs.io/en/latest/)
- [Mastering Ethereum](https://github.com/ethereumbook/ethereumbook/blob/develop/07smart-contracts-solidity.asciidoc)

