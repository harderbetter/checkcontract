# An Automated Vulnerability Detection Framework for Smart Contracts

The smart contract for Majority and Union datasets can be founded at https://drive.google.com/drive/folders/1oLmKChhjkjmeLfXeuC0nB0XxcOqCzSsP?usp=sharing

The codesmell dataset: https://github.com/CodeSmell2019/CodeSmell 

This is the dataset of 'Defining Domain-Specific Code Smells in Smart Contracts'. CodeSmells.csv is the labeling results conduct from 587 smart contracts. Keywords.txt is the keywords we used to filter smart-contract-related posts on StackExchange.

The solidifi dataset: https://github.com/smartbugs/SolidiFI-benchmark

SolidiFI-benchmark repository contains a dataset of buggy contracts injected by 9369 bugs from 7 different bug types, namely, reentrancy, timestamp dependency, uhnadeled exceptions, unchecked send, TOD, integer overflow/underflow, and use of tx.origin. The bugs have been injected in the contracts using SolidiFI.

In addition to the dataset of the vulnerable contracts, the repository contains the injection logs that can be used to refrence the injection locations, where the bugs have been injected in the code, and the type of each bug.

Comparable tools:

Vandal: https://github.com/usyd-blockchain/vandal

Oyente: https://github.com/enzymefinance/oyente

Mythril: https://github.com/ConsenSys/mythril
