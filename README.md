# Weighted Model Counting with Algebraic Decision Diagrams
- This repository supplements Vu Phan's [MS thesis](https://scholarship.rice.edu/handle/1911/107761).
- We provide ADDMC, an exact solver for *weighted model counting (WMC)*.
- ADDMC uses *algebraic decision diagrams (ADDs)*.

--------------------------------------------------------------------------------

## Cloning this repository
```bash
git clone https://github.com/vuphan314/ms-thesis
```

--------------------------------------------------------------------------------

## [Evaluation](./eval/)

--------------------------------------------------------------------------------

## Installation

### Prerequisites
#### External libraries
- automake 1.16
- cmake 3.16
- g++ 11.2
#### [Included libraries](./addmc/libraries/)
- [cudd 3.0](https://github.com/vuphan314/cudd)
- [cxxopts 2.1](https://github.com/jarro2783/cxxopts)

### Command to build statically linked executable
```bash
./INSTALL.sh
```

--------------------------------------------------------------------------------

## Examples

### Showing command-line options
#### Command
```bash
build/addmc -h
```
#### Output
```
==================================================================
ADDMC: Algebraic Decision Diagram Model Counter (help: 'addmc -h')
Version mc-2020, released on 2020/06/07
==================================================================

Usage:
  addmc [OPTION...]

 Optional options:
  -h, --hi      help information
      --cf arg  cnf file path (to use stdin, type: '--cf -')      Default: -
      --wf arg  weight format in cnf file:
           1    UNWEIGHTED
           2    MINIC2D
           3    CACHET
           4    MCC                                               Default: 4
      --ch arg  clustering heuristic:
           1    MONOLITHIC
           2    LINEAR
           3    BUCKET_LIST
           4    BUCKET_TREE
           5    BOUQUET_LIST
           6    BOUQUET_TREE                                      Default: 6
      --cv arg  cluster variable order heuristic (negate to invert):
           1    APPEARANCE
           2    DECLARATION
           3    RANDOM
           4    MCS
           5    LEXP                                              Default: 5
           6    LEXM
      --dv arg  diagram variable order heuristic (negate to invert):
           1    APPEARANCE
           2    DECLARATION
           3    RANDOM
           4    MCS                                               Default: 4
           5    LEXP
           6    LEXM
      --rs arg  random seed                                       Default: 10
      --vl arg  verbosity level:
           0    solution only                                     Default: 0
           1    parsed info as well
           2    clusters as well
           3    cnf literal weights as well
           4    input lines as well
```

### Computing model count given CNF file from stdin
#### Command
```bash
build/addmc < examples/track2_000.mcc2020_wcnf
```
#### Output
```
c ==================================================================
c ADDMC: Algebraic Decision Diagram Model Counter (help: 'addmc -h')
c Version mc-2020, released on 2020/06/07
c ==================================================================

c Process ID of this main program:
c pid 208191

c Reading CNF formula...
c ==================================================================
c Getting cnf from stdin... (end input with 'Enter' then 'Ctrl d')
c Getting cnf from stdin: done
c ==================================================================

c Computing output...
c ------------------------------------------------------------------
s wmc 1.37729e-05
c ------------------------------------------------------------------

c ==================================================================
c seconds                       0.034
c ==================================================================
```

### Computing model count given CNF file with weight format `UNWEIGHTED`
#### Command
```bash
build/addmc --cf examples/UNWEIGHTED.cnf --wf 1
```
#### Output
```
c ==================================================================
c ADDMC: Algebraic Decision Diagram Model Counter (help: 'addmc -h')
c Version mc-2020, released on 2020/06/07
c ==================================================================

c Process ID of this main program:
c pid 358012

c Reading CNF formula...

c Computing output...
c ------------------------------------------------------------------
s mc 1
c ------------------------------------------------------------------

c ==================================================================
c seconds                       0.019
c ==================================================================
```

### Computing model count given CNF file with weight format `MINIC2D`
#### Command
```bash
build/addmc --cf examples/MINIC2D.cnf --wf 2
```
#### Output
```
c ==================================================================
c ADDMC: Algebraic Decision Diagram Model Counter (help: 'addmc -h')
c Version mc-2020, released on 2020/06/07
c ==================================================================

c Process ID of this main program:
c pid 358102

c Reading CNF formula...

c Computing output...
c ------------------------------------------------------------------
s wmc 2.2
c ------------------------------------------------------------------

c ==================================================================
c seconds                       0.018
c ==================================================================
```

### Computing model count given CNF file with weight format `CACHET`
#### Command
```bash
build/addmc --cf examples/CACHET.cnf --wf 3
```
#### Output
```
c ==================================================================
c ADDMC: Algebraic Decision Diagram Model Counter (help: 'addmc -h')
c Version mc-2020, released on 2020/06/07
c ==================================================================

c Process ID of this main program:
c pid 358118

c Reading CNF formula...

c Computing output...
c ------------------------------------------------------------------
s wmc 0.3
c ------------------------------------------------------------------

c ==================================================================
c seconds                       0.019
c ==================================================================
```

--------------------------------------------------------------------------------

## Acknowledgment
- Tabajara: [RSynth](https://bitbucket.org/lucas-mt/rsynth)
- Somenzi: [CUDD package](https://github.com/ivmai/cudd)
- Rutenbar: [CUDD tutorial](http://db.zmitac.aei.polsl.pl/AO/dekbdd/F01-CUDD.pdf)
- Kebo: [CUDD visualization](https://davidkebo.com/cudd#cudd6)
- Beck: [cxxopts](https://github.com/jarro2783/cxxopts)
- Kautz and Sang: [Cachet](https://cs.rochester.edu/u/kautz/Cachet)
- Hecher and Fichte: [model-counting competition](https://mccompetition.org/2020/mc_format)
