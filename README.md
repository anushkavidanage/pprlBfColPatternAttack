Efficient pattern-mining based attack on Bloom filter encoding for PPRL
=======================================================================

Anushka Vidanage, Peter Christen, Thilina Ranbaduge, and Rainer Schnell 

Paper title: Efficient Pattern Mining based Cryptanalysis for
             Privacy-Preserving Record Linkage


Copyright 2018 Australian National University and others.
All Rights reserved.

See the file COPYING for the terms under which the computer program
code and associated documentation in this package are licensed.

15 October 2018.

Contact: anushka.vidanage@anu.edu.au

-------------------------------------------------------------------

Requirements:
=============

The Python programs included in this package were written and
tested using Python 2.7.6 running on Ubuntu 16.04

The following extra packages are required:
- numpy
- bitarray:  https://pypi.python.org/pypi/bitarray

Running the attack program:
===========================

To run the program, use the following command (with an example setting):

  python pprl_bf_col_pattern_attack.py 2 dh 10 1000 
    none clk False [maxminer] 1.0 5.0 10000 path-to-encode-dataset.csv.gz 0 , 
    True [1] path-to-plain-text-dataset.csv.gz 0 , True [1] 10 bf_tuple
    single 5 None None

For moe details about the command line arguments see comments at the top of 
'pprl_bf_col_pattern_attack.py'
