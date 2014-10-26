#!/bin/sh

python setup.py build_ext --inplace
pythran -o d2q9_nxnyns_pythran.so d2q9_nxnyns.py
