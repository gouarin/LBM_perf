#!/bin/sh

python setup.py build_ext --inplace
pythran -o d2q9_nxnyns_vec_pythran.so d2q9_nxnyns_vec.py
pythran -o d2q9_nxnyns_loop_pythran.so d2q9_nxnyns_loop.py
pythran -o d2q9_nsnxny_vec_pythran.so d2q9_nsnxny_vec.py
pythran -o d2q9_nsnxny_loop_pythran.so d2q9_nsnxny_loop.py
