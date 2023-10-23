#!/usr/bin/env bash

for trans in original student reparamApprox reparamTrue reweight local1; do
    python lightspeed_robust_$trans.py 100 0 > results/diff_data_eps_0_$trans.txt
done
