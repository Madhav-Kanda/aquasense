#!/usr/bin/env bash

#trans=reparam
for trans in original student reparamApprox reparamTrue reweight local1; do
    echo "$trans={" $(for i in `seq 0 10`; do value=$(python lightspeed_robust_$trans.py 100 $i); echo $value,; done) "}"
done
