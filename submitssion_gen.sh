#!/bin/bash

head -2 nn.py > submission.py
tail -n +3 myFuncTools.py >> submission.py
tail -n +5 nn.py >> submission.py
sed "s/nn.//g" test.py | tail -n +4 >> submission.py
sed "s/autoHyperTung/autoHyperTunning/g" submission.py > ../nn.py
cd ..
rm -rf my-tensorflow
