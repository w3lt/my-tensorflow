#!/bin/bash

head -2 nn.py > submission.py
tail -n +3 myFuncTools.py >> submission.py
tail -n +5 nn.py >> submission.py
sed "s/nn.//g" test.py | tail -n +4 >> submission.py
cd ..
cp my-tensorflow/submission.py ./nn.py
rm -rf my-tensorflow