#!/bin/sh
export LC_ALL='C'
for i in 1 2 3 4 5 6 7 8; do cat output/diffs.set${i}.domain1.test | sort -n -k 1 -r > output/diffs.set${i}.domain1.test.sorted ; done
cat output/diffs.set2.domain2.test | sort -n -k 1 -r > output/diffs.set2.domain2.test.sorted 
cat output/*.sorted | sort -n -k 1 -r > output/all.sorted
