export LC_ALL='C'
for i in 1 2 3 4 5 6 7 8; do cat val.set${i}.domain1.tsv | sort -n -k 1 -r > val.set${i}.domain1.tsv.sorted ; done
cat val.set2.domain2.tsv | sort -n -k 1 -r > val.set2.domain2.tsv.sorted
cat *.sorted | sort -n -k 1 -r > all.sorted
