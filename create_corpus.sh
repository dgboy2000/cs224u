filename='data/training_set_rel3.tsv'
lns=`(cat $filename|wc -l)`
let lns=$lns-1
cat $filename | awk -F'\t' '{print $3}' | tail -n $lns > data/corpus.txt

filename='data/valid_set.tsv'
lns=`(cat $filename|wc -l)`
let lns=$lns-1
cat $filename | awk -F'\t' '{print $3}' | tail -n $lns >> data/corpus.txt

