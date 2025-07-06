set -e

data_root=$1


python diamond_data.py -df $data_root/train_data.pkl -of $data_root/train_data.fa
python diamond_data.py -df $data_root/valid_data.pkl -of $data_root/valid_data.fa
python diamond_data.py -df $data_root/test_data.pkl -of $data_root/test_data.fa

# Add validation sequences to training
cat $data_root/valid_data.fa >> $data_root/train_data.fa

# Create diamond database
diamond makedb --in $data_root/train_data.fa --db $data_root/train_data.dmnd
# Run blastp
diamond blastp --more-sensitive -d $data_root/train_data.dmnd -q $data_root/test_data.fa --outfmt 6 qseqid sseqid bitscore pident > $data_root/test_diamond.res
