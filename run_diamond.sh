data_root=data
diamond makedb --in $data_root/train_data.fa --db $data_root/train_data.dmnd
# Run blastp
diamond blastp --very-sensitive -d $data_root/train_data.dmnd -q $data_root/test_data.fa --outfmt 6 qseqid sseqid bitscore pident > $data_root/test_diamond.res
