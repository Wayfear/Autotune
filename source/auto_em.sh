#!/bin/bash
PS4="#:"
set -x
dimension='512'
model_name='20180402-114759.pb'
python clear_pk.py
python feature_extraction_classifier.py --model_dir $model_name --dimension_num $dimension
para=15
sta_num=20
en_num=70
start_num=20
end_num=70
for i in {0..15}
do
    now=$(date '+%Y-%m-%d-%H-%M-%S')
    python match.py --copy_pic 0 --cycle_num $i --start_num $start_num --end_num $end_num --header $now --dimension_num $dimension --hyper_para $para
    python voting.py --start_num $sta_num --end_num $en_num --header $now --dimension_num $dimension --flag 0
    python evaluate_by_global.py --load_label 1 --header $now
    python generate_train_data.py --source_data_dir $now\_$sta_num\_$en_num\_voting --data_dir $now
    python fine_tune_soft.py --epoch_size 5 --max_nrof_epochs 5 --data_dir $now --center_loss_factor 0.1 --embedding_size $dimension --keep_probability 1
    python freeze_graph.py --model_dir $now\_center_loss_factor_0.10  --output_file $now.pb
    python feature_extraction_classifier.py --model_dir $now.pb --dimension_num $dimension
    python match.py --copy_pic 0 --cycle_num $i --start_num $start_num --end_num $end_num --header $now --dimension_num $dimension --hyper_para $para
    python voting.py --start_num $sta_num --end_num $en_num --header $now
    python occ_update.py --load_label 1 --err_rate 0.05 --cycle_num $((i+1)) --center_file $now\_voting_center.pk
    let para-=1
    #para=$(echo "$para - $decay" | bc)
done

python finish.py
