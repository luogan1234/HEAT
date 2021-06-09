python init.py
for encoder in "bilstm" "bert"; do
  for dataset in "type_net" "med_mentions"; do
    for task in 0; do
      for seed in {0..1}; do
        cmd="python main.py -dataset $dataset -text_encoder $encoder -task $task -remove_img -seed $seed"
        echo $cmd & $cmd
      done
    done
  done
  for dataset in "flowers" "birds"; do
    for task in 0; do
      for seed in {0..1}; do
        cmd="python main.py -dataset $dataset -text_encoder $encoder -task $task -remove_name -seed $seed"
        echo $cmd & $cmd
      done
    done
  done
done
