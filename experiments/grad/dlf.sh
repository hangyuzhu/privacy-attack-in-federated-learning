BASE_DATADIR="../federated_learning/data"

cd ..
cd ..

# you can change if using restore_label or not
# inside the file dlf_attack.py
python grad_attack.py \
    --attack dlf \
    --model cnn3 \
    --base_data_dir $BASE_DATADIR \
    --dataset cifar100 \
    --normalize \
    --device cuda \
    --save_results