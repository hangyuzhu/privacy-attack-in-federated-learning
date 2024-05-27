BASE_DATADIR="../federated_learning/data"

cd ..
cd ..

# run ig single
python grad_attack.py \
    --attack ig_single \
    --model cnn \
    --base_data_dir $BASE_DATADIR \
    --dataset cifar10 \
    --device cuda \
    --save_results

# run ig weight
python grad_attack.py \
    --attack ig_weight \
    --model cnn \
    --base_data_dir $BASE_DATADIR \
    --dataset cifar10 \
    --device cuda \
    --save_results

# run ig multi
python grad_attack.py \
    --attack ig_multi \
    --model cnn \
    --base_data_dir $BASE_DATADIR \
    --dataset cifar10 \
    --device cuda \
    --save_results