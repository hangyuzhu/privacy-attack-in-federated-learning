BASE_DATADIR="../federated_learning/data"

cd ..
cd ..

# the outcomes are sensitive to data normalization
# run grnn
python grad_attack.py \
    --attack grnn \
    --model cnn \
    --base_data_dir $BASE_DATADIR \
    --dataset cifar100 \
    --normalize \
    --device cuda \
    --save_results

# unnormalize data generates images with high quality
python grad_attack.py \
    --attack grnn \
    --model cnn \
    --base_data_dir $BASE_DATADIR \
    --dataset cifar100 \
    --device cuda \
    --save_results