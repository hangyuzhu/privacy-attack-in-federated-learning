BASE_DATADIR="../federated_learning/data"

cd ..
cd ..

# run rtf
# model can be switched to resnet18
python grad_attack.py \
    --attack rtf \
    --model cnn \
    --imprint \
    --base_data_dir $BASE_DATADIR \
    --dataset cifar10 \
    --normalize \
    --device cuda \
    --save_results