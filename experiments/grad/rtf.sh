BASE_DATADIR="../federated_learning/data"

cd ..
cd ..

# run rtf
python grad_attack.py \
    --attack rtf \
    --model cnn \   # resnet18
    --imprint \
    --base_data_dir $BASE_DATADIR \
    --dataset cifar10 \
    --device cuda \
    --save_results