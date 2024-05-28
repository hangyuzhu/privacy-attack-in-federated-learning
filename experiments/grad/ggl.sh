BASE_DATADIR="../federated_learning/data"

cd ..
cd ..

# run ggl
# inside the code select if pretrain GGL generator or not
python grad_attack.py \
    --attack ggl \
    --model resnet18 \
    --base_data_dir $BASE_DATADIR \
    --dataset cifar10 \
    --normalize \
    --device cuda \
    --save_results