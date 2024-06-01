BASE_DATADIR="../federated_learning/data"

cd ..
cd ..

# run cpa
python grad_attack.py \
    --attack cpa \
    --model fc2 \
    --base_data_dir $BASE_DATADIR \
    --dataset tiny_imagenet \
    --normalize \
    --device cuda \
    --save_results

# run cpa
python grad_attack.py \
    --attack cpa \
    --model vgg16 \
    --base_data_dir $BASE_DATADIR \
    --dataset imagenet \
    --normalize \
    --device cuda \
    --save_results