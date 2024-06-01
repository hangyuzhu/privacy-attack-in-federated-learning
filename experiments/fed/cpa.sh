BASE_DATADIR="../federated_learning/data"

cd ..
cd ..

# run cpa
# non-iid
python server_attack.py \
    --attack cpa \
    --model fc2 \
    --base_data_dir $BASE_DATADIR \
    --dataset tiny_imagenet \
    --normalize \
    --p_type dirichlet \
    --beta 0.5 \
    --total_clients 10 \
    --num_rounds 10 \
    --local_epochs 1 \
    --batch_size 50 \
    --lr 0.1 \
    --lr_decay 0.95 \
    --client_momentum 0 \
    --rec_epochs 25000 \
    --rec_batch_size 4 \
    --rec_lr 0.001 \
    --decor 1.47 \
    --T 12.4 \
    --tv 3.1 \
    --nv 0 \
    --l1 0 \
    --fi 1 \
    --device cuda \
    --save_results

# run cpa + fi
# non-iid
# model is not pretrained
python server_attack.py \
    --attack cpa \
    --model vgg16 \
    --base_data_dir $BASE_DATADIR \
    --dataset tiny_imagenet \
    --normalize \
    --p_type dirichlet \
    --beta 0.5 \
    --total_clients 10 \
    --num_rounds 10 \
    --local_epochs 1 \
    --batch_size 50 \
    --lr 0.1 \
    --lr_decay 0.95 \
    --client_momentum 0 \
    --rec_epochs 25000 \
    --rec_batch_size 4 \
    --rec_lr 1e-3 \
    --decor 5.3 \
    --T 7.7 \
    --tv 0.1 \
    --nv 0.13 \
    --l1 5 \
    --fi 1 \
    --device cuda \
    --save_results