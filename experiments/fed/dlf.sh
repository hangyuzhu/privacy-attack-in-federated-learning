BASE_DATADIR="../federated_learning/data"

cd ..
cd ..

# run dlf
# non-iid
python server_attack.py \
    --attack dlf \
    --model cnn3 \
    --base_data_dir $BASE_DATADIR \
    --dataset cifar100 \
    --normalize \
    --p_type dirichlet \
    --beta 0.5 \
    --total_clients 100 \
    --num_rounds 1 \
    --local_epochs 1 \
    --batch_size 10 \
    --lr 0.004 \
    --lr_decay 0.95 \
    --client_momentum 0 \
    --rec_epochs 200 \
    --rec_batch_size 10 \
    --rec_lr 0.1 \
    --tv 0.0002 \
    --reg_clip 10 \
    --reg_reorder 6.075 \
    --device cuda \
    --save_results