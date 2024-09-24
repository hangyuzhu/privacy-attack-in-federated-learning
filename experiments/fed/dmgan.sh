BASE_DATADIR="../federated_learning/data"

cd ..
cd ..

# run dmgan
# non-iid
python client_attack.py \
    --base_data_dir $BASE_DATADIR \
    --dataset mnist \
    --normalize \
    --p_type dirichlet \
    --beta 0.5 \
    --total_clients 10 \
    --num_rounds 30 \
    --local_epochs 2 \
    --batch_size 50 \
    --lr 0.1 \
    --lr_decay 0.95 \
    --client_momentum 0 \
    --tracked_class 3 \
    --rec_epochs 10 \
    --rec_batch_size 1 \
    --device cuda \
    --save_results