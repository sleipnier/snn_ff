python shd_ff_train_v1.py --device cuda:0,cuda:1 -out-dir ./result/2layers/lif/ -b 4096 -epochs 100 --model lif --tau 4.0 --v-threshold 1.0 --input-gain 1.5 --lr 3e-4 --weight-decay 1e-05 
