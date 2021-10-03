export CUDA_VISIBLE_DEVICES=2
mkdir -p ./logs
Model=GGNN     # GGNN RGCN RGAT RGIN GNN-Edge-MLP RGDCN GNN-FiLM
#python train.py ${Model} VarMisuse 2>&1| tee ./logs/${Model}_train.log
TYPE=("graphs-test" "graphs-testonly")
Interval=("." "threshold_1_3" "threshold_3_5" "threshold_5_6" "threshold_6_7" "threshold_7_8" "threshold_8_10" "threshold_unreachable")
Model_path='./trained_models/VarMisuse_GGNN_2021-09-26-03-46-02_18506_best_model.pickle'
for type in ${TYPE[*]}; do
  for interval in ${Interval[*]}; do
    python test.py ${Model_path} ./data/varmisuse/${interval}/${type} 2>&1| tee ./logs/${Model}_test.log
  done
done
