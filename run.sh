echo NAIC Online Score Reproduction of DMT

#echo Train
#
python divided_dataset.py --data_dir_query ../data/MyDataSet/image_B_v1_1/query --data_dir_gallery ../data/MyDataSet/image_B_v1_1/gallery --save_dir ../data/MyDataSet/image_B_v1_1/
#
#python train.py --config_file configs/naic_round2_model_a.yml
#
#python train_UDA.py --config_file configs/naic_round2_model_b.yml --config_file_test configs/naic_round2_model_a.yml --data_dir_query ../data/test/query_a --data_dir_gallery ../data/test/gallery_a
#
#python train_UDA.py --config_file configs/naic_round2_model_se.yml --config_file_test configs/naic_round2_model_b.yml --data_dir_query ../data/test/query_a --data_dir_gallery ../data/test/gallery_a
#
echo Test
## here we will get Distmat Matrix after test.
#python test.py --config_file configs/naic_round2_model_a.yml

python test.py --config_file configs/naic_round2_model_b_128.yml
python test.py --config_file configs/naic_round2_model_b_192.yml
python test.py --config_file configs/naic_round2_model_b_240.yml

python test.py --config_file configs/naic_round2_model_b_128_50.yml
python test.py --config_file configs/naic_round2_model_b_192_50.yml
python test.py --config_file configs/naic_round2_model_b_240_50.yml

#python test.py --config_file configs/naic_round2_model_se.yml
#
#echo Ensemble
#
python ensemble_dist.py