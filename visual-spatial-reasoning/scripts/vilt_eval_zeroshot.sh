CUDA_VISIBLE_DEVICES=$1 python src/eval.py \
	--checkpoint_path tmp/vilt_zeroshot_split/best_checkpoint\
	--img_feature_path data/images/ \
	--test_json_path data/splits/zeroshot/test.jsonl \
	--model_type vilt \
	--output_preds
