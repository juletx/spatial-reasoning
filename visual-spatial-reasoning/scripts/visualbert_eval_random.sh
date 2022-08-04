CUDA_VISIBLE_DEVICES=$1 python src/eval.py \
	--checkpoint_path tmp/visualbert-vsr-random \
    --result_path results/visualbert_random.txt \
	--img_feature_path data/features/visualbert/ \
	--test_json_path data/splits/random/test.jsonl \
	--model_type visualbert \
	--output_preds
