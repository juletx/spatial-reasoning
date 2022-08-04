CUDA_VISIBLE_DEVICES=$1 python src/eval.py \
	--checkpoint_path tmp/lxmert-vsr-random/	\
    --result_path results/lxmert_random.txt \
	--img_feature_path data/features/lxmert/ \
	--test_json_path data/splits/random/test.jsonl \
	--model_type lxmert \
	--output_preds
