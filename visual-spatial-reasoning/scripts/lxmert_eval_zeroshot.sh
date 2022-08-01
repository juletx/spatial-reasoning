CUDA_VISIBLE_DEVICES=$1 python src/eval.py \
	--checkpoint_path tmp/lxmert_zeroshot_split/best_checkpoint/	\
	--img_feature_path data/features/lxmert/ \
	--test_json_path data/splits/zeroshot/test.jsonl \
	--model_type lxmert \
	--output_preds
