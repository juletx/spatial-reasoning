CUDA_VISIBLE_DEVICES=$1 python src/eval.py \
	--checkpoint_path tmp/lxmert-vsr-zeroshot/	\
    --result_path results/lxmert_zeroshot.txt \
	--img_feature_path data/features/lxmert/ \
	--test_json_path data/splits/zeroshot/test.jsonl \
	--model_type lxmert \
	--output_preds
