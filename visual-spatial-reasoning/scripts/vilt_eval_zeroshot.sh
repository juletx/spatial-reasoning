CUDA_VISIBLE_DEVICES=$1 python src/eval.py \
	--checkpoint_path tmp/vilt-vsr-zeroshot\
    --result_path results/vilt_zeroshot.txt \
	--img_feature_path data/images/ \
	--test_json_path data/splits/zeroshot/test.jsonl \
	--model_type vilt \
	--output_preds
