CUDA_VISIBLE_DEVICES=$1 python src/eval.py \
	--checkpoint_path dandelin/vilt-b32-finetuned-nlvr2 \
    --result_path results/vilt_nlvr2_random.txt \
	--img_feature_path data/images/ \
	--test_json_path data/splits/random/test.jsonl \
	--model_type vilt_nlvr2 \
	--output_preds
