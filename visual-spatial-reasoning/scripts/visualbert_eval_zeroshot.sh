CUDA_VISIBLE_DEVICES=$1 python src/eval.py \
	--checkpoint_path tmp/visualbert-vsr-zeroshot \
    --result_path results/visualbert_zeroshot.txt \
	--img_feature_path data/features/visualbert/ \
	--test_json_path data/splits/zeroshot/test.jsonl \
	--model_type visualbert \
	--output_preds
