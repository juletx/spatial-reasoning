CUDA_VISIBLE_DEVICES=$1 python src/eval.py \
	--checkpoint_path https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_nlvr.pth \
    --result_path results/blip_nlvr2_zeroshot_dev.txt \
	--img_feature_path data/images/ \
	--test_json_path data/splits/zeroshot/dev.jsonl \
	--model_type blip_nlvr2 \
	--output_preds
