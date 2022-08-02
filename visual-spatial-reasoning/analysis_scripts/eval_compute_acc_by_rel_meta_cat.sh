python analysis_scripts/eval_compute_acc_by_rel_meta_cat.py data/splits/random/test.jsonl results/visualbert_random.txt analysis_scripts/rel_meta_category_dict.txt > results/visualbert_random_rel_meta_cat.tsv

python analysis_scripts/eval_compute_acc_by_rel_meta_cat.py data/splits/random/test.jsonl results/lxmert_random.txt analysis_scripts/rel_meta_category_dict.txt > results/lxmert_random_rel_meta_cat.tsv

python analysis_scripts/eval_compute_acc_by_rel_meta_cat.py data/splits/random/test.jsonl results/vilt_random.txt analysis_scripts/rel_meta_category_dict.txt  > results/vilt_random_rel_meta_cat.tsv

python analysis_scripts/eval_compute_acc_by_rel_meta_cat.py data/splits/random/test.jsonl results/vilt_nlvr2_random.txt analysis_scripts/rel_meta_category_dict.txt  > results/vilt_nlvr2_random_rel_meta_cat.tsv

python analysis_scripts/eval_compute_acc_by_rel_meta_cat.py data/splits/zeroshot/test.jsonl results/vilt_zeroshot.txt analysis_scripts/rel_meta_category_dict.txt  > results/vilt_zeroshot_rel_meta_cat.tsv

python analysis_scripts/eval_compute_acc_by_rel_meta_cat.py data/splits/zeroshot/test.jsonl results/vilt_nlvr2_zeroshot.txt analysis_scripts/rel_meta_category_dict.txt  > results/vilt_nlvr2_zeroshot_rel_meta_cat.tsv