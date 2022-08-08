python analysis_scripts/eval_compute_acc_by_rel.py data/splits/random/test.jsonl results/visualbert_random.txt > results/visualbert_random_rel.tsv

python analysis_scripts/eval_compute_acc_by_rel.py data/splits/zeroshot/test.jsonl results/visualbert_zeroshot.txt > results/visualbert_zeroshot_rel.tsv

python analysis_scripts/eval_compute_acc_by_rel.py data/splits/random/test.jsonl results/lxmert_random.txt > results/lxmert_random_rel.tsv

python analysis_scripts/eval_compute_acc_by_rel.py data/splits/zeroshot/test.jsonl results/lxmert_zeroshot.txt > results/lxmert_zeroshot_rel.tsv

python analysis_scripts/eval_compute_acc_by_rel.py data/splits/random/test.jsonl results/vilt_random.txt > results/vilt_random_rel.tsv

python analysis_scripts/eval_compute_acc_by_rel.py data/splits/random/test.jsonl results/vilt_nlvr2_random.txt > results/vilt_nlvr2_random_rel.tsv

python analysis_scripts/eval_compute_acc_by_rel.py data/splits/zeroshot/test.jsonl results/vilt_zeroshot.txt > results/vilt_zeroshot_rel.tsv

python analysis_scripts/eval_compute_acc_by_rel.py data/splits/zeroshot/test.jsonl results/vilt_nlvr2_zeroshot.txt > results/vilt_nlvr2_zeroshot_rel.tsv

python analysis_scripts/eval_compute_acc_by_rel.py data/splits/random/test.jsonl results/blip_nlvr2_random.txt > results/blip_nlvr2_random_rel.tsv

python analysis_scripts/eval_compute_acc_by_rel.py data/splits/zeroshot/test.jsonl results/blip_nlvr2_zeroshot.txt > results/blip_nlvr2_zeroshot_rel.tsv