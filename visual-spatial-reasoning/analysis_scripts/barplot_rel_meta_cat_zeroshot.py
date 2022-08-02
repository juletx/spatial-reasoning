from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv("results/vilt_zeroshot_rel_meta_cat.tsv", sep="\t", index_col=0, names=["rel_meta_cat", "ViLT", "occur"])

df_nlvr2 = pd.read_csv("results/vilt_nlvr2_zeroshot_rel_meta_cat.tsv", sep="\t", index_col=0, names=["rel_meta_cat", "ViLT NLVR2", "occur"])
df["ViLT NLVR2"] = df_nlvr2["ViLT NLVR2"]

df_visualbert = pd.read_csv("results/visualbert_zeroshot_rel_meta_cat.tsv", sep="\t", index_col=0, names=["rel", "VisualBERT", "occur"])
df["VisualBERT"] = df_visualbert["VisualBERT"]

df_lxmert = pd.read_csv("results/lxmert_zeroshot_rel_meta_cat.tsv", sep="\t", index_col=0, names=["rel", "LXMERT", "occur"])
df["LXMERT"] = df_lxmert["LXMERT"]

ax = df.plot.barh(y=["VisualBERT", "LXMERT", "ViLT", "ViLT NLVR2"], figsize=(5,5))

plt.axvline(x=0.5, linewidth=1, color='r', linestyle="dashdot", label="random baseline")
plt.axvline(x=0.954, linewidth=1, color='b', linestyle="dashed", label="human ceiling")
plt.legend()
plt.xlabel("accuracy")
plt.tight_layout()

fig = ax.get_figure()
fig.savefig("figures/performance_rel_meta_cat_zeroshot.png")
