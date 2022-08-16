from matplotlib import pyplot as plt
import pandas as pd

df_visualbert = pd.read_csv("results/visualbert_random_rel_meta_cat.tsv", sep="\t", index_col=0, names=["category", "VisualBERT", "number"])
df_visualbert = df_visualbert[["number", "VisualBERT"]]
df = df_visualbert

df_lxmert = pd.read_csv("results/lxmert_random_rel_meta_cat.tsv", sep="\t", index_col=0, names=["category", "LXMERT", "number"])
df["LXMERT"] = df_lxmert["LXMERT"]

df_vilt = pd.read_csv("results/vilt_random_rel_meta_cat.tsv", sep="\t", index_col=0, names=["category", "ViLT", "number"])
df["ViLT"] = df_vilt["ViLT"]

df_nlvr2 = pd.read_csv("results/vilt_nlvr2_random_rel_meta_cat.tsv", sep="\t", index_col=0, names=["category", "ViLT NLVR2", "number"])
df["ViLT NLVR2"] = df_nlvr2["ViLT NLVR2"]

df_blip_nlvr2 = pd.read_csv("results/blip_nlvr2_random_rel_meta_cat.tsv", sep="\t", index_col=0, names=["category", "BLIP NLVR2", "number"])
df["BLIP NLVR2"] = df_blip_nlvr2["BLIP NLVR2"]

df.round(1).to_csv("results/random_rel_meta_cat.tsv", sep="\t")
df.round(1).to_latex("results/random_rel_meta_cat.tex")

ax = df.plot.barh(y=["VisualBERT", "LXMERT", "ViLT", "ViLT NLVR2", "BLIP NLVR2"], figsize=(5,5))

plt.axvline(x=50, linewidth=1, color='r', linestyle="dashdot", label="random baseline")
plt.axvline(x=75, linewidth=1, color='gray')
plt.axvline(x=95.4, linewidth=1, color='b', linestyle="dashed", label="human ceiling")
plt.xticks([0, 25, 50, 75, 100])
plt.legend()
plt.xlabel("accuracy")
plt.ylabel("")
plt.gca().invert_yaxis()
plt.tight_layout()

fig = ax.get_figure()
fig.savefig("figures/performance_rel_meta_cat_random.png")
