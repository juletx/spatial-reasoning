from matplotlib import pyplot as plt
import pandas as pd

df_visualbert = pd.read_csv("results/visualbert_zeroshot_rel.tsv", sep="\t", index_col=0, names=["relation", "VisualBERT", "number"])
df_visualbert = df_visualbert[["number", "VisualBERT"]]
df = df_visualbert

df_lxmert = pd.read_csv("results/lxmert_zeroshot_rel.tsv", sep="\t", index_col=0, names=["relation", "LXMERT", "number"])
df["LXMERT"] = df_lxmert["LXMERT"]

df_vilt = pd.read_csv("results/vilt_zeroshot_rel.tsv", sep="\t", index_col=0, names=["relation", "ViLT", "number"])
df["ViLT"] = df_vilt["ViLT"]

df_vilt_nlvr2 = pd.read_csv("results/vilt_nlvr2_zeroshot_rel.tsv", sep="\t", index_col=0, names=["relation", "ViLT NLVR2", "number"])
df["ViLT NLVR2"] = df_vilt_nlvr2["ViLT NLVR2"]

df_blip_nlvr2 = pd.read_csv("results/blip_nlvr2_zeroshot_rel.tsv", sep="\t", index_col=0, names=["relation", "BLIP NLVR2", "number"])
df["BLIP NLVR2"] = df_blip_nlvr2["BLIP NLVR2"]

df.round(1).to_csv("results/zeroshot_rel.tsv", sep="\t")

df = df[df["number"] >= 5]

df.round(1).to_latex("results/zeroshot_rel.tex")

ax = df.plot.bar(y=["VisualBERT", "LXMERT", "ViLT", "ViLT NLVR2", "BLIP NLVR2"], figsize=(15,5))

plt.axhline(y=50, linewidth=1, color='r', linestyle="dashdot", label="random baseline")
plt.axhline(y=75, linewidth=1, color='gray')
plt.axhline(y=95.4, linewidth=1, color='b', linestyle="dashed", label="human ceiling")
plt.yticks([0, 25, 50, 75, 100])
plt.legend()
plt.xlabel("")
plt.ylabel("accuracy")
plt.tight_layout()

fig = ax.get_figure()
fig.savefig("figures/performance_rel_zeroshot.png")
