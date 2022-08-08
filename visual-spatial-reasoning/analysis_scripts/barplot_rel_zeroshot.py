from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv("results/vilt_zeroshot_rel.tsv", sep="\t", index_col=0, names=["rel", "ViLT", "occur"])
df = df[df["occur"] >= 5]

df_nlvr2 = pd.read_csv("results/vilt_nlvr2_zeroshot_rel.tsv", sep="\t", index_col=0, names=["rel", "ViLT NLVR2", "occur"])
df_nlvr2 = df_nlvr2[df_nlvr2["occur"] >= 5]
df["ViLT NLVR2"] = df_nlvr2["ViLT NLVR2"]

df_nlvr2 = pd.read_csv("results/blip_nlvr2_zeroshot_rel.tsv", sep="\t", index_col=0, names=["rel", "BLIP NLVR2", "occur"])
df_nlvr2 = df_nlvr2[df_nlvr2["occur"] >= 5]
df["BLIP NLVR2"] = df_nlvr2["BLIP NLVR2"]

df_visualbert = pd.read_csv("results/visualbert_zeroshot_rel.tsv", sep="\t", index_col=0, names=["rel", "VisualBERT", "occur"])
df_visualbert = df_visualbert[df_visualbert["occur"] >= 5]
df["VisualBERT"] = df_visualbert["VisualBERT"]

df_lxmert = pd.read_csv("results/lxmert_zeroshot_rel.tsv", sep="\t", index_col=0, names=["rel", "LXMERT", "occur"])
df_lxmert = df_lxmert[df_lxmert["occur"] >= 5]
df["LXMERT"] = df_lxmert["LXMERT"]

ax = df.plot.bar(y=["VisualBERT", "LXMERT", "ViLT", "ViLT NLVR2", "BLIP NLVR2"], figsize=(15,5))

plt.axhline(y=0.5, linewidth=1, color='r', linestyle="dashdot", label="random baseline")
plt.axhline(y=0.954, linewidth=1, color='b', linestyle="dashed", label="human ceiling")
plt.legend()
plt.ylabel("accuracy")
plt.tight_layout()

fig = ax.get_figure()
fig.savefig("figures/performance_rel_zeroshot.png")
