from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv("results/vilt_random_rel.tsv", sep="\t", index_col=0, names=["rel", "ViLT", "occur"])
df = df[df["occur"] >= 15]
df_nlvr2 = pd.read_csv("results/vilt_nlvr2_random_rel.tsv", sep="\t", index_col=0, names=["rel", "ViLT NLVR2", "occur"])
df_nlvr2 = df_nlvr2[df_nlvr2["occur"] >= 15]
df["ViLT NLVR2"] = df_nlvr2["ViLT NLVR2"]

ax = df.plot.bar(y=["ViLT", "ViLT NLVR2"], figsize=(15,5))

plt.axhline(y=0.5, linewidth=1, color='r', linestyle="dashdot", label="random baseline")
plt.axhline(y=0.954, linewidth=1, color='b', linestyle="dashed", label="human ceiling")
plt.legend()
plt.ylabel("accuracy")
plt.tight_layout()

fig = ax.get_figure()
fig.savefig("figures/performance_rel_random.png")
