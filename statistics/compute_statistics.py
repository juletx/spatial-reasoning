import json
import statistics
from tabulate import tabulate
import numpy as np
import random
import scipy.stats
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from tags_with_examples import tags_with_examples
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from matplotlib.lines import Line2D
random.seed(0)

def text_correct(result):
    return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

def image_correct(result):
    return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

def group_correct(result):
    return image_correct(result) and text_correct(result)

def load_results(filename):
    results = {}
    for line in open(filename, "r").readlines():
        score_dict = json.loads(line)
        id, c, i = score_dict["label"].split("_")
        id = int(id)
        scores = results.get(id, {})
        scores["_".join([c,i])] = score_dict["score"]
        results[id] = scores
    return results

def get_ordered_caption_lengths():
    ordered_list = []
    examples = sorted([json.loads(line) for line in open("data/examples.jsonl").readlines()], key=lambda example: example["id"])
    for example in tqdm(examples):
        length = statistics.mean([len(example["caption_0"].split()), len(example["caption_1"].split())])
        ordered_list.append(length)
    return ordered_list

def get_ordered_gpt2_perplexities():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", pad_token="<PAD>")
    ordered_list = []
    examples = sorted([json.loads(line) for line in open("data/examples.jsonl").readlines()], key=lambda example: example["id"])

    def get_perplexity(text):
        encodings = tokenizer([text], padding=True, return_tensors="pt", return_special_tokens_mask=True)
        encoded_texts = encodings["input_ids"]
        special_tokens_masks = encodings["special_tokens_mask"]

        max_model_length = model.config.n_positions

        ppls = []
        stride=1
        for text_index in range(0, len(encoded_texts)):
            encoded_text = encoded_texts[text_index]
            special_tokens_mask = special_tokens_masks[text_index]

            encoded_text_length = len(encoded_text) - special_tokens_mask.sum()

            nlls = []

            target_index = max(1, min(stride - 1, encoded_text_length - 1))

            while target_index < encoded_text_length:
                start_index = max(0, target_index - (max_model_length - 1))

                input_ids = encoded_text[start_index : target_index + 1]

                target_ids = input_ids.clone()
                target_ids[:-1] = -100

                attn_mask = torch.ones(len(input_ids))
                attn_mask[-1] = 0

                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids, attention_mask=attn_mask)
                    neg_log_likelihood = outputs[0]

                nlls.append(neg_log_likelihood)

                target_index += stride

            if len(nlls) > 0:
                ppls.append(torch.exp2(torch.mean(torch.stack(nlls))))

        ppl = torch.mean(torch.stack(ppls))
        return ppl

    for example in tqdm(examples):
        caption_0_perplexity = get_perplexity(example["caption_0"])
        caption_1_perplexity = get_perplexity(example["caption_1"])
        ordered_list.append(caption_0_perplexity)
        ordered_list.append(caption_0_perplexity)
        ordered_list.append(caption_1_perplexity)
        ordered_list.append(caption_1_perplexity)
    return ordered_list

def get_ordered_model_group_correct(results):
    ordered_list = []
    examples = sorted([json.loads(line) for line in open("data/examples.jsonl").readlines()], key=lambda example: example["id"])
    for i in range(len(examples)):
        ordered_list.append(group_correct(results[i]))
    return ordered_list

def get_ordered_model_scores_for_perplexity_comparison(results):
    ordered_list = []
    examples = sorted([json.loads(line) for line in open("data/examples.jsonl").readlines()], key=lambda example: example["id"])
    for i in range(len(examples)):
        ordered_list.append(results[i]["c0_i0"])
        ordered_list.append(results[i]["c0_i1"])
        ordered_list.append(results[i]["c1_i0"])
        ordered_list.append(results[i]["c1_i1"])
    return ordered_list

def filter_examples(filter_dict):
    examples = [json.loads(line) for line in open("data/examples.jsonl").readlines()]
    def filter_criteria(example):
        for key, value in filter_dict.items():
            if example[key] != value:
                return False
        return True
    filtered_ids = {example["id"] for example in filter(filter_criteria, examples)}
    return filtered_ids

def get_table_dict(models, filtered_ids=None, bootstrap_confidence_intervals=False):

    table_dict = {"Model": [], "Text": [], "Image": [], "Group": []}
    for filename, display_name, _, _, _ in models:
        results = load_results(filename)
        if filtered_ids is not None:
            filtered_results = {}
            for key, value in results.items():
                if key in filtered_ids:
                    filtered_results[key] = value
            results = filtered_results

        text_results = [text_correct(result) for result in results.values()]
        image_results = [image_correct(result) for result in results.values()]
        group_results = [group_correct(result) for result in results.values()]
        if bootstrap_confidence_intervals:
            text_bootstrap_result = scipy.stats.bootstrap(data=(text_results,), statistic=scipy.mean)
            image_bootstrap_result = scipy.stats.bootstrap(data=(image_results,), statistic=scipy.mean)
            group_bootstrap_result = scipy.stats.bootstrap(data=(group_results,), statistic=scipy.mean)
            text_score = "{:0.2f}".format(100*statistics.mean(text_results))\
                + " & [" + "{:0.2f}".format(100*text_bootstrap_result.confidence_interval.low)\
                + "," + "{:0.2f}".format(100*text_bootstrap_result.confidence_interval.high) + "]"
            image_score = "{:0.2f}".format(100*statistics.mean(image_results))\
                + " & [" + "{:0.2f}".format(100*image_bootstrap_result.confidence_interval.low)\
                + "," + "{:0.2f}".format(100*image_bootstrap_result.confidence_interval.high) + "]"
            group_score = "{:0.2f}".format(100*statistics.mean(group_results))\
                + " & [" + "{:0.2f}".format(100*group_bootstrap_result.confidence_interval.low)\
                + "," + "{:0.2f}".format(100*group_bootstrap_result.confidence_interval.high) + "]"
        else:
            text_score = 100*statistics.mean(text_results)
            image_score = 100*statistics.mean(image_results)
            group_score = 100*statistics.mean(group_results)
        table_dict["Model"].append(display_name)
        table_dict["Text"].append(text_score)
        table_dict["Image"].append(image_score)
        table_dict["Group"].append(group_score)

    return table_dict

def bolden_table_dict(table_dict):
    new_table_dict = table_dict.copy()
    new_text = []
    for item in table_dict["Text"]:
        if item > 25:
            new_text.append("\\textbf{" + f'{item:.2f}' + "}")
        else:
            new_text.append(f'{item:.2f}')
    new_table_dict["Text"] = new_text

    new_image = []
    for item in table_dict["Image"]:
        if item > 25:
            new_image.append("\\textbf{" + f'{item:.2f}' + "}")
        else:
            new_image.append(f'{item:.2f}')
    new_table_dict["Image"] = new_image

    new_group = []
    for item in table_dict["Group"]:
        if item > 16.67:
            new_group.append("\\textbf{" + f'{item:.2f}' + "}")
        else:
            new_group.append(f'{item:.2f}')
    new_table_dict["Group"] = new_group
    return new_table_dict

def get_perplexity_table_dict(models):
    table_dict = {"Model": [], "Corr": [], "P-value": []}
    ordered_perplexities = get_ordered_gpt2_perplexities()
    for filename, display_name, _, _, _ in models:
        results = load_results(filename)
        ordered_model_scores = get_ordered_model_scores_for_perplexity_comparison(results)
        corr, p_value = scipy.stats.pearsonr(ordered_perplexities, ordered_model_scores)
        table_dict["Model"].append(display_name)
        table_dict["Corr"].append(corr)
        table_dict["P-value"].append(p_value)
    return table_dict

def get_caption_length_table_dict(models):
    table_dict = {"Model": [], "Corr": [], "P-value": []}
    ordered_lengths = get_ordered_caption_lengths()
    for filename, display_name, _, _, _ in models:
        results = load_results(filename)
        ordered_model_scores = get_ordered_model_group_correct(results)
        corr, p_value = scipy.stats.pearsonr(ordered_lengths, ordered_model_scores)
        table_dict["Model"].append(display_name)
        table_dict["Corr"].append(corr)
        table_dict["P-value"].append(p_value)
    return table_dict

models = [
    ("statistics/model_scores/human.jsonl", "MTurk Human", None, None, None),
    ("statistics/model_scores/vinvl.jsonl", "VinVL", 1.89, 4.87, "single-stream"),
    ("statistics/model_scores/uniter_large.jsonl", "UNITER$_{large}$", 4.197, 9.583, "single-stream"),
    ("statistics/model_scores/uniter_base.jsonl", "UNITER$_{base}$", 4.197, 9.583, "single-stream"),
    ("statistics/model_scores/villa_large.jsonl", "ViLLA$_{large}$", 4.197, 9.583, "single-stream"),
    ("statistics/model_scores/villa_base.jsonl", "ViLLA$_{base}$", 4.197, 9.583, "single-stream"),
    ("statistics/model_scores/visualbert.jsonl", "VisualBERT$_{base}$", 0.296, 0.517, "single-stream"),
    ("statistics/model_scores/vilt.jsonl", "ViLT (ViT-B/32)", 4.098, 9.854, "single-stream"),
    ("statistics/model_scores/lxmert.jsonl", "LXMERT", 0.18, 9.18, "dual-stream"),
    ("statistics/model_scores/vilbert.jsonl", "ViLBERT$_{base}$", 3.3, 3.3, "dual-stream"),
    ("statistics/model_scores/unit.jsonl", "UniT$_{ITM Finetuned}$", 0.688, 1.91, "dual-stream"),
    ("statistics/model_scores/flava_itm.jsonl", "FLAVA$_{ITM}$", 70, 70, "dual-stream"),
    ("statistics/model_scores/flava_zero_shot.jsonl", "FLAVA$_{Contrastive}$", 70, 70, "dual-stream"),
    ("statistics/model_scores/clip.jsonl", "CLIP (ViT-B/32)", 400, 400, "dual-stream"),
    ("statistics/model_scores/vse_coco_resnet_ft.jsonl", "VSE++$_{COCO}$ (ResNet)", 0.113, 0.565, "rnn"),
    ("statistics/model_scores/vse_coco_vgg_ft.jsonl", "VSE++$_{COCO}$ (VGG)", 0.113, 0.565, "rnn"),
    ("statistics/model_scores/vse_f30k_resnet_ft.jsonl", "VSE++$_{Flickr30k}$ (ResNet)", 0.031, 0.155, "rnn"),
    ("statistics/model_scores/vse_f30k_vgg_ft.jsonl", "VSE++$_{Flickr30k}$ (VGG)", 0.031, 0.155, "rnn"),
    ("statistics/model_scores/vsrn_coco.jsonl", "VSRN$_{COCO}$", 0.113, 0.565, "rnn"),
    ("statistics/model_scores/vsrn_flickr.jsonl", "VSRN$_{Flickr30k}$", 0.031, 0.155, "rnn"),
]

models_without_data_size_outliers = [model for model in models if model[1] not in ("MTurk Human", "CLIP (ViT-B/32)", "FLAVA$_{ITM}$", "FLAVA$_{Contrastive}$")]

print("Aggregate with confidence intervals:")
# Note that the confidence interval code has been updated to use non-parametric bootstrapping. In the paper,
# a t-distribution was assumed. Even though the new code assumes less, the bounds tend to be tighter.
print(tabulate(get_table_dict(models, bootstrap_confidence_intervals=True), tablefmt="latex_raw", floatfmt=".2f"))

print("Aggregate without confidence intervals:")
print(tabulate(bolden_table_dict(get_table_dict(models)), tablefmt="latex_raw", floatfmt=".2f"))

print("Correlations between pretraining data size and scores (excluding CLIP and FLAVA)")
table_dict_no_outliers = get_table_dict(models_without_data_size_outliers)
pretrain_img = [obj[2] for obj in models_without_data_size_outliers]
pretrain_cap = [obj[3] for obj in models_without_data_size_outliers]
pretrain_table_dict = {"Score": [], "Pretraining Type": [], "Corr": [], "P-value": []}
corr, p_value = scipy.stats.pearsonr(table_dict_no_outliers["Text"], pretrain_img)
pretrain_table_dict["Score"].append("Text")
pretrain_table_dict["Pretraining Type"].append("Image")
pretrain_table_dict["Corr"].append(corr)
pretrain_table_dict["P-value"].append(p_value)
corr, p_value = scipy.stats.pearsonr(table_dict_no_outliers["Image"], pretrain_img)
pretrain_table_dict["Score"].append("Image")
pretrain_table_dict["Pretraining Type"].append("Image")
pretrain_table_dict["Corr"].append(corr)
pretrain_table_dict["P-value"].append(p_value)
corr, p_value = scipy.stats.pearsonr(table_dict_no_outliers["Group"], pretrain_img)
pretrain_table_dict["Score"].append("Group")
pretrain_table_dict["Pretraining Type"].append("Image")
pretrain_table_dict["Corr"].append(corr)
pretrain_table_dict["P-value"].append(p_value)
corr, p_value = scipy.stats.pearsonr(table_dict_no_outliers["Text"], pretrain_cap)
pretrain_table_dict["Score"].append("Text")
pretrain_table_dict["Pretraining Type"].append("Caption")
pretrain_table_dict["Corr"].append(corr)
pretrain_table_dict["P-value"].append(p_value)
corr, p_value = scipy.stats.pearsonr(table_dict_no_outliers["Image"], pretrain_cap)
pretrain_table_dict["Score"].append("Image")
pretrain_table_dict["Pretraining Type"].append("Caption")
pretrain_table_dict["Corr"].append(corr)
pretrain_table_dict["P-value"].append(p_value)
corr, p_value = scipy.stats.pearsonr(table_dict_no_outliers["Group"], pretrain_cap)
pretrain_table_dict["Score"].append("Group")
pretrain_table_dict["Pretraining Type"].append("Caption")
pretrain_table_dict["Corr"].append(corr)
pretrain_table_dict["P-value"].append(p_value)
print(tabulate(pretrain_table_dict, tablefmt="latex_raw", floatfmt=".2f"))

print("Linguistic")
linguistic_table_dict = bolden_table_dict(get_table_dict(models, filter_examples(filter_dict={"collapsed_tag": "Object"})))
relation_table_dict = bolden_table_dict(get_table_dict(models, filter_examples(filter_dict={"collapsed_tag": "Relation"})))
linguistic_table_dict["Text2"] = relation_table_dict["Text"]
linguistic_table_dict["Image2"] = relation_table_dict["Image"]
linguistic_table_dict["Group2"] = relation_table_dict["Group"]
both_table_dict = bolden_table_dict(get_table_dict(models, filter_examples(filter_dict={"collapsed_tag": "Both"})))
linguistic_table_dict["Text3"] = both_table_dict["Text"]
linguistic_table_dict["Image3"] = both_table_dict["Image"]
linguistic_table_dict["Group3"] = both_table_dict["Group"]
main_pred1_table_dict = bolden_table_dict(get_table_dict(models, filter_examples(filter_dict={"num_main_preds": 1})))
linguistic_table_dict["Text4"] = main_pred1_table_dict["Text"]
linguistic_table_dict["Image4"] = main_pred1_table_dict["Image"]
linguistic_table_dict["Group4"] = main_pred1_table_dict["Group"]
main_pred2_table_dict = bolden_table_dict(get_table_dict(models, filter_examples(filter_dict={"num_main_preds": 2})))
linguistic_table_dict["Text5"] = main_pred2_table_dict["Text"]
linguistic_table_dict["Image5"] = main_pred2_table_dict["Image"]
linguistic_table_dict["Group5"] = main_pred2_table_dict["Group"]
print(tabulate(linguistic_table_dict, tablefmt="latex_raw", floatfmt=".2f"))

print("Visual")
visual_table_dict = bolden_table_dict(get_table_dict(models, filter_examples(filter_dict={"secondary_tag": "Symbolic"})))
pragmatics_table_dict = bolden_table_dict(get_table_dict(models, filter_examples(filter_dict={"secondary_tag": "Pragmatics"})))
visual_table_dict["Text2"] = pragmatics_table_dict["Text"]
visual_table_dict["Image2"] = pragmatics_table_dict["Image"]
visual_table_dict["Group2"] = pragmatics_table_dict["Group"]
series_table_dict = bolden_table_dict(get_table_dict(models, filter_examples(filter_dict={"secondary_tag": "Series"})))
visual_table_dict["Text3"] = series_table_dict["Text"]
visual_table_dict["Image3"] = series_table_dict["Image"]
visual_table_dict["Group3"] = series_table_dict["Group"]
print(tabulate(visual_table_dict, tablefmt="latex_raw", floatfmt=".2f"))

print("Perplexity and length correlation")
correlation_dict = get_perplexity_table_dict(models)
length_correlation_dict = get_caption_length_table_dict(models)
correlation_dict["Corr2"] = length_correlation_dict["Corr"]
correlation_dict["P-value2"] = length_correlation_dict["P-value"]

print(tabulate(correlation_dict, tablefmt="latex_raw", floatfmt=".2f"))

print("Examples for each fine-grained linguistic tag")
color_to_tag = {"Object": ("\\rowcolor{LightBlue} ", "\\rowcolor{LighterBlue} "), "Relation": ("\\rowcolor{LightGreen} ", "\\rowcolor{LighterGreen} "), "Both": ("\\rowcolor{LightYellow} ", "\\rowcolor{LighterYellow} ")}
tag_table = []
tag_headers = ["Tag", "Sub-Tag", "Example"]
tag_to_sub_tags = {"Object": [], "Relation": [], "Both": []}
for sub_tag, tag, example in tags_with_examples:
    tag_to_sub_tags[tag].append((sub_tag, example))
for tag in ["Object", "Relation", "Both"]:
    tag_index = int(len(set(tag_to_sub_tags[tag]))/2) - 1
    index = 0
    for sub_tag, example in set(tag_to_sub_tags[tag]):
        if index == tag_index and index != 0:
            first = color_to_tag[tag][index % 2] + tag
        elif index == 0 and index == tag_index:
            first = color_to_tag[tag][index % 2] + "\midrule " + tag
        elif index == 0 and index != tag_index:
            first = color_to_tag[tag][index % 2] + "\midrule"
        else:
            first = color_to_tag[tag][index % 2]
        index += 1
        tag_table.append([first, sub_tag, example])
print(tabulate(tag_table, tag_headers, tablefmt="latex_raw"))
print("Number of fine-grained linguistic tags: ", len(tags_with_examples))

print("Plotting model performance figures...")
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
single_stream_indices = [item[0] for item in enumerate(models_without_data_size_outliers) if item[1][4] == "single-stream"]
dual_stream_indices = [item[0] for item in enumerate(models_without_data_size_outliers) if item[1][4] == "dual-stream"]
rnn_indices = [item[0] for item in enumerate(models_without_data_size_outliers) if item[1][4] == "rnn"]


for score_type, color in [("Text", "blue"), ("Image", "red"), ("Group", "black")]:
    single_img = ax1.scatter([pretrain_img[i] for i in single_stream_indices], [table_dict_no_outliers[score_type][i] for i in single_stream_indices], c=color, marker="o")
    dual_img = ax1.scatter([pretrain_img[i] for i in dual_stream_indices], [table_dict_no_outliers[score_type][i] for i in dual_stream_indices], c=color, marker="s")
    rnn_img = ax1.scatter([pretrain_img[i] for i in rnn_indices], [table_dict_no_outliers[score_type][i] for i in rnn_indices], c=color, marker="d")

    single_cap = ax2.scatter([pretrain_cap[i] for i in single_stream_indices], [table_dict_no_outliers[score_type][i] for i in single_stream_indices], c=color, marker="o")
    dual_cap = ax2.scatter([pretrain_cap[i] for i in dual_stream_indices], [table_dict_no_outliers[score_type][i] for i in dual_stream_indices], c=color, marker="s")
    rnn_cap = ax2.scatter([pretrain_cap[i] for i in rnn_indices], [table_dict_no_outliers[score_type][i] for i in rnn_indices], c=color, marker="d")

ybox_ax2_1 = TextArea("Text", textprops=dict(color="b", size=15,rotation=90,ha='left',va='bottom'))
ybox_ax2_2 = TextArea("Image", textprops=dict(color="r", size=15,rotation=90,ha='left',va='bottom'))
ybox_ax2_3 = TextArea("Group", textprops=dict(color="k", size=15,rotation=90,ha='left',va='bottom'))
ybox_ax1 = VPacker(children=[ybox_ax2_1, ybox_ax2_2, ybox_ax2_3],align="bottom", pad=0, sep=5)
anchored_ybox1 = AnchoredOffsetbox(loc=8, child=ybox_ax1, pad=0., frameon=False, bbox_to_anchor=(-0.08, 0.25),
                                   bbox_transform=ax1.transAxes, borderpad=0.)
ax1.add_artist(anchored_ybox1)
ax1.set_xlabel("# Pretraining Images", fontsize=14)
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Single-Str. Transformer', markerfacecolor='none', markeredgecolor='black'),
                   Line2D([0], [0], marker='s', color='w', label='Dual-Str. Transformer', markerfacecolor='none', markeredgecolor='black'),
                   Line2D([0], [0], marker='d', color='w', label='Dual-Str. RNN', markerfacecolor='none', markeredgecolor='black')]
ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, prop={'size': 10})

ybox_ax2_1 = TextArea("Text", textprops=dict(color="b", size=15,rotation=90,ha='left',va='bottom'))
ybox_ax2_2 = TextArea("Image", textprops=dict(color="r", size=15,rotation=90,ha='left',va='bottom'))
ybox_ax2_3 = TextArea("Group", textprops=dict(color="k", size=15,rotation=90,ha='left',va='bottom'))
ybox_ax2 = VPacker(children=[ybox_ax2_1, ybox_ax2_2, ybox_ax2_3],align="bottom", pad=0, sep=5)
anchored_ybox2 = AnchoredOffsetbox(loc=8, child=ybox_ax2, pad=0., frameon=False, bbox_to_anchor=(-0.08, 0.25),
                                  bbox_transform=ax2.transAxes, borderpad=0.)
ax2.add_artist(anchored_ybox2)
ax2.set_xlabel("# Pretraining Captions", fontsize=14)
ax2.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, prop={'size': 10})

ax1.axhline(25, linestyle=':', c="black")
ax1.axhline(16.67, linestyle='--', c="black")
ax1.text(3.2, 25.2, 'Random (Img, Txt)')
ax1.text(3.34, 16.87, 'Random (Group)')
fig1.subplots_adjust(bottom=0.2)

ax2.axhline(25, linestyle=':', c="black")
ax2.axhline(16.67, linestyle='--', c="black")
ax2.text(7.5, 25.2, 'Random (Img, Txt)')
ax2.text(7.82, 16.87, 'Random (Group)')
fig2.subplots_adjust(bottom=0.2)

plt.show()


