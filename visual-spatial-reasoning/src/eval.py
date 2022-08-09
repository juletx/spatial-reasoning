import os
import argparse
from tqdm.auto import tqdm
import torch
from transformers import BertTokenizer, \
        VisualBertForVisualReasoning, LxmertForPreTraining, LxmertTokenizer
from lxmert_for_classification import LxmertForBinaryClassification
from data import ImageTextClassificationDataset
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../../BLIP'))

def evaluate(data_loader, model, model_type="visualbert"):
    model.cuda()
    model.eval()

    correct, total, all_true = 0, 0, 0
    preds = []
    
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        if model_type == "visualbert":
            batch_cap, batch_img, y = data
            batch_inputs = {}
            for k,v in batch_cap.items():
                batch_inputs[k] = v.cuda()
            img_attention_mask = torch.ones(batch_img.shape[:-1], dtype=torch.long)
            img_token_type_ids = torch.ones(batch_img.shape[:-1], dtype=torch.long)
            batch_inputs.update({
                "visual_embeds": batch_img.cuda(),
                "visual_token_type_ids": img_token_type_ids.cuda(),
                "visual_attention_mask": img_attention_mask.cuda(),
                })

        elif  model_type == "lxmert":
            batch_cap, batch_box, batch_img, y = data
            batch_inputs = {}
            for k,v in batch_cap.items():
                batch_inputs[k] = v.cuda()
            batch_inputs.update({
                "visual_feats": batch_img.cuda(),
                "visual_pos": batch_box.cuda(),
                })
        elif model_type in ("vilt", "vilt_nlvr2"):
            input_ids, pixel_values, y = data
        elif model_type == "blip_nlvr2":
            images, text, y = data
        y = y.cuda()
        with torch.no_grad():
            if model_type in ["visualbert", "lxmert"]:
                outputs = model(**batch_inputs, labels=y)
            elif model_type in ("vilt", "vilt_nlvr2"):
                batch_cap = input_ids.cuda()
                batch_img = pixel_values.cuda()
                outputs = model(input_ids=batch_cap, 
                        pixel_values=batch_img)
                #logits = outputs.logits
                #idx = logits.argmax(-1).item()
                #model.config.id2label[idx]
            elif model_type == "blip_nlvr2":
                batch_cap = text
                batch_img = images.cuda()
                outputs = model(batch_img, batch_cap, targets=y, train=False)  

        if model_type == "blip_nlvr2":
            scores = outputs
        else:
            scores = outputs.logits
        preds_current = torch.argmax(scores, dim=1)
        correct += sum(y == preds_current)
        preds += preds_current.cpu().numpy().tolist()
        total += len(batch_cap)
        all_true += sum(y)

        # print errors
        #print (y != torch.argmax(scores, dim=1))

    # TODO: save also predictions
    return correct / total, total, all_true, preds
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='visualbert')
    parser.add_argument('--img_feature_path', type=str, required=True)
    parser.add_argument('--test_json_path', type=str, required=True)
    parser.add_argument('--output_preds', action='store_true')

    args = parser.parse_args()

    model_type = args.model_type
    # load model
    if model_type == "visualbert":
        model = VisualBertForVisualReasoning.from_pretrained(args.checkpoint_path)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    elif model_type == "lxmert":
        model = LxmertForPreTraining.from_pretrained(args.checkpoint_path)
        model = LxmertForBinaryClassification(model)
        tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased") 

    elif model_type in ("vilt", "vilt_nlvr2"):
        from transformers import ViltProcessor, ViltForImagesAndTextClassification
        processor = ViltProcessor.from_pretrained(args.checkpoint_path)
        model = ViltForImagesAndTextClassification.from_pretrained(args.checkpoint_path)

    elif model_type == "blip_nlvr2":
        from models.blip_nlvr import blip_nlvr
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode 
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
        model = blip_nlvr(pretrained=args.checkpoint_path, image_size=384, vit='base')
    
    # load data
    def collate_fn_batch_visualbert(batch):
        captions, img_features, labels = zip(*batch)
        toks = tokenizer.batch_encode_plus(
            list(captions), 
            max_length=32, 
            padding="max_length", 
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt")
        img_features = torch.stack(img_features, dim=0)
        labels = torch.tensor(labels)
        return toks, img_features, labels
    
    def collate_fn_batch_lxmert(batch):
        captions, boxes, img_features, labels = zip(*batch)
        toks = tokenizer.batch_encode_plus(
            list(captions), 
            max_length=32, 
            padding="max_length", 
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt")
        img_features = torch.stack(img_features, dim=0)
        boxes = torch.stack(boxes)
        labels = torch.tensor(labels)
        return toks, boxes, img_features, labels

    def collate_fn_batch_vilt(batch):
        imgs, captions, labels = zip(*batch)
        inputs = processor(list(imgs), list(captions), return_tensors="pt", padding=True, truncation=True)
        labels = torch.tensor(labels)
        return inputs.input_ids, inputs.pixel_values.unsqueeze(1), labels

    def collate_fn_batch_vilt_nlvr2(batch):
        imgs, captions, labels = zip(*batch)
        inputs = processor(list(imgs), list(captions), return_tensors="pt", padding=True, truncation=True)
        labels = torch.tensor(labels)
        return inputs.input_ids, inputs.pixel_values.unsqueeze(1).repeat(1, 2, 1, 1, 1), labels

    def collate_fn_batch_blip_nlvr2(batch):
        imgs, captions, labels = zip(*batch)
        imgs = [transform(img) for img in list(imgs)]
        imgs = torch.stack(imgs)
        images = torch.cat([imgs, imgs], dim=0)
        labels = torch.tensor(labels)
        return images, captions, labels

    img_feature_path = args.img_feature_path
    json_path = args.test_json_path
    if model_type in ["visualbert", "lxmert", "blip_nlvr2"]:
        dataset = ImageTextClassificationDataset(img_feature_path, json_path, model_type=model_type)
    elif model_type in ("vilt", "vilt_nlvr2"):
        dataset = ImageTextClassificationDataset(img_feature_path, json_path, model_type=model_type, vilt_processor=processor)
    if model_type == "visualbert":
        collate_fn_batch = collate_fn_batch_visualbert
    elif model_type == "lxmert":
        collate_fn_batch = collate_fn_batch_lxmert
    elif model_type == "vilt":
        collate_fn_batch = collate_fn_batch_vilt
    elif model_type == "vilt_nlvr2":
        collate_fn_batch = collate_fn_batch_vilt_nlvr2
    elif model_type == "blip_nlvr2":
        collate_fn_batch = collate_fn_batch_blip_nlvr2

    test_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn = collate_fn_batch,
        batch_size=16,
        shuffle=False,
        num_workers=16,)
    acc, total, all_true, preds = evaluate(test_loader, model, model_type=model_type)
    print (f"total example: {total}, # true example: {all_true}, acccuracy: {acc}")

    # save preds
    if args.output_preds:
        with open(os.path.join(args.result_path), "w") as f:
            for i in range(len(preds)):
                f.write(str(preds[i])+"\n")
        
