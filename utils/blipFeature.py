from lavis.models import load_model_and_preprocess
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"

class blipImageDataset(Dataset):
    def __init__(self, data, transform=None):
        super(Dataset, self).__init__()
        self.data = data
        self.transform = transform
        self.unloader = transforms.ToPILImage()

    def __getitem__(self, index):
        x = self.data[index]
        if type(x) == torch.Tensor:
            x = self.unloader(x)
        elif type(x) == str:
            x = Image.open(x).convert('RGB')
        elif type(x) == np.ndarray:
             x = Image.fromarray(x)
        if self.transform:
            x = self.transform(x)
        return x
    
    def __len__(self):
        return len(self.data)

class blipImageDataset_with_label(Dataset):
    def __init__(self, data, label, transform=None, label_transform=None):
        super(Dataset, self).__init__()
        self.data = data
        self.transform = transform
        self.label_transform = label_transform
        self.unloader = transforms.ToPILImage()
        self.labels = label

    def __getitem__(self, index):
        x = self.data[index]
        label = self.labels[index]
        if type(x) == torch.Tensor:
            x = self.unloader(x)
        elif type(x) == str:
            x = Image.open(x)
        elif type(x) == np.ndarray:
             x = Image.fromarray(x)
        if self.transform:
            x = self.transform(x)
        if self.label_transform:
            label = self.label_transform(label)
        return x, label
    
    def __len__(self):
        return len(self.data)
    
def get_image_features(imgs):
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)
    data_loader = DataLoader(blipImageDataset(imgs,vis_processors["eval"]), pin_memory=True, 
                shuffle=False, batch_size=50, num_workers=2, drop_last=False)
    
    all_image_features = []

    with torch.no_grad():
        for imgs in tqdm.tqdm(data_loader):
            imgs = imgs.to(device)
            sample = {"image": imgs}
            features = model.extract_features(sample, mode="image")
            
            all_image_features.append(features.image_embeds_proj[:,0,:])

        return torch.cat(all_image_features,dim=0)

def get_text_features(texts):
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)
    text_input = [txt_processors["eval"](text) for text in texts]
    with torch.no_grad():
        features_text = model.extract_features({"text_input": text_input}, mode="text")
    return features_text.text_embeds_proj[:,0,:]

def get_captions(imgs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
    data_loader = DataLoader(blipImageDataset(imgs,vis_processors["eval"]), pin_memory=True, 
                shuffle=False, batch_size=100, num_workers=2, drop_last=False)
    print('Get captions..')

    with torch.no_grad():
        captions = []
        for imgs in tqdm.tqdm(data_loader):
            imgs = imgs.to(device)
            caption = model.generate({"image": imgs})
            captions += caption
        return captions

def get_captions_blip2_hugging_face(imgs):
    from PIL import Image
    import requests
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    )
    model.to(device)
    dataset = blipImageDataset(imgs)
    captions = []
    for i in tqdm.tqdm(range(len(dataset)//100)):
        img = [dataset[i * 100 + j] for j in range(100)]
        img = processor(images=img, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**img)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        captions += generated_text
    return captions

def get_img_text_similarity(imgs, texts):
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)
    
    data_loader = DataLoader(blipImageDataset(imgs,vis_processors["eval"]), pin_memory=True, 
                shuffle=False, batch_size=128, num_workers=2, drop_last=False)
    
    text_input = [txt_processors["eval"](text) for text in texts]
    all_logits_per_image = []
    with torch.no_grad():
        features_text = model.extract_features({"text_input": text_input}, mode="text")
        for imgs in tqdm.tqdm(data_loader):
            imgs = imgs.to(device)
            sample = {"image": imgs}
            features_image = model.extract_features(sample, mode="image")
            similarity = features_image.image_embeds_proj[:,0,:] @ features_text.text_embeds_proj[:,0,:].t()
            all_logits_per_image.append(similarity)
    return torch.cat(all_logits_per_image,dim=0)

def get_img_text_similarity_with_features(imgs, text):
    return imgs @ text.t()

def question_answering(imgs, question):
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)
    data_loader = DataLoader(blipImageDataset(imgs,vis_processors["eval"]), pin_memory=True, 
                shuffle=False, batch_size=128, num_workers=2, drop_last=False)
    question = txt_processors["eval"](question)
    all_answer_per_image = []
    with torch.no_grad():
        for imgs in tqdm.tqdm(data_loader):
            imgs = imgs.to(device)
            text_input = [question for i in range(len(imgs))]
            answer = model.predict_answers(samples={"image": imgs, "text_input": text_input}, inference_method="generate")
            all_answer_per_image += answer
    return all_answer_per_image

def question_answering_list(imgs, question):
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)
    data_loader = DataLoader(blipImageDataset_with_label(imgs,question,vis_processors["eval"],txt_processors["eval"]), pin_memory=True, 
                shuffle=False, batch_size=128, num_workers=2, drop_last=False)
    all_answer_per_image = []
    with torch.no_grad():
        for imgs, question in tqdm.tqdm(data_loader):
            imgs = imgs.to(device)
            answer = model.predict_answers(samples={"image": imgs, "text_input": question}, inference_method="generate")
            all_answer_per_image += answer
    return all_answer_per_image