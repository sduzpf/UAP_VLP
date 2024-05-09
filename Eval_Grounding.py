import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM

from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import grounding_dataset
from dataset.utils import grounding_eval
from refTools.refer_python3 import REFER
from torchvision import transforms
from PIL import Image

def val(model, ref_model, data_loader, tokenizer, device, block_num):
    model.eval()
    ref_model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print_freq = 50

    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))    

    # load uap
    uap_root = os.path.join(args.per_dir, 'uap.pt')
    uap_noise = torch.load(uap_root)
    uap_noise = uap_noise.to(device)
     
    result = []
    for images, text, ref_ids in metric_logger.log_every(data_loader, print_freq, header):
        
        images = images.to(device)
        images = images + uap_noise
        images = torch.clamp(images, 0, 1)
        
        t_adv_img_list = []
        for itm in images:
            t_adv_img_list.append(args.test_transform(itm))
        t_adv_imgs = torch.stack(t_adv_img_list, 0).to(device)
        images = images_normalize(t_adv_imgs)
        
        text_input = tokenizer(text, padding='longest', return_tensors="pt").to(device)

        model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True

        image_embeds = model.visual_encoder(images)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(images.device)
        output = model.text_encoder(text_input.input_ids,
                                attention_mask = text_input.attention_mask,
                                encoder_hidden_states = image_embeds,
                                encoder_attention_mask = image_atts,
                                return_dict = True,
                               )

        vl_embeddings = output.last_hidden_state[:,0,:]
        vl_output = model.itm_head(vl_embeddings)
        loss = vl_output[:,1].sum()

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            mask = text_input.attention_mask.view(text_input.attention_mask.size(0),1,-1,1,1)

            grads = model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attn_gradients().detach()
            cams = model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attention_map().detach()

            cams = cams[:, :, :, 1:].reshape(images.size(0), 12, -1, 24, 24) * mask
            grads = grads[:, :, :, 1:].clamp(min=0).reshape(images.size(0), 12, -1, 24, 24) * mask

            gradcam = cams * grads
            gradcam = gradcam.mean(1).mean(1)


        for r_id, cam in zip(ref_ids, gradcam):
            result.append({'ref_id':r_id.item(), 'pred':cam})

        model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = False

    return result


def load_model(model_name, model_ckpt, text_encoder, device):
    tokenizer = BertTokenizer.from_pretrained(text_encoder)
    ref_model = BertForMaskedLM.from_pretrained(text_encoder)    
    if model_name in ['ALBEF', 'TCL']:
        model = ALBEF(config=config, text_encoder=text_encoder, tokenizer=tokenizer)
        checkpoint = torch.load(model_ckpt, map_location='cpu')
    else:
        model, preprocess = clip.load(args.target_image_encoder, device=device)
        model.set_tokenizer(tokenizer)
        return model, ref_model, tokenizer
    
    try:
        state_dict = checkpoint['model']
    except:
        state_dict = checkpoint

    if model_name == 'TCL':
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 

    for key in list(state_dict.keys()):
        if 'bert' in key:
            encoder_key = key.replace('bert.', '')
            state_dict[encoder_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    
    return model, ref_model, tokenizer

def main(args, config):

    device = args.gpu[0]

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    s_test_transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),       
    ])
    args.test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ]) 
    
    
    grd_test_dataset = grounding_dataset(config['test_file'], s_test_transform, config['image_root'], mode='test')

    test_loader = DataLoader(grd_test_dataset, batch_size=config['batch_size'], num_workers=4)
       
    tokenizer = BertTokenizer.from_pretrained(args.target_text_encoder)
        
    ## refcoco evaluation tools
    refer = REFER(config['refcoco_data'], 'refcoco+', 'unc')
    dets = json.load(open(config['det_file'],'r'))
    cocos = json.load(open(config['coco_file'],'r'))    

    #### Model #### 
    print("Creating target model")

    model, ref_model, tokenizer = load_model(args.target_model, args.target_ckpt, args.target_text_encoder, device)
    model = model.to(device)
    ref_model = ref_model.to(device)
    
    print("Start Evaluating")
    start_time = time.time()    

    result = val(model, ref_model, test_loader, tokenizer, device, args.block_num)
    grounding_acc = grounding_eval(result, dets, cocos, refer, alpha=0.5, mask_size=24)

    log_stats = {**{f'{k}': v for k, v in grounding_acc.items()},
                 'cls': args.cls, 'eps': config['epsilon'], 'iters':config['num_iters']}

    with open(os.path.join(args.output_dir, "log.txt"),"a+") as f:
        f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluating time {}'.format(total_time_str))

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Grounding.yaml')

    #source model
    parser.add_argument('--source_model', default='CLIP')
    parser.add_argument('--source_text_encoder', default='bert-base-uncased')
    parser.add_argument('--source_image_encoder', default='ViT-B/16')
    parser.add_argument('--source_dataset', default='flickr30')
    parser.add_argument('--source_ckpt', default=None, type=str)
    
    #target model
    parser.add_argument('--target_model', default='ALBEF', choices=['ALBEF'])
    parser.add_argument('--target_text_encoder', default='bert-base-uncased')
    parser.add_argument('--target_image_encoder', default='ViT-B/16', choices=['ViT-B/16'])
    parser.add_argument('--target_dataset', default='refcoco')
    parser.add_argument('--target_ckpt', default='./ALBEF/checkpoint/refcoco.pth', type=str)
    
    parser.add_argument('--method', default='your attack method name')
    parser.add_argument('--block_num', default=8, type=int)
    parser.add_argument('--gpu', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--cls', default=False)

    args = parser.parse_args()

    yaml = yaml.YAML(typ="safe", pure=True)
    config = yaml.load(open(args.config, 'r'))

    #output path
    args.output_dir = os.path.join('output', args.method, 'uap_Grounding', str(args.source_model) +'_'+ str(args.source_image_encoder), str(args.target_model) +'_'+ str(args.target_image_encoder), str(args.source_dataset) +'_'+ str(args.target_dataset), str(config['epsilon']))
    
    #uap path    
    args.per_dir = os.path.join('output', args.method, 'uap', str(args.source_model), str(args.source_image_encoder), str(args.source_dataset), str(config['epsilon']), str(config['batch_size_test']))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
