import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models.clip import clip
from models.blip.blip_retrieval import BLIP_Retrieval
from models.blip.blip import blip_decoder, BLIP_Decoder
from models.tokenization_bert import BertTokenizer
from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed

from transformers import BertForMaskedLM
import utils
from attacks.step import LinfStep, L2Step
from dataset import pair_dataset
from dataset.coco_karpathy_dataset import coco_karpathy_train, coco_karpathy_caption_eval, coco_karpathy_retrieval_eval
from PIL import Image
from torchvision import transforms


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    model.eval() 
    
    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10

    # load uap
    uap_root = os.path.join(args.per_dir, 'uap.pt')
    uap_noise = torch.load(uap_root)
    
    uap_noise = uap_noise.to(device)

    result = []
    for images, image_id in metric_logger.log_every(data_loader, print_freq, header):         
        
        images = images.to(device)       
        images = images + uap_noise
        images = torch.clamp(images, 0, 1)
        
        t_adv_img_list = []
        for itm in images:
            t_adv_img_list.append(args.test_transform(itm))
        t_adv_imgs = torch.stack(t_adv_img_list, 0).to(device)  
            
        images = images_normalize(t_adv_imgs)
    
        captions = model.generate(images, sample=True, num_beams=config['num_beams'], max_length=config['max_length'], 
                                  min_length=config['min_length'])  #sample=False
        
        for caption, img_id in zip(captions, image_id):
            result.append({"image_id": img_id.item(), "caption": caption})
  
    return result


def load_model(model_name, model_ckpt, text_encoder, device):
    tokenizer = BertTokenizer.from_pretrained(text_encoder)
    ref_model = BertForMaskedLM.from_pretrained(text_encoder)    
    if model_name in ['ALBEF', 'TCL']:
        model = ALBEF(config=config, text_encoder=text_encoder, tokenizer=tokenizer)
        checkpoint = torch.load(model_ckpt, map_location='cpu')
    elif model_name in ['BLIP']:
        model = BLIP_Decoder(config=config)
        checkpoint = torch.load(model_ckpt, map_location='cpu')
        
    else:
        model, preprocess = clip.load(args.target_image_encoder, device=device)
        model.set_tokenizer(tokenizer)
        return model, ref_model, tokenizer
    
    try:
        state_dict = checkpoint['model']
    except:
        state_dict = checkpoint

    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)    
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    
    return model, ref_model, tokenizer


def main(args, config):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU {device} - {torch.cuda.get_device_name(device)}")
    else:
        print("CUDA is not available. No GPU devices found.")

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    #### Model ####
    print("Creating target model")

    
    
    args.target_dataset_root = './Datasets/MSCOCO/'

    model, ref_model, tokenizer = load_model(args.target_model, args.target_ckpt, args.target_text_encoder, device)

    model = model.to(device)
    ref_model = ref_model.to(device)

    #### Dataset ####
    print("Creating dataset")
    args.data_shape=(3,config['image_res'],config['image_res'])
    
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
    
    print("Creating captioning dataset")
    test_dataset = coco_karpathy_caption_eval(s_test_transform, args.target_dataset_root, config['ann_root'], 'test')   
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size_test'], num_workers=4)
    
    print("Start eval")
    start_time = time.time()

    result = evaluate(model, test_loader, device, config) 
    print(result)

    test_result_file = save_result(result, args.output_dir, 'test_epoch%d'%100, remove_duplicate='image_id')  
    coco_test = coco_caption_eval(config['coco_gt_root'], test_result_file,'test')
    
    
    log_stats = {**{f'test_{k}': v for k, v in coco_test.eval.items()},
                             }
    with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
        f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluate time {}'.format(total_time_str))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_coco.yaml')

    #source model
    parser.add_argument('--source_model', default='CLIP', choices=['ALBEF', 'CLIP', 'CLIP', 'BLIP'])
    parser.add_argument('--source_text_encoder', default='bert-base-uncased')
    parser.add_argument('--source_image_encoder', default='ViT-B/16')
    parser.add_argument('--source_dataset', default='flickr30')
    parser.add_argument('--source_ckpt', default=None, type=str)
    
    #target model
    parser.add_argument('--target_model', default='BLIP')
    parser.add_argument('--target_text_encoder', default='bert-base-uncased', choices=['ALBEF', 'CLIP', 'TCL', 'BLIP'])
    parser.add_argument('--target_image_encoder', default='ViT-B/16', choices=['ViT-L/14', 'ViT-B/16', 'ViT-B/32', 'RN50', 'RN101'])
    parser.add_argument('--target_dataset', default='mscoco', choices=['mscoco', 'flickr30', 'CLIP', 'BLIP'])
    parser.add_argument('--target_ckpt', default='./checkpoint/BLIP/model_base_caption_capfilt_large.pth', type=str)
        
    parser.add_argument('--method', default='your attack method name')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--cls', default=False)
        
    args = parser.parse_args()

    args.cls = False
    yaml = yaml.YAML(typ="safe", pure=True)
    config = yaml.load(open(args.config, 'r'))

    #output path
    args.output_dir = os.path.join('output', args.method, 'uap_cross', str(args.source_model) +'_'+ str(args.source_image_encoder), str(args.target_model) +'_'+ str(args.target_image_encoder), str(args.source_dataset) +'_'+ str(args.target_dataset), str(config['epsilon']))

    #uap path        
    args.per_dir = os.path.join('output', args.method, 'uap', str(args.source_model), str(args.source_image_encoder), str(args.source_dataset), str(config['epsilon']), str(config['batch_size_test']))
        
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    yaml.dump(config, open(os.path.join(args.per_dir, 'config.yaml'), 'w'))

    main(args, config)

