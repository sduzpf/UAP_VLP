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
from models.tokenization_bert import BertTokenizer
from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed

from transformers import BertForMaskedLM
import utils
from attacks.step import LinfStep, L2Step
from dataset import pair_dataset
from PIL import Image
from torchvision import transforms


def retrieval_eval(model, ref_model, data_loader, tokenizer, device, args, config):
    # test
    model.float()
    model.eval()
    ref_model.eval()

    print('Prepare memory')
    num_text = len(data_loader.dataset.text)
    num_image = len(data_loader.dataset.ann)

    image_feats = torch.zeros(num_image, config['embed_dim'])
    image_embeds = torch.zeros(num_image, 577, 768)
    text_feats = torch.zeros(num_text, config['embed_dim'])
    text_embeds = torch.zeros(num_text, 30, 768)
    text_atts = torch.zeros(num_text, 30).long()
    
    print('Computing features for evaluation adv...')
    start_time = time.time()

    local_transform = transforms.Resize(config['image_res'])
    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    print('Load perturbations')

    # load uap
    uap_root = os.path.join(args.per_dir, 'uap.pt')
    uap_noise = torch.load(uap_root)
    uap_noise = uap_noise.to(device)


    print('Compute similarity')
    for images, texts, texts_ids in data_loader:
        texts_input = tokenizer(texts, padding='max_length', truncation=True, max_length=30,
                                return_tensors="pt").to(device)
        images_ids = [data_loader.dataset.txt2img[i.item()] for i in texts_ids]
        images = images.to(device)
        with torch.no_grad():
            images = images + uap_noise
            images = torch.clamp(images, 0, 1)
            t_adv_img_list = []
            for itm in images:
                t_adv_img_list.append(args.test_transform(itm))
            t_adv_imgs = torch.stack(t_adv_img_list, 0).to(device)
                
            images = images_normalize(t_adv_imgs)
            output_img = model.inference_image(images)
            output_txt = model.inference_text(texts_input)
            image_feats[images_ids] = output_img['image_feat'].cpu().detach()
            image_embeds[images_ids] = output_img['image_embed'].cpu().detach()
            text_feats[texts_ids] = output_txt['text_feat'].cpu().detach()
            text_embeds[texts_ids] = output_txt['text_embed'].cpu().detach()
            text_atts[texts_ids] = texts_input.attention_mask.cpu().detach()
    
    score_matrix_i2t, score_matrix_t2i = retrieval_score(model, image_feats, image_embeds, text_feats,
                                                         text_embeds, text_atts, num_image, num_text, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy(), uap_noise


@torch.no_grad()
def retrieval_score(model, image_feats, image_embeds, text_feats, text_embeds, text_atts, num_image, num_text, device=None):
    if device is None:
        device = image_embeds.device

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation Direction Similarity With Bert Attack:'

    sims_matrix = image_feats @ text_feats.t()
    score_matrix_i2t = torch.full((num_image, num_text), -100.0).to(device)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_embeds[i].repeat(config['k_test'], 1, 1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds=text_embeds[topk_idx].to(device),
                                    attention_mask=text_atts[topk_idx].to(device),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    mode='fusion'
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[i, topk_idx] = score

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((num_text, num_image), -100.0).to(device)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_embeds[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds=text_embeds[i].repeat(config['k_test'], 1, 1).to(device),
                                    attention_mask=text_atts[i].repeat(config['k_test'], 1).to(device),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    mode='fusion'
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[i, topk_idx] = score

    return score_matrix_i2t, score_matrix_t2i



@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, img2txt, txt2img):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    #ASR
    after_attack_tr1 = np.where(ranks < 1)[0]
    after_attack_tr5 = np.where(ranks < 5)[0]
    after_attack_tr10 = np.where(ranks < 10)[0]
    
    original_rank_index_path = args.original_rank_index_path
    origin_tr1 = np.load(f'{original_rank_index_path}/{args.target_model}_tr1_rank_index.npy')
    origin_tr5 = np.load(f'{original_rank_index_path}/{args.target_model}_tr5_rank_index.npy')
    origin_tr10 = np.load(f'{original_rank_index_path}/{args.target_model}_tr10_rank_index.npy')

    asr_tr1 = round(100.0 * len(np.setdiff1d(origin_tr1, after_attack_tr1)) / len(origin_tr1), 2)
    asr_tr5 = round(100.0 * len(np.setdiff1d(origin_tr5, after_attack_tr5)) / len(origin_tr5), 2)
    asr_tr10 = round(100.0 * len(np.setdiff1d(origin_tr10, after_attack_tr10)) / len(origin_tr10), 2)


    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2


    #ASR
    after_attack_ir1 = np.where(ranks < 1)[0]
    after_attack_ir5 = np.where(ranks < 5)[0]
    after_attack_ir10 = np.where(ranks < 10)[0]
    
    origin_ir1 = np.load(f'{original_rank_index_path}/{args.target_model}_ir1_rank_index.npy')
    origin_ir5 = np.load(f'{original_rank_index_path}/{args.target_model}_ir5_rank_index.npy')
    origin_ir10 = np.load(f'{original_rank_index_path}/{args.target_model}_ir10_rank_index.npy')

    asr_ir1 = round(100.0 * len(np.setdiff1d(origin_ir1, after_attack_ir1)) / len(origin_ir1), 2) 
    asr_ir5 = round(100.0 * len(np.setdiff1d(origin_ir5, after_attack_ir5)) / len(origin_ir5), 2) 
    asr_ir10 = round(100.0 * len(np.setdiff1d(origin_ir10, after_attack_ir10)) / len(origin_ir10), 2)

    
    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
                   
                   
    ASR_result = {'txt_r1_ASR (txt_r1)': f'{asr_tr1}({tr1})',
                   'txt_r5_ASR (txt_r5)': f'{asr_tr5}({tr5})',
                   'txt_r10_ASR (txt_r10)': f'{asr_tr10}({tr10})',
                   'img_r1_ASR (img_r1)': f'{asr_ir1}({ir1})',
                   'img_r5_ASR (img_r5)': f'{asr_ir5}({ir5})',
                   'img_r10_ASR (img_r10)': f'{asr_ir10}({ir10})'}               
    return eval_result, ASR_result



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

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU {device} - {torch.cuda.get_device_name(device)}")
    else:
        print("CUDA is not available. No GPU devices found.")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Model ####
    print("Creating target model")

    # target model and dataset path
    if args.dataset == 'flickr30':
        if args.model == 'ALBEF':
            args.target_ckpt = './checkpoint/ALBEF/flickr30k.pth'
        
        elif args.model == 'TCL':
            args.target_ckpt = './checkpoint/TCL/checkpoint_retrieval_flickr_finetune.pth'
        args.original_rank_index_path = './std_eval_idx/flickr30k'
        args.dataset_root = './Datasets/flickr30k_images/'
        args.test_file= './Datasets/data/flickr30k_test.json'

    elif args.dataset == 'mscoco':
        if args.model == 'ALBEF':
            args.target_ckpt = './checkpoint/ALBEF/mscoco.pth'
        
        elif args.model == 'TCL':
            args.target_ckpt = './checkpoint/TCL/checkpoint_retrieval_coco_finetune.pth'
        args.original_rank_index_path = './std_eval_idx/mscoco'
        args.dataset_root = './Datasets/MSCOCO/test2014/val2014/'
        args.test_file= './Datasets/data/coco_test.json'

    model, ref_model, tokenizer = load_model(args.target_model, args.target_ckpt, args.target_text_encoder, device)

    model = model.to(device)
    ref_model = ref_model.to(device)

    #### Dataset ####
    print("Creating dataset")
    args.data_shape=(3,config['image_res'],config['image_res'])
    if args.source_model in ['ALBEF', 'TCL']:
        s_test_transform = transforms.Compose([
          transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
          transforms.ToTensor(),       
        ])
    else:     
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
    
    
    test_dataset = pair_dataset(args.test_file, s_test_transform, args.target_dataset_root)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size_test'], num_workers=4)
    
    print("Start eval")
    start_time = time.time()

    score_i2t, score_t2i, uap_noise = retrieval_eval(model, ref_model, test_loader, tokenizer, device, args, config)

    result, ASR_result = itm_eval(score_i2t, score_t2i, test_dataset.img2txt, test_dataset.txt2img)
    print(result)
    print(ASR_result)
    
    log_stats = {**{f'test_{k}': v for k, v in result.items()},
                 'eval type': args.adv, 'cls':args.cls, 'eps': config['epsilon'], 'iters':config['num_iters'], 'alpha': args.alpha}
    log_stats_asr = {**{f'test_{k}': v for k, v in ASR_result.items()},
                 'eval type': args.adv, 'cls':args.cls, 'eps': config['epsilon'], 'iters':config['num_iters'], 'alpha': args.alpha}
    with open(os.path.join(args.output_dir, str(config['batch_size_test'])+ "_log_CLIP.txt"), "a+") as f:
        f.write(json.dumps(log_stats) + "\n")
        f.write(json.dumps(log_stats_asr) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluate time {}'.format(total_time_str))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval_flickr_coco.yaml')

    #source model
    parser.add_argument('--source_model', default='TCL', choices=['ALBEF', 'CLIP', 'TCL', 'BLIP'])
    parser.add_argument('--source_text_encoder', default='bert-base-uncased')
    parser.add_argument('--source_image_encoder', default='ViT-B/16', choices=['ViT-L/14', 'ViT-B/16', 'ViT-B/32', 'RN50', 'RN101'])
    parser.add_argument('--source_dataset', default='flickr30')
    parser.add_argument('--source_ckpt', default=None, type=str)
    
    #target model
    parser.add_argument('--target_model', default='TCL', choices=['ALBEF', 'CLIP', 'TCL', 'BLIP'])
    parser.add_argument('--target_text_encoder', default='bert-base-uncased', choices=['ALBEF', 'CLIP', 'TCL', 'BLIP'])
    parser.add_argument('--target_image_encoder', default='ViT-B/16', choices=['ViT-L/14', 'ViT-B/16', 'ViT-B/32', 'RN50', 'RN101'])
    parser.add_argument('--target_dataset', default='flickr30', choices=['flickr30', 'mscoco'])

    parser.add_argument('--method', default='your attack method name')
    parser.add_argument('--gpu', type=int, nargs='+', default=[1])
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

