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

from transformers import BertForMaskedLM

import utils
from attacks.step import LinfStep, L2Step
from dataset import pair_dataset
from PIL import Image
from torchvision import transforms


STEPS = {
    'Linf': LinfStep,
    'L2': L2Step,
}

def retrieval_eval(model, ref_model, data_loader, tokenizer, device, config):
    # test
    model.float()
    model.eval()
    ref_model.eval()

    #loss
    criterion = torch.nn.KLDivLoss(reduction='batchmean')

    print('Computing features for evaluation adv...')
    start_time = time.time()

    local_transform = transforms.RandomResizedCrop(224, scale=(0.1, 0.5))
    local_img_transform = transforms.RandomResizedCrop(224, scale=(0.5, 0.8))
    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    args.eps = config['epsilon'] / 255.
  
    uap_noise = torch.zeros(1, *args.data_shape)
 
    args.step_size = args.eps / config['num_iters'] * 1.25
    
    uap_noise = uap_noise.to(device)
    orig_uap_noise = uap_noise.clone().detach()
    step = STEPS[config['constraint']](orig_uap_noise, args.eps, args.step_size)

    print('Prepare memory')
    num_text = len(data_loader.dataset.text)
    num_image = len(data_loader.dataset.ann)
    
    print(num_image) 
    print(num_text)
    
    image_feats = torch.zeros(num_image, model.visual.output_dim)
    text_feats = torch.zeros(num_text, model.visual.output_dim)

    data_iter = iter(data_loader)

    print('Forward')
    iterator = tqdm(range(config['num_iters']), total=config['num_iters'])
    if args.adv != 0:
    for i in iterator:
    
        for images, texts, texts_ids in data_loader:
            images = images.to(device)
            batch_size = images.size(0)

            uap_noise = uap_noise.clone().detach().requires_grad_(True)
                
            #local region of UAP
            uap_noise = uap_noise.to(device)
            patch_uap_noise = local_transform(uap_noise)
            patch_uap_noise = patch_uap_noise.to(device)
                
            #ScMix    
            patch_images = local_img_transform(images)
            patch_images1 = local_img_transform(images)
            patch_images1 = patch_images1.to(device)
            patch_images = patch_images.to(device) 
                            
            l = np.random.beta(args.beta, args.beta)
            l = max(l, 1 - l)
            dp_images = l * patch_images + (1 - l) *patch_images1
            idx = torch.randperm(images.size(0))
                    
            dp_images = 0.8 * dp_images + 0.2 *images[idx]
                
                
            with torch.no_grad():
                text_input = tokenizer(texts, padding='max_length', truncation=True, max_length=77,
                                            return_tensors="pt").to(device)
                text_output = model.inference_text(text_input)
                image_output = model.inference_image(images_normalize(images))
                image_output_patch = model.inference_image(images_normalize(patch_images))
                image_output_patch1 = model.inference_image(images_normalize(patch_images1))
                if args.cls:
                    text_embed = text_output['text_embed'][:, 0, :].detach()
                    image_embed = image_output['image_embed'][:, 0, :].detach()
                    image_embed_patch = image_output_patch['image_embed'][:, 0, :].detach()
                    image_embed_patch1 = image_output_patch1['image_embed'][:, 0, :].detach()
                else:
                    text_embed = text_output['text_embed'].flatten(1).detach()
                    image_embed = image_output['image_embed'].flatten(1).detach()
                    image_embed_patch = image_output_patch['image_embed'].flatten(1).detach()
                    image_embed_patch1 = image_output_patch1['image_embed'].flatten(1).detach()
                    
                image_embed_p = l * image_embed_patch + (1 - l) *image_embed_patch1
                

            image_adv = torch.clamp(images + uap_noise, 0, 1)
            image_adv1 = torch.clamp(images + patch_uap_noise, 0, 1)
            image_adv2 = torch.clamp(dp_images + patch_uap_noise, 0, 1)

            image_adv = images_normalize(image_adv)
            image_adv1 = images_normalize(image_adv1)
            image_adv2 = images_normalize(image_adv2)

            image_adv_output = model.inference_image(image_adv)
            image_adv_output1 = model.inference_image(image_adv1)
            image_adv_output2 = model.inference_image(image_adv2)
                
            if args.cls:
                image_adv_embed = image_adv_output['image_embed'][:, 0, :]
                image_adv_embed1 = image_adv_output1['image_embed'][:, 0, :]
                image_adv_embed2 = image_adv_output2['image_embed'][:, 0, :]
            else:
                image_adv_embed = image_adv_output['image_embed'].flatten(1)
                image_adv_embed1 = image_adv_output1['image_embed'].flatten(1)
                image_adv_embed2 = image_adv_output2['image_embed'].flatten(1)

                
            loss_kl_image = criterion(image_adv_embed.log_softmax(dim=-1), image_embed.softmax(dim=-1)) 
            loss_kl_text = criterion(image_adv_embed.log_softmax(dim=-1), text_embed.softmax(dim=-1))
            loss_kl_image1 = criterion(image_adv_embed1.log_softmax(dim=-1), image_embed.softmax(dim=-1))
            loss_kl_text1 = criterion(image_adv_embed1.log_softmax(dim=-1), text_embed.softmax(dim=-1))                
            loss_kl_image2 = criterion(image_adv_embed2.log_softmax(dim=-1), image_embed.softmax(dim=-1)) + criterion(image_adv_embed2.log_softmax(dim=-1), image_embed_p.softmax(dim=-1))
            loss_kl_text2 = criterion(image_adv_embed2.log_softmax(dim=-1), text_embed.softmax(dim=-1))
                
            loss = - loss_kl_image - loss_kl_text - loss_kl_image1 - loss_kl_text1 - loss_kl_image2 - loss_kl_text2
            

            grad = torch.autograd.grad(loss, [uap_noise])[0]
            with torch.no_grad():
                uap_noise = step.step(uap_noise, grad)
                uap_noise = step.project(uap_noise)
                
    #eval
    for images, texts, texts_ids in data_loader:
        images_ids = [data_loader.dataset.txt2img[i.item()] for i in texts_ids]
        images = images.to(device)
        with torch.no_grad():
            if args.adv != 0:
                images = images + uap_noise 
            images = torch.clamp(images, 0, 1)
            images = images_normalize(images)
            output = model.inference(images, texts)

        image_feats[images_ids] = output['image_feat'].cpu().float().detach()
        text_feats[texts_ids] = output['text_feat'].cpu().float().detach()

    sims_matrix = image_feats @ text_feats.t()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return sims_matrix.cpu().numpy(), sims_matrix.t().cpu().numpy(), uap_noise


@torch.no_grad()
def retrieval_score(model,  image_embeds, text_embeds, text_atts, num_image, num_text, device=None):
    if device is None:
        device = image_embeds.device

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation Direction Similarity With Bert Attack:'

    sims_matrix = F.normalize(image_embeds, dim=-1) @ F.normalize(text_embeds, dim=-1).t()
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
    origin_tr1 = np.load(f'{original_rank_index_path}/{args.model}_ViT_tr1_rank_index.npy')
    origin_tr5 = np.load(f'{original_rank_index_path}/{args.model}_ViT_tr5_rank_index.npy')
    origin_tr10 = np.load(f'{original_rank_index_path}/{args.model}_ViT_tr10_rank_index.npy')

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
    
    origin_ir1 = np.load(f'{original_rank_index_path}/{args.model}_ViT_ir1_rank_index.npy')
    origin_ir5 = np.load(f'{original_rank_index_path}/{args.model}_ViT_ir5_rank_index.npy')
    origin_ir10 = np.load(f'{original_rank_index_path}/{args.model}_ViT_ir10_rank_index.npy')

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

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model, preprocess = clip.load(args.image_encoder, device=device)
    model.set_tokenizer(tokenizer)
    ref_model = BertForMaskedLM.from_pretrained(args.text_encoder)

    model = model.to(device)
    ref_model = ref_model.to(device)

    #### Dataset ####
    print("Creating dataset")
    n_px = model.visual.input_resolution
    args.data_shape=(3,n_px,n_px)
    test_transform = transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.ToTensor(),
    ])
    
    test_dataset = pair_dataset(config['test_file'], test_transform, config['image_root'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size_test'], shuffle = True, num_workers=4)
    

    print("Start eval")
    start_time = time.time()

    score_i2t, score_t2i, uap_noise = retrieval_eval(model, ref_model, test_loader, tokenizer, device, config)

    result, ASR_result = itm_eval(score_i2t, score_t2i, test_dataset.img2txt, test_dataset.txt2img)
    print(result)
    print(ASR_result)
    log_stats = {**{f'test_{k}': v for k, v in result.items()},
                 'cls':args.cls, 'eps': config['epsilon'], 'iters':config['num_iters']}
    with open(os.path.join(args.output_dir, str(args.eta) +"log_CLIP.txt"), "a+") as f:
        f.write(json.dumps(log_stats) + "\n")

    torch.save(uap_noise.cpu().data, '{}/{}'.format(args.output_dir, 'uap.pt'))
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluate time {}'.format(total_time_str))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--dataset', type=str, default='flickr30', choices=['flickr30', 'pascal', 'wikipedia', 'xmedianet'])
    parser.add_argument('--model', default='CLIP')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--image_encoder', default='ViT-B/16', choices=['ViT-L/14', 'ViT-B/16', 'ViT-B/32', 'RN50', 'RN101'])
    parser.add_argument('--method', default='your method name')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--cls', default=True)
    parser.add_argument('--original_rank_index_path', default='./std_eval_idx/flickr30k')  

    args = parser.parse_args()

    args.cls = False
    yaml = yaml.YAML(typ="safe", pure=True)
    config = yaml.load(open(args.config, 'r'))

    args.output_dir = os.path.join('output', args.method, 'uap', str(args.model), str(args.image_encoder), str(args.dataset), str(config['epsilon']), str(config['batch_size_test']))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)

