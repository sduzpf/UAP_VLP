
# [Universal Adversarial Perturbations for Vision-Language Pre-trained Models](https://arxiv.org/abs/2405.05524)

This is the official PyTorch implementation of the paper "[Universal Adversarial Perturbations for Vision-Language Pre-trained Models](https://arxiv.org/abs/2405.05524)" at *SIGIR 24*. 

<!-- <img src="img.png" width=500> -->

## Requirements
- pytorch 1.10.2
- transformers 4.8.1
- timm 0.4.9
- bert_score 0.3.11


### Prepare datasets and models
Download the datasets, [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) and [MSCOCO](https://cocodataset.org/#home) (the annotations are provided in ./data_annotation/), and put them into `./Dataset`. Set the root path of the dataset in `./configs/Retrieval_flickr.yaml, image_root`.  

The checkpoints of the fine-tuned VLP models are accessible in [CLIP](https://huggingface.co/openai/clip-vit-base-patch16), [ALBEF](https://github.com/salesforce/ALBEF), [TCL](https://github.com/uta-smile/TCL), [BLIP](https://github.com/salesforce/BLIP/tree/main), and put them into `./checkpoint`.

## Learn universal adversarial perturbations
Set paths of source/target model names and checkpoints, dataset names and roots, test file path, original_rank_index_path and so on in corresponding main files before running them. 

```
# Learn UAPs by taking CLIP as the victim
python RetrievalEval.py

# Learn UAPs by taking ALBEF/TCL as the victim 
python Attack_ALBEFTCL.py
```
## Evaluation
### Image-Text Retrieval
```
# Eval CLIP models:
python Eval_Retrieval_CLIP.py

# Eval ALBEF models:
python Eval_Retrieval_ALBEF.py

# Eval TCL models:
python Eval_Retrieval_TCL.py
```

### Visual Grounding
```
Download Refcoco+ datasets from the origin website, and set 'image_root' in configs/Grounding.yaml accordingly.
# Eval:
python Eval_Grounding.py
```

### Image Captioning
```
Download the MSCOCO dataset from the original websites, and set 'image_root' in configs/caption_coco.yaml accordingly.
# Eval:
python Eval_ImgCap_BLIP.py
```

## Citation
If you find this code to be useful for your research, please consider citing.
```

```

## Reference
- [Co-Attack](https://github.com/adversarial-for-goodness/Co-Attack/tree/main), [SGA](https://github.com/Zoky-2020/SGA/tree/main), [ALBEF](https://github.com/salesforce/ALBEF), [BLIP](https://github.com/salesforce/BLIP/tree/main).
