
# [Universal Adversarial Perturbations for Vision-Language Pre-trained Models](https://arxiv.org/abs/2206.09391)

This is the official PyTorch implementation of the paper "[Universal Adversarial Perturbations for Vision-Language Pre-trained Models](https://arxiv.org/abs/2206.09391)" at *SIGIR 2024*. 

<!-- <img src="img.png" width=500> -->

## Requirements
- pytorch 1.10.2
- transformers 4.8.1
- timm 0.4.9
- bert_score 0.3.11


### Prepare datasets and models
Download the datasets, [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) and [MSCOCO](https://cocodataset.org/#home) (the annotations is provided in ./data_annotation/). Set the root path of the dataset in `./configs/Retrieval_flickr.yaml, image_root`.  
The checkpoints of the fine-tuned VLP models are accessible in [ALBEF](https://github.com/salesforce/ALBEF), [TCL](https://github.com/uta-smile/TCL), [CLIP](https://huggingface.co/openai/clip-vit-base-patch16).


## Learn universal adversarial perturbations
Set paths of source/target model checkpoint, datasets and others in corresponding main files 
```
# Learn UAPs by taking CLIP as the victim
python RetrievalEval.py

# Learn UAPs by taking ALBEF/TCL as the victim 
python Attack_ALBEFTCL.py

## Evaluation
### Image-Text Retrieval
```
Eval CLIP models:
python Eval_CLIP.py

Eval ALBEF models:
python Eval_ALBEF.py

Eval TCL models:
python Eval_TCL.py
```

### Visual Grounding
```
Download Refcoco+ datasets from the origin website.
Eval:
python Eval_Grounding.py
```

### Image Captioning
```
Eval:
python Eval_ImgCap_BLIP.py
```

## Citation
If you find this code to be useful for your research, please consider citing.
```
@inproceedings{zhang2022towards,
  title={Towards Adversarial Attack on Vision-Language Pre-training Models},
  author={Zhang, Jiaming and Yi, Qi and Sang, Jitao},
  booktitle="Proceedings of the 30th ACM International Conference on Multimedia",
  year={2022}
}
```

## Reference
- [ALBEF](https://github.com/salesforce/ALBEF)
