# EC-NL

Code and data for paper [Linking Emergent and Natural Languages via Cospus Transfer]() at ICLR 2022 (spotlight).
```bibtex
@inproceedings{yao2022linking,
  title = {Linking Emergent and Natural Languages via Corpus Transfer},
  author = {Yao, Shunyu and Yu, Mo and Zhang, Yang and Narasimhan, Karthik and Tenenbaum, Joshua and Gan, Chuang},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2022},
  html = {https://openreview.net/pdf?id=49A1Y6tRhaq},
}
```

## Dependencies

- PyTorch 1.8
- SciPy 1.4
- Transformers 4.4.2
- (Optional) Wandb

## Data

[Google Drive](https://drive.google.com/drive/folders/1dBdGaZzvQ4yn-RMpDMxLlFNLzSSbkOWF?usp=sharing) includes

- ```image_features```: Image features of coco-2014 (``coco.pt``) and Conceptual Captions (``cc.pt``) datasets from a pre-trained ResNet, to be used in EC pre-training.

- ```lm_corpora```: Corpora used for language modeling transfer experiments. 

| Name   | Usage | Comment      |
|--------------|-----------|---------|
| cc.pt    | pre-train         | Emergent language       |
| paren-zipf.pt    | pre-train         | Regular language of nesting parentheses  |
| wiki-es.pt    | pre-train         | Spanish (IE-Romance) Wikipedia       |
| wiki-da.pt    | fine-tune         | Danish (IE-Germanic) Wikipedia       |
| wiki-eu.pt    | fine-tune         | Basque (Basque) Wikipedia       |
| wiki-ja.pt    | fine-tune         | Japanese (Japanese) Wikipedia       |
| wiki-ro.pt    | fine-tune         | Romanian (IE-Romance) Wikipedia       |
| wiki-fi.pt    | fine-tune         | Finnish (Uralic) Wikipedia       |
| wiki-id.pt    | fine-tune         | Indonesian (Austronesian) Wikipedia       |
| wiki-kk.pt    | fine-tune         | Kazakh (Turkic) Wikipedia       |
| wiki-he.pt    | fine-tune         | Hebrew (Afro-Asiatic) Wikipedia       |
| wiki-ur.pt    | fine-tune         | Urdu (IE-Indic) Wikipedia       |
| wiki-fa.pt    | fine-tune         | Persian (IE-Iranian) Wikipedia       |





## Experiments

### Emergent Communication (EC) Game
This part aims to generate emergent langauge corpus for downstream tasks.
Download ```image_features``` from Google Drive to ```./ec-pretrain/data```.
To run the emergent communication training, 
```bash
cd ec-game
python train.py
```


Some major options:
- ```--dataset```: use Conceptual Captions (```cc```) or MS-COCO (```coco_2014```) dataset.
- ```--vocab_size```: Vocabulary size (default ```4035```).
- ```--seq_len```: Sequence length limit (default ```15```).

Such a game training automatically stores EC agents (e.g. ```./ckpt/cc_vocab_4035_seq_15_reset_-1_nlayers_1/run77926/model_90.6_1000_4035.pt```) and emergent language corpora (e.g. ```./ckpt/cc_vocab_4035_seq_15_reset_-1_nlayers_1/run77926/model_90.6_1000_4035.pt-cc.pt```, which can be used in place of ```lm_corpora/cc.pt``` from Google Drive)  from different training steps. In the example, ```90.6_1000_4035``` represents game accuracy, game training steps, and game vocabulary size respectively.



### Language Modeling Transfer
This part aims to reproduce Figure 2 of the paper. 
Download ```lm_corpora``` from Google Drive to ```./ec-pretrain/data```.

To run the pre-training, 
```bash
export size=2 # 2,5,10,15,30
export pt_name="wiki-es" # "paren-zipf", "cc"
. pretrain.sh
```

To run the fine-tuning,
```bash
export size=2 # 2,5,10,15,30
export pt_name="wiki-es" # "paren-zipf", "cc"
export ft_name="wiki-ro"
export ckpt=3000
. finetune.sh
```

Meaning of variables above:
- ```size```: Token size (million) of pre-training corpus (```[2, 5, 10, 15, 30]```).
- ```pt_name```: Name of pre-training corpus (```["wiki-es", "paren-zipf", "cc"]```).
- ```ft_name```: Name of fine-tuning corpus (```["wiki-ro", "wiki-da.pt]```).
- ```ckpt```: Which pre-training checkpoint to use for fine-tuning (default ```3000```).

   

## Acknowledgements

The EC part of the code is based on [ECNMT](https://github.com/cambridgeltl/ECNMT), which was partly based on [Translagent](https://github.com/facebookresearch/translagent). 

The LM part of the code is based on [Huggingface run_clm.py](https://github.com/huggingface/transformers/blob/v4.4.2/examples/language-modeling/run_clm.py).

The datasets for our EC experiments include [MS COCO](http://cocodataset.org/#home) and [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions).

The datasets for our LM experiments derive from [tilt-transfer](https://github.com/toizzy/tilt-transfer).

Please cite these resources accordingly. For any question, contact [Shunyu](mailto:shunyuyao.cs@gmail.com).
