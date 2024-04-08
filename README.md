# Spuriousness-Aware Meta-Learning for Learning Robust Classifiers
## Preparation
### Download datasets:
Download all the datasets and decompress them into individual folders.
- [Waterbirds](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz)
- [Celeba](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
- [NICO](https://drive.google.com/drive/folders/17-jl0fF9BxZupG75BtpOqJaB6dJ2Pv8O?usp=sharing)
- [ImageNet (train)](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar)
- [ImageNet (val)](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar)
- [ImageNet-A](https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar)

For the NICO dataset, run the following to prepare metadata:
```python 
from data.nico_data import prepare_metadata
prepare_metadata(NICO_DATA_FOLDER, NICO_CXT_DIC_PATH, NICO_CLASS_DIC_PATH)
```

For the ImageNet-9 and ImageNet-A datasets, run the following to prepare metadata:
```python 
from data.in9_data import prepare_imagenet9_metadata, prepare_imageneta_metadata
prepare_imagenet9_metadata("/path/to/imagenet")
prepare_imageneta_metadata("/path/to/imagenet-a")
```
### Extract attributes
In the `extract_concepts.py` file, modify `csv_path` and `img_path`. Then, run the following script:
```python
python extract_concepts.py --dataset waterbirds --model vit-gpt2
```

## How to run
In the `config/waterbirds.yaml` file (or other `yaml` files), specify the following parameters:
```
data_folder: /path/to/datasets
save_folder: /path/to/results/
vit_gpt2_attr_embed_path: "/path/data/vit-gpt2_img_embeddings.pickle"
vit_gpt2_vocab_path: "/path/data/vit-gpt2_vocab.pickle"
blip_attr_embed_path: "/path/data/blip_img_embeddings.pickle"
blip_vocab_path: "/path/data/blip_vocab.pickle"
vlm: "vit-gpt2"
```
Then run the following code:
```python
python train_meta_spurious.py --config config/waterbirds.yaml
```

The final results are shown in the last line starting with "[PseudoUnbiasedModel]".