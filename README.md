# Fantastic Beasts Datasets: A Benchmark for Open-Vocabulary Semantic Segmentation

This repository contains the collected dataset used in the NeurIPS 2023 paper: AttrSeg: Open-Vocabulary Semantic Segmentation via Attribute Decomposition-Aggregation. [See the paper here](https://arxiv.org/pdf/2309.00096).

![poster](assets/poster.png)



## Brief Introduction

Existing datasets often lack the inclusion of rare or obscure vocabulary.
To address this limitation, we manually curated a dataset titled "Fantastic Beasts", which consists of 20 categories of magical creatures from the film series *Fantastic Beasts and Where to Find Them*. 
This dataset is designed for comprehensive evaluation and simulating real-world scenarios, specifically for two common situations where attribute descriptions are essential:

**Neologisms**: Vanilla category names represent new vocabularies that are often unseen by large language models (LLMs) and vision-language pre-trainings (VLPs).

**Unnameability**: When users encounter unfamiliar objects, they may struggle to name them, particularly in the case of rare or obscure categories.

For more details, please refer to the [paper](https://arxiv.org/pdf/2309.00096).



## Dataset Structure

### Category Names and Attributes
There are 20 categories in Fantastic Beasts dataset, listed as below in alphabetical order:
```
Augurey, Billywig, Chupacabra, Diricawl, Doxy, Erumpent, Fwooper, Graphorn, Grindylow, Kappa, Leucrotta, Matagot, Mooncalf, Murtlap, Nundu, Occamy, Runespoor, Swoopingevil, Thunderbird, Zouwu
```
The class names and their corresponding attributes are stored in `attributes.json`.

### Images and Masks
For each category, images and masks are stored in `images` and `masks` folder, respectively.



## An Example of Usage

Below is an toy example of how to use the Fantastic Beasts dataset. The whole code can be found in `examples/fantastic_beasts_dataset.py`.

```python 
class FantasticBeastsDataset(Dataset):
    def __init__(self, img_root, msk_root, attr_json, transform=None):
        self.img_root = img_root
        self.msk_root = msk_root
        with open(attr_json, 'r') as f:
            self.attr = json.load(f)
        self.transform = transform
        self.categories = ['Augurey', 'Billywig', 'Chupacabra', 'Diricawl', 'Doxy', 'Erumpent', 'Fwooper', 'Graphorn', 'Grindylow', 'Kappa', 'Leucrotta', 'Matagot', 'Mooncalf', 'Murtlap', 'Nundu', 'Occamy', 'Runespoor', 'Swoopingevil', 'Thunderbird', 'Zouwu']
        self.img_pathes = self.get_pathes(self.img_root)
        self.msk_pathes = self.get_pathes(self.msk_root)

    def get_pathes(self, root):
        img_pathes = []
        for category in self.categories:
            category_path = Path(root) / category
            for img_file in category_path.glob("*"):
                img_pathes.append(img_file.resolve().as_posix())
        img_pathes.sort()
        return img_pathes

    def read_img(self, img_path):
        img = np.array(Image.open(img_path)) #img are uint8 with size (h,w,3)
        return img
    
    def read_msk(self, msk_path):
        msk = np.array(Image.open(msk_path)) #msk are uint8 with size (h,w)
        msk[msk > 0] = 1
        return msk

    def read_attr(self, category):
        return self.attr[category]

    def __len__(self):
        return len(self.img_pathes)

    def __getitem__(self, index):
        img_path = self.img_pathes[index]
        msk_path = self.msk_pathes[index]
        img = self.read_img(img_path)
        msk = self.read_msk(msk_path)
        attr = self.read_attr(Path(img_path).name.split('_')[0])
        if self.transform:
            img, msk = self.transform(img, msk)
        return img, msk, attr

fb = FantasticBeastsDataset(img_root="./images", msk_root="./masks", attr_json="./attributes.json")
for img, msk, attr in fb:
    print(img.shape, msk.shape, attr)
```



## Citation
If this dataset is useful for your research, please consider citing:
```
@article{ma2023attrseg,
  title   = {AttrSeg: Open-Vocabulary Semantic Segmentation via Attribute Decomposition-Aggregation},
  author  = {Chaofan Ma and Yuhuan Yang and Chen Ju and Fei Zhang and Ya Zhang and Yanfeng Wang},
  journal = {Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS)},
  year    = {2023}
}
```



## Acknowledgements

We would like to thank the following people for their direct or indirect contributions to the creation of this dataset:
- J.K. Rowling, as the creator of the Wizarding World and the original author of the Harry Potter series, whose work is foundational.
- David Yates, the director of the film, for contributing to its vision and execution.
- David Heyman, the producer of the film, for his pivotal role in bringing the story to the screen.
- The VFX artists and technicians at Framestore and their team leaders, Tim Burke, Christian Manz, and Pablo Grillo, for their incredible work in creating the magical creatures.
- All the Harry Potter fans who support me in creating this dataset.


