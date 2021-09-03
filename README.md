# Zero-shot-img-text


Zero shot classification through img to text matching. 

## Data source
https://github.com/EthanZhu90/ZSL_PP_CVPR17   
[Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) birds images with corresponding wikipedia text articles for each class


## Approach
Img --> ResNet50 --> visual embedding 

Text --> [LongFormer](https://arxiv.org/abs/2004.05150) --> text embedding

visual embedding,text_embedding --> MLP --> match score for each text-img couple

Train MLP through standard cross entropy over outputted scores, ResNet and Transformer models are pretrained.