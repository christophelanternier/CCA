# CCA

This project was realized in the frame of the Master of Science MVA from ENS Cachan, during fall 2016.

It implements a three-view Canonical Correlation Analysis algorithm, following for the biggest part the following paper: https://arxiv.org/abs/1212.4522

This study was realized with the COCO database: http://mscoco.org/

Image features were computed using the fc2 layer of a pre-trained VGG16 CNN implemented here: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

Text features were computed using pre-trained Word2Vec or Glove.

The notebook extract_features.ipynb can help you extract the features (caution for images it can take up to 12 hours)

main.py performs basic operations like Image-to-Tag or Tag-to-Image search.

main.ipynb can help you perform more tasks
