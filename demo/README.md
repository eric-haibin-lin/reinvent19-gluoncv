## 1. Build, Train, and Deploy Natural Language Processing Models on SageMaker

[GluonNLP](http://gluon-nlp.mxnet.io) provides implementations of the state-of-the-art deep learning models for natural language processing, built on top of [Apache MXNet](http://mxnet.io). GluonNLP model zoo features:
- Machine Translation
- Language Modeling
- Text Classification and Sentiment Analysis
- Word Embedding

#### Finetune and Deploy BERT Model for Sentiment Analysis on SageMaker

1. Send an email to `reinvent19-gluonnlp@request-nb.mxnet.io` to request a SageMaker notebook instance
2. Wait for a few minutes and click the notebook access URL in the follow-up email
3. In the browser, open `reinvent19-gluonnlp/tutorial/train_deploy_bert.ipynb` and follow the instructions

If the notebook access URL does not work, alternatively you can view the jupyter notebook content [here](https://nbviewer.jupyter.org/github/eric-haibin-lin/reinvent19-gluonnlp/blob/master/tutorial/train_deploy_bert.ipynb).
If you want to reuse the content after the event, you can setup a SageMaker notebook as instructed [here](https://github.com/eric-haibin-lin/reinvent19-gluonnlp/tree/master/tutorial).

## 2. Run Computer Vision Models on Your Laptop

[GluonCV](http://gluon-cv.mxnet.io) is a deep learning toolkit for computer vision built on top of [Apache MXNet](http://mxnet.io). GluonCV provides several implementations of state-of-the-art deep learning algorithms in computer vision:
- Image Classification
- Object Detection
- Semantic / Instance Segmentation
- Pose Estimation
- Action Recognition

#### Run Pose Estimation using Your WebCam

1. `conda env create -f gluoncv.yml`
2. `conda activate gluoncv`
3. `python pose_estimation_demo.py`
