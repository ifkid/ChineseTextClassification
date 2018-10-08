## CNN+RNN for  Chinese text classification

使用卷积神经网络和循环神经网络进行中文文本的分类

CNN做句子分类的论文可以参看: [Convolutional Neural Networks for Sentence Classification][1]

还可以去读dennybritz大牛的博客：[Implementing a CNN for Text Classification in TensorFlow][2]
以及字符级CNN的论文：[Character-level Convolutional Networks for Text Classification][3]

本文是基于TensorFlow在中文数据集上的简化实现，使用了字符级CNN和RNN对中文文本进行分类，达到了较好的效果。

文中所使用的Conv1D与论文中有些不同，详细参考官方文档：[tf.nn.conv1d][4]
### Requirements

Tensorfow >= 1.3
Python 2/3
numpy
scikit-learn
scipy


### Usage

python run.py cnn/rnn train/test

please choose one model from cnn and rnn to train first, and then test on test text. Finally you can get a predict text set to predict.

### Note
size of training text is too large to upload, you can download it from [HERE][5], and then put it in path data/cnews




[1]: https://arxiv.org/abs/1408.5882
[2]: http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow
[3]: https://arxiv.org/abs/1509.01626
[4]: https://www.tensorflow.org/api_docs/python/tf/nn/conv1d
[5]: https://pan.baidu.com/s/1KhdqYyfC047vRXTZ0m7ivw   "Training text"


