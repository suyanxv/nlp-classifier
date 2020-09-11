# NLP-CLASSIFIER

使用LSTM对处理的数据集进行训练

`src/lstm/nltk-lstm-classification.ipynb`：使用样本的实验

`src/lstm/nltk-lstm-test.ipynb`：使用全部数据的实验

`src/lstm/text-classification-starter.ipynb`：Naive base和SVM实验


# 主要组件
1. [结巴](https://github.com/fxsjy/jieba)：GitHub上的Python中文分词组件（2.4万收藏）
2. [Keras](https://keras.io/)：LSTM算法所在的开源申请网络库
3. [NLTK](https://www.nltk.org/)
4. [Scikit-Learn](https://scikit-learn.org/stable/)：很多好用的经典算法（`Naive Base`），NLP相关算法（`CountVectorizer`，`TfidfTransformer`）等（`pipeline`，`train_test_split`, etc.）
5. 其他（`pandas`, `numpy`, etc.）


# 数据导入

读取文件并将数据汇集到一个`DataFrame`，分为三列：中性，买入，卖出，并标号。



# 数据分析
买入：17123, 中性：6634, 卖出：12, 空字符串：17, `Null`：0


# 数据预处理
1. Optional: 提取样本（如：中性和买入各6000）
2. 移除空字符串数据
3. 移除newline(`\n`)，tab(`\t`)，null(`\0`)
4. 整理并移除停止词
5. 整理新词并引入到[结巴](https://github.com/fxsjy/jieba)分词器
6. Optional: 仅保留中文字/移除所有数字/移除所有英文/移除标点符合
7. 分词后用空格`join`所有词汇，以便用于`tokenizer`
7. 设置每个文件最大词汇量，如500，并truncate/pad每个文件使文件大小相同
7. 将样本随机分为80%训练集，20%测试集（用于LSTM），或64%训练集，20%测试集和16%验证集（用于Bert）

# LSTM
参数：
``` python
epochs=5
batch_size=64
LSTM units=100
loss=binary_crossentropy
optimizer=adam
activation=sigmoid
Dense units=3 #number of categories
```

用`epoch=5`在2.4万文件上训练，用时大约1.5小时 (Intel i5 2.9 GHz)

# 结果
最佳达到0.759准确率（2.4万数据量，20%用于测试集），后续实验暂时没能做到更好。相比之下，只经过简单的预处理（`CountVectorizer`和`TfidfTransformer`）Naive Base算法便可以达到0.81准确率，SVM算法可达到0.754。因此推测LSTM的中文分词预处理不够

# 后续计划
1. 数据预处理时比较移除和保留数字的区别
2. 提取样本使主要两个分类（买入，中性）个数相当再进行训练
1. 在利用[结巴](https://github.com/fxsjy/jieba)中文分词时，尽量引入新词，比如公司名称等
1. 使用[Bert](https://github.com/google-research/bert)算法，参考的[Bert教程（网络资源）](https://blog.csdn.net/qq_20989105/article/details/89492442)，Bert环境：Python=3.6, Tensorflow=1.14（最新版Tensowflow 2.x 无法运行Bert，需downgrade）
1. 利用[词向量](https://en.wikipedia.org/wiki/Word_embedding)进行训练




# 附录

## Bert相关代码（需clone Bert，然后放入run_classifier.py）
```python
class MyTaskProcessor(DataProcessor):
  def __init__(self):
    self.labels = ['中性','买入','卖出'] 
  
  def get_train_examples(self, data_dir):
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, 'train.csv')), 
      'train')

  def get_dev_examples(self, data_dir):
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, 'val.csv')), 'val')

  def get_test_examples(self, data_dir):
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, 'test.csv')), 'test')
  
  def get_labels(self):
    return self.labels
  
  def _create_examples(self, lines, set_type):
    examples = []
    for(i, line) in enumerate(lines):
      guid = '%s-%s' %(set_type, i)
      text_a = tokenization.convert_to_unicode(line[0])
      label = tokenization.convert_to_unicode(line[1])
      examples.append(InputExample(guid=guid, text_a=text_a, label=label))
    return examples
```
