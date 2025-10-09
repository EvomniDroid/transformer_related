'''
德语->英语翻译数据集
参考: https://pytorch.org/tutorials/beginner/translation_transformer.html
兼容 torchtext 0.4.0/0.6.0 老版本
'''

from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
import spacy
import os
import urllib.request
import tarfile

# 特殊token
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM = '<unk>', '<pad>', '<bos>', '<eos>'

# 手动下载数据集
def download_multi30k():
    data_dir = '.data/multi30k'
    os.makedirs(data_dir, exist_ok=True)
    
    urls = {
        'train': 'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz',
        'valid': 'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz',
    }
    
    for split, url in urls.items():
        tar_path = os.path.join(data_dir, f'{split}.tar.gz')
        if not os.path.exists(tar_path):
            print(f"正在下载 {split} 数据集...")
            try:
                urllib.request.urlretrieve(url, tar_path)
                print(f"{split} 数据集下载完成")
                # 解压
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(data_dir)
                print(f"{split} 数据集解压完成")
            except Exception as e:
                print(f"下载失败: {e}")
                print("请手动下载数据集或使用其他数据源")
                raise

# 尝试下载数据集
try:
    download_multi30k()
except:
    print("数据集下载失败，将使用简化的示例数据")

# 加载spacy分词器
try:
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')
except:
    print("警告: spacy模型未安装，使用简单的split分词")
    spacy_de = None
    spacy_en = None

# 分词器函数
def de_tokenizer(text):
    if spacy_de:
        return [tok.text for tok in spacy_de.tokenizer(text)]
    return text.split()

def en_tokenizer(text):
    if spacy_en:
        return [tok.text for tok in spacy_en.tokenizer(text)]
    return text.split()

# 定义Field
de_field = Field(tokenize=de_tokenizer, init_token=BOS_SYM, eos_token=EOS_SYM, 
                 pad_token=PAD_SYM, unk_token=UNK_SYM, lower=True)
en_field = Field(tokenize=en_tokenizer, init_token=BOS_SYM, eos_token=EOS_SYM, 
                 pad_token=PAD_SYM, unk_token=UNK_SYM, lower=True)

# 下载翻译数据集
print("正在加载Multi30k数据集...")
train_data, valid_data = Multi30k.splits(
    exts=('.de', '.en'), 
    fields=(de_field, en_field),
    test=None  # 不使用test数据集
)
print(f"数据集加载完成，训练集样本数: {len(train_data)}")

# 构建词表
print("正在构建词表...")
de_field.build_vocab(train_data, min_freq=2)
en_field.build_vocab(train_data, min_freq=2)
print(f"德语词表大小: {len(de_field.vocab)}")
print(f"英语词表大小: {len(en_field.vocab)}")

# 获取词表对象
de_vocab = de_field.vocab
en_vocab = en_field.vocab

# 转换为列表格式，兼容原代码
train_dataset = [(example.src, example.trg) for example in train_data.examples]

# 句子特征预处理
def de_preprocess(de_sentence):
    # 如果输入是列表（已分词），直接使用；如果是字符串，先分词
    if isinstance(de_sentence, list):
        tokens = de_sentence
    else:
        tokens = de_tokenizer(de_sentence)
    tokens = [BOS_SYM] + tokens + [EOS_SYM]
    # 使用词表的stoi字典将tokens转换为ids
    ids = [de_vocab.stoi.get(tok, UNK_IDX) for tok in tokens]
    return tokens, ids

def en_preprocess(en_sentence):
    # 如果输入是列表（已分词），直接使用；如果是字符串，先分词
    if isinstance(en_sentence, list):
        tokens = en_sentence
    else:
        tokens = en_tokenizer(en_sentence)
    tokens = [BOS_SYM] + tokens + [EOS_SYM]
    # 使用词表的stoi字典将tokens转换为ids
    ids = [en_vocab.stoi.get(tok, UNK_IDX) for tok in tokens]
    return tokens, ids

if __name__ == '__main__':
    # 词表大小
    print('de vocab:', len(de_vocab))
    print('en vocab:', len(en_vocab))

    # 特征预处理
    de_sentence, en_sentence = train_dataset[0]
    # train_dataset存储的是已经分词的列表，需要转回字符串
    de_text = ' '.join(de_sentence) if isinstance(de_sentence, list) else de_sentence
    en_text = ' '.join(en_sentence) if isinstance(en_sentence, list) else en_sentence
    print('de preprocess:', *de_preprocess(de_text))
    print('en preprocess:', *en_preprocess(en_text))
