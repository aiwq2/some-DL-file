import gensim
import numpy as np
import torch



def build_word2id(trainpath, validatepath, testpath):
    """
    :param file: word2id #保存地址
    :return: None
    """
    word2id = {'_PAD_': 0}
    id2word = {0: '_PAD_'}
    path=[trainpath,validatepath,testpath]
    for _path in path:
        with open(_path, encoding='utf-8') as f:
            for line in f.readlines():
                sp=line.strip().split()
                for word in sp[1:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)
    # 也可以选择写入到文件中去，但是这里我们暂时保存在内存中
    # with open(file, 'w', encoding='utf-8') as f:
    #     for w in word2id:
    #         f.write(w+'\t')
    #         f.write(str(word2id[w]))
    #         f.write('\n')
    for key, val in word2id.items():
        id2word[val] = key
    return word2id, id2word

def build_word2vec(word2vec_pretrained, word2id, save_to_path=None):
    """
    :param fname # 预训练的 word2vec.
    :param word2id # 语料文本中包含的词汇集.
    :param save_to_path # 保存训练语料库中的词组对应的 word2vec 到本地
    :return #语料文本中词汇集对应的 word2vec 向量{id: word2vec}
    """
    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_pretrained, binary=True)
    id_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            id_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in id_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    # 如果不加这一句，会报错"AttributeError: 'numpy.ndarray' object has no attribute 'dim'"
    id_vecs = torch.from_numpy(id_vecs)
    return id_vecs