
"""
词性贡献计算器，主要负责计算词性对在句子中的贡献值。
采取离线计算方式
"""
import torch

"""
    ADJ: adjective
    ADP: adposition
    ADV: adverb
    AUX: auxiliary
    CCONJ: coordinating conjunction
    DET: determiner
    INTJ: interjection
    NOUN: noun
    NUM: numeral
    PART: particle
    PRON: pronoun
    PROPN: proper noun
    PUNCT: punctuation
    SCONJ: subordinating conjunction
    SYM: symbol
    VERB: verb
    X: other

ADJ：形容词
ADP:位置
ADV：副词
AUX：辅助
CCONJ：协调配合
DET：确定器
INTJ：感叹词
名词
NUM：数字
零件：颗粒
PRON：代词
专有名词
PUNCT：标点符号
SCONJ：从属连词
SYM：符号
VERB：动词
X： 其他
    NOUN (名词): 9分
        名词通常是句子的主要信息载体，表示人、事、物等实体，对于确定文本主题和内容至关重要。
    VERB (动词): 9分
        动词表达行为、状态或过程，是理解事件动态的核心，对于语义相似度判断至关重要。
    ADJ (形容词): 8分
        形容词修饰名词，提供关键的描述信息，影响对实体特征的理解和比较。
    ADV (副词): 7分
        副词修饰动词、形容词或其他副词，提供动作的方式、时间、地点等细节，对精确理解语境很重要。
    PRON (代词): 7分
        代词替代名词，对于跟踪指代关系和理解上下文极为重要，尤其是在解决“它/他/她”等指代问题时。
    DET (限定词): 6分
        限定词如“the”, “a”，“an”等虽小，但对确定名词的特指或泛指性质有影响，从而影响语义理解。
    PROPN (专有名词): 8分
        专有名词如人名、地名等，直接关联到特定实体，对于识别文本涉及的具体对象非常关键。
    CCONJ (并列连词): 6分
        并列连词连接同等成分，有助于理解句子结构和逻辑关系，但在相似度计算中可能不如实体词直接。
    SCONJ (从属连词): 5分
        引导从句，揭示句子间的关系，对理解复杂句式结构有帮助，但对直接的语义相似度贡献较间接。
    AUX (助动词): 4分
        助动词辅助构成时态、语态等，对语法结构理解有帮助，但在语义层面的直接贡献有限。
    ADP (介词): 5分
        介词构建短语表达时空关系和其他抽象关系，对理解上下文有辅助作用，但单独贡献不大。
    NUM (数词): 4分
        数词提供量化的信息，对某些类型文本的相似度判断有直接影响，但在广泛语境下相对次要。
    PART (助词、语气词): 3分
        如英语中的“to”、“up”等，它们的功能多样但往往依赖上下文，对相似度的直接影响较小。
    INTJ (感叹词): 2分
        感叹词表达情感或反应，对文本意义贡献有限，除非在特定情境下反映作者态度。
    PUNCT (标点符号): 1分
        标点帮助断句和理解句子结构，但不是直接的语义携带者。
    SYM (符号): 1分
        符号如“$”、“#”等，除非在特定上下文中，否则对语义贡献极小。
    X (其他): 2分
        包括难以分类的词，其重要性依具体情况而定，一般情况下较低。
"""

class PosContributionCalculator:
    def __init__(self, pos_con_path: str, contribution: dict={}, batch_size=64):
        self.pos_scores ={
            "ADJ": 8,
            "ADP": 5,
            "ADV": 7,
            "AUX": 4,
            "CCONJ": 6,
            "DET": 6,
            "INTJ": 2,
            "NOUN": 9,
            "NUM": 4,
            "PART": 3,
            "PRON": 7,
            "PROPN": 8,
            "PUNCT": 1,
            "SCONJ": 5,
            "SYM": 1,
            "VERB": 9,
            "X": 1,
            "PAD": 0,
            "CLS": 0,
            "SEP": 0
        }
        self.pos_scores.update(contribution)
        self.batch_size = batch_size
        self.pos_scores_for_ids = self.build_vocab_pos(pos_con_path)

    def build_vocab_pos(self, file_path):
        with open(file_path, "r") as f:
            pos = list()
            pos_scores_for_ids = [ self.pos_scores[word.strip()] for word in f.read().splitlines()]

        return torch.Tensor(pos_scores_for_ids).unsqueeze(0)


    def __call__(self, vecs: torch.Tensor):
        """
        计算句子中词性对的贡献值。
        :param sentence: input_ids
        :return: 词性对的贡献值
        """
        pos_scores_for_ids = self.pos_scores_for_ids.to(vecs.device)
        return torch.gather(pos_scores_for_ids.repeat(vecs.shape[0], 1), 1, vecs)