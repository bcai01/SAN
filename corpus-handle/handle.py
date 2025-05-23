# 原始文件在下载中，corpus.tar.gz
import json
# import en_core_web_sm
from tqdm import tqdm
import string
 
import spacy

nlp = spacy.load("en_core_web_sm")


PUNCTUATION = list(string.punctuation)
special_words = ["am", "is", "was", "are", "were", "can", "could", "will",
                 "would", "shall", "should", "may", "must", "might"]


from PosContributionCalculator import PosContributionCalculator

pos_caltor = PosContributionCalculator(pos_con_path='./vocab_pos_end.txt')
from transformers import AutoTokenizer, BertForMaskedLM
import torch

path='/mnt/model-files/bert-base-uncased/'

tokenizer = AutoTokenizer.from_pretrained(path)
model = BertForMaskedLM.from_pretrained(path)
model = model.to('cuda:0')

# def convert_to_negation(parser, sentence):
def convert_to_negation(parsered_sentence):
    # parsered_sentence = parser(sentence)
    tokens = [str(_) for _ in parsered_sentence]
    deps = [_.dep_ for _ in parsered_sentence]
    tags = [_.tag_ for _ in parsered_sentence]
    lemmas = [_.lemma_ for _ in parsered_sentence]

    if "not" in tokens:
        index = tokens.index("not")
        del tokens[index]
        sentence_negation = " ".join(tokens)
        return sentence_negation

    flag = 0
    for dep in deps:
        if dep == "aux" or dep == "auxpass":
            flag = 1
            break
        if dep == "ROOT":
            flag = 2
            break

    if flag == 1:
        for i, dep in enumerate(deps):
            if dep == "aux" or dep == "auxpass":
                tokens[i] += " not"
                break
    elif flag == 2:
        index = deps.index("ROOT")
        if tokens[index].lower() in special_words:
            tokens[index] += " not"
        elif tags[index] == "VBP":
            tokens[index] = "do not " + lemmas[index]
        elif tags[index] == "VBZ":
            tokens[index] = "does not " + lemmas[index]
        elif tags[index] == "VBD":
            tokens[index] = "did not " + lemmas[index]
        else:
            tokens.insert(0, "Not")
    else:
        tokens.insert(0, "Not")

    sentence_negation = " ".join(tokens)

    return sentence_negation.strip().replace('\t', ' ')


def replace_with_mask(sentence):
    # if sentence[-1] not in PUNCTUATION:
        # sentence += "."
    sentence = [_ + '.' if _[-1] not in PUNCTUATION else _ for _ in sentence]
    
    encodes = tokenizer(sentence, padding=True, return_tensors='pt')

    encodes['input_ids'] = encodes['input_ids'].to('cuda:0')
    encodes['attention_mask'] = encodes['attention_mask'].to('cuda:0')
    encodes['token_type_ids'] = encodes['token_type_ids'].to('cuda:0')

    pos_scores = pos_caltor(encodes['input_ids']) / 20
    mask = torch.bernoulli(pos_scores).bool()
    # print(encodes['input_ids'])
    encodes['input_ids'][mask] = tokenizer.mask_token_id
    # print(encodes['input_ids'])
    # print(mask)
    # print(pos_scores)

    outputs = model(**encodes)

    # logits = outputs.logits
    vocab_ids = torch.argmax(outputs.logits, dim=-1)
    # print(vocab_ids.shape)
    vocab_ids = vocab_ids * encodes['attention_mask']
    vocab_ids[encodes['input_ids']==tokenizer.cls_token_id] = tokenizer.cls_token_id
    vocab_ids[encodes['input_ids']==tokenizer.sep_token_id] = tokenizer.sep_token_id

    outs = tokenizer.batch_decode(vocab_ids, skip_special_tokens=True)
    return [_.strip().replace('\t', ' ') for _ in outs]

if __name__ == "__main__":

    # parser = en_core_web_sm.load()

    in_file = r"../corpus/wiki1m_for_simcse.txt"

    out_file = r"../outputs/corpus/001.from-prev-exp"

    f = open(in_file)
    lines = f.readlines()

    f1 = open(out_file, "w")
    f1.write(f'text\tneg_soft\tre_mask1\tre_mask2\tre_mask3\tre_mask4\n')

    # print(model.device)
    batch_size = 64
    all_len = len(lines)
    num_batches = (all_len + batch_size - 1) // batch_size
    
    for index in tqdm(range(num_batches), total=num_batches):
        start_index = index * batch_size
        end_index = min((index + 1) * batch_size, all_len)
        
        sentence = [ _.strip().replace('\t', ' ') for _ in lines[start_index : end_index]]
        sentence = [ _[:300] if len(_) > 300 else _ for _ in sentence]
        docs = nlp.pipe(sentence, batch_size=len(sentence))
        # negation = convert_to_negation(parser=nlp, sentence=sentence)
        negation = [convert_to_negation(doc) for doc in docs]
        t1 = replace_with_mask(sentence)
        t2 = replace_with_mask(t1)
        t3 = replace_with_mask(t2)
        t4 = replace_with_mask(t3)

        t1 = t1.replace('"', '""')
        t2 = t2.replace('"', '""')
        t3 = t3.replace('"', '""')
        t4 = t4.replace('"', '""')
        # temp = [sentence, negation]
        # for i in range(len(sentence)):
        # print(len(sentence), len(negation), len(t1), len(t2), len(t3), len(t4))
        # print(sentence)
        # print(negation)
        res = ''.join( [ f'"{sentence[i]}"\t"{negation[i]}"\t"{t1[i]}"\t"{t2[i]}"\t"{t3[i]}"\t"{t4[i]}"\n' for i in range(len(sentence))])
        f1.write(res)

    f.close()
    f1.close()