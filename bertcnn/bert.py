# Libraries import

import torch
import re
import string
import collections
from transformers import BertTokenizerFast, BertForQuestionAnswering


def run_bert(query, context):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    BERT_MODEL_NAME = 'bert-base-uncased'

    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def get_tokens(s):
        if not s: return []
        return normalize_answer(s).split()

    def compute_exact(a_gold, a_pred):
        return int(normalize_answer(a_gold) == normalize_answer(a_pred))

    def compute_f1(a_gold, a_pred):
        gold_toks = get_tokens(a_gold)
        pred_toks = get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def predict(model: BertForQuestionAnswering, query: str, context: str):
        with torch.no_grad():
            model.eval()
            inputs = tokenizer.encode_plus(text=context, text_pair=query, max_length=512, padding='max_length', truncation=True, return_tensors='pt').to(device)
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids'])
            ans_start = torch.argmax(outputs[0])
            ans_end = torch.argmax(outputs[1])
            ans = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][ans_start:ans_end+1]))
        return ans
    

    model = BertForQuestionAnswering.from_pretrained(BERT_MODEL_NAME).to(device)
    model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')), strict=False)

    answer = predict(model, query, context)

    return answer

