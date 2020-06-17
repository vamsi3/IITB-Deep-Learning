import argparse
import os
import numpy as np
import dataloader
import criteria
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset

from transformers import BertTokenizer, BertForSequenceClassification


# class UniversalSentenceEncoder(object):
#     def __init__(self, cache_path):
#         super().__init__()
#         os.environ['TFHUB_CACHE_DIR'] = cache_path
#         module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
#         self.embed = hub.Module(module_url)
#         config = tf.ConfigProto()
#         config.gpu_options.allow_growth = True
#         self.sess = tf.Session(config=config)
#         self.build_graph()
#         self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

#     def build_graph(self):
#         self.sts_input1 = tf.placeholder(tf.string, shape=(None))
#         self.sts_input2 = tf.placeholder(tf.string, shape=(None))
#         sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
#         sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
#         self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
#         clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
#         self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

#     def semantic_sim(self, sents1, sents2):
#         scores = self.sess.run([self.sim_scores], feed_dict={ self.sts_input1: sents1, self.sts_input2: sents2 })
#         return scores

from InferSent.models import InferSent

torch.set_grad_enabled(False)

class UniversalSentenceEncoder:
    def __init__(self):
        super().__init__()
        model_version = 1
        MODEL_PATH = "InferSent/encoder/infersent%s.pkl" % model_version
        W2V_PATH = 'InferSent/GloVe/glove.840B.300d.txt' if model_version == 1 else 'fastText/crawl-300d-2M.vec'
        
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
        self.model = InferSent(params_model)
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()
        use_cuda = True
        self.model = self.model.cuda() if use_cuda else self.model
        self.model.set_w2v_path(W2V_PATH)
        self.model.build_vocab_k_words(K=100000)

    def semantic_sim(self, sents1, sents2):
        embed1 = self.model.encode(sents1, tokenize=False)
        embed2 = self.model.encode(sents2, tokenize=False)
        embed1 = torch.tensor(embed1)
        embed2 = torch.tensor(embed2)
        sts_encode1 = embed1 / torch.norm(embed1, p=2, dim=1, keepdim=True)
        sts_encode2 = embed2 / torch.norm(embed2, p=2, dim=1, keepdim=True)
        cosine_similarities = torch.sum(sts_encode1 * sts_encode2, dim=1)
        clip_cosine_similarities = torch.clamp(cosine_similarities, -1.0, 1.0)
        scores = 1.0 - torch.acos(clip_cosine_similarities)
        return scores.cpu().numpy()


def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    sim_words, sim_values = [], []
    for idx, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[idx]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        sim_word = [idx2word[id] for id in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
    return sim_words, sim_values


class BERTInference(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 nclasses,
                 max_seq_length=128,
                 batch_size=32):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses, output_attentions=True).cuda()
        self.dataset = BERTDataset(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data, batch_size=32):
        self.model.eval()
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        probs_all = []
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            with torch.no_grad():
                logits, attentions = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0), attentions


class BERTDataset(Dataset):
    def __init__(self, pretrained_dir, max_seq_length=128, batch_size=32):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def transform_text(self, data, batch_size=32):
        features = []
        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        for text_a in data:
            tokens_a = self.tokenizer.tokenize(' '.join(text_a))
            tokens_a = tokens_a[:(self.max_seq_length - 2)]
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (self.max_seq_length - len(input_ids))

            all_input_ids.append(input_ids + padding)
            all_input_mask.append([1] * len(input_ids) + padding)
            all_segment_ids.append([0] * len(tokens) + padding)
        
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
        return eval_dataloader


def attack(text_ls, true_label, predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
           batch_size=32):
    # first check the prediction of the original text
    orig_probs, import_scores = predictor([text_ls])
    # import_scores = import_scores[-1][0].mean(0).sum(0).cpu().numpy()
    orig_probs = orig_probs.squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:
        len_text = len(text_ls)
        import_scores = import_scores[:len_text]
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(text_ls)

        # get importance score
        leave_1_texts = [text_ls[:ii] + ['<oov>'] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
        leave_1_probs, _ = predictor(leave_1_texts, batch_size=batch_size)
        num_queries += len(leave_1_texts)
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
        import_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
                    leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
                                                                      leave_1_probs_argmax))).data.cpu().numpy()

        # get words to perturb ranked by importance scorefor word in words_perturb
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            try:
                if score > import_score_threshold and text_ls[idx] not in stop_words_set:
                    words_perturb.append((idx, text_ls[idx]))
            except:
                print(idx, len(text_ls), import_scores.shape, text_ls, len(leave_1_texts))

        # find synonyms
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
        synonyms_all = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = text_ls[:]
        text_cache = text_prime[:]
        num_changed = 0
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs, _ = predictor(new_texts, batch_size=batch_size)

            # compute semantic similarity
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            semantic_sims = \
            sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                       list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

            num_queries += len(new_texts)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # prevent incompatible pos
            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1
                break
            else:
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                        (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]
        return ' '.join(text_prime), num_changed, orig_label, torch.argmax(predictor([text_prime])[0]), num_queries


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--dataset_path",
                        type=str,
                        required=True,
                        help="Which dataset to attack.")
    parser.add_argument("--nclasses",
                        type=int,
                        default=2,
                        help="How many classes for classification.")
    parser.add_argument("--target_model",
                        type=str,
                        required=True,
                        choices=['wordLSTM', 'bert', 'wordCNN'],
                        help="Target models for text classification: fasttext, charcnn, word level lstm "
                             "For NLI: InferSent, ESIM, bert-base-uncased")
    parser.add_argument("--target_model_path",
                        type=str,
                        required=True,
                        help="pre-trained target model path")
    parser.add_argument("--word_embeddings_path",
                        type=str,
                        default='',
                        help="path to the word embeddings for the target model")
    parser.add_argument("--counter_fitting_embeddings_path",
                        type=str,
                        required=True,
                        help="path to the counter-fitting embeddings we used to find synonyms")
    parser.add_argument("--counter_fitting_cos_sim_path",
                        type=str,
                        default='',
                        help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
    parser.add_argument("--USE_cache_path",
                        type=str,
                        required=True,
                        help="Path to the USE encoder cache.")
    parser.add_argument("--output_dir",
                        type=str,
                        default='adv_results',
                        help="The output directory where the attack results will be written.")

    ## Model hyperparameters
    parser.add_argument("--sim_score_window",
                        default=15,
                        type=int,
                        help="Text length or token number to compute the semantic similarity score")
    parser.add_argument("--import_score_threshold",
                        default=-1.,
                        type=float,
                        help="Required mininum importance score.")
    parser.add_argument("--sim_score_threshold",
                        default=0.7,
                        type=float,
                        help="Required minimum semantic similarity score.")
    parser.add_argument("--synonym_num",
                        default=50,
                        type=int,
                        help="Number of synonyms to extract")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size to get prediction")
    parser.add_argument("--data_size",
                        default=1000,
                        type=int,
                        help="Data size to create adversaries")
    parser.add_argument("--perturb_ratio",
                        default=0.,
                        type=float,
                        help="Whether use random perturbation for ablation study")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="max sequence length for BERT target model")

    args = parser.parse_args()

    # get data to attack
    texts, labels = dataloader.read_corpus(args.dataset_path)
    data = list(zip(texts, labels))
    data = data[:args.data_size] # choose how many samples for adversary
    print("Data import finished!")

    # construct the model
    print("Building Model...")
    model = BERTInference(args.target_model_path, nclasses=args.nclasses, max_seq_length=args.max_seq_length)
    predictor = model.text_pred
    print("Model built!")

    # prepare synonym extractor
    # build dictionary via the embedding file
    idx2word = {}
    word2idx = {}

    print("Building vocab...")
    with open(args.counter_fitting_embeddings_path, 'r') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1

    print("Building cos sim matrix...")
    cos_sim = np.load(args.counter_fitting_cos_sim_path)
    print("Cos sim import finished!")

    # build the semantic similarity module
    # use = UniversalSentenceEncoder(args.USE_cache_path)
    use = UniversalSentenceEncoder()

    stop_words_set = criteria.get_stopwords()
    print('Start attacking!')
    changed_rates = []
    total_time, success, total = 0, 0, 0
    for idx, (text, true_label) in enumerate(data):
        tick = time.time()
        new_text, num_changed, orig_label, \
        new_label, num_queries = attack(text, true_label, predictor, stop_words_set,
                                        word2idx, idx2word, cos_sim, sim_predictor=use,
                                        sim_score_threshold=args.sim_score_threshold,
                                        import_score_threshold=args.import_score_threshold,
                                        sim_score_window=args.sim_score_window,
                                        synonym_num=args.synonym_num,
                                        batch_size=args.batch_size)

        old_text = ' '.join(text)
        print(f"Original: {old_text}")
        print()
        print(f"New:      {new_text}")
        print("--------------------------------------------------------------")

        changed_rate = 1.0 * num_changed / len(text)
        if true_label == orig_label and true_label != new_label:
            changed_rates.append(changed_rate)
            tock = time.time()
            total_time += tock - tick
            success += 1
        total += 1
    print(f"Time: {total_time}\tAvg. Change Rate: {np.mean(changed_rates)*100}\tSuccess Rate: {(success / total) * 100}")


if __name__ == "__main__":
    main()
