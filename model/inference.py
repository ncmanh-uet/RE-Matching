import json
import os
import numpy as np
import data_process
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig
from evaluation import extract_relation_emb, evaluate
from model import REMatchingModel
from argparse import ArgumentParser
from runtime import set_seed, ROOT, resolve_device, ordered_labels, load_checkpoint

def main():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed, should match the splited dataset if you don not wanna use original splited")
    parser.add_argument("--gamma", type=float, default=0.06, help="Loss function: margin factor gamma")
    parser.add_argument("--alpha", type=float, default=0.33,
                        help="Similarity: balance entity and context weights in single sample")
    parser.add_argument("--lambda_", type=float, default=0.5, help='Reversal rate in Gradient Reversal Layer')
    parser.add_argument("--num_negsample", type=int, default=7,
                        help='the number of negative sample per positive sample')
    parser.add_argument("--entity_way", type=str, choices=['tmp', 'keyword'], default='tmp',
                        help='the description entity represent way')
    # file_path
    parser.add_argument("--dataset_path", type=str, default=str(ROOT / 'data'), help='where data stored')
    parser.add_argument("--rel2id_path", type=str, default=str(ROOT / 'data' / 'rel2id'))
    parser.add_argument("--relation_description", type=str,
                        default='relation_description_addsynonym_processed.json',
                        help='relation description of manual design')
    parser.add_argument("--ckpt_save_path", type=str, default=str(ROOT / 'ckpt'))
    # model and cuda config
    parser.add_argument("--ckpt_name", type=str, default='')
    parser.add_argument("--visible_device", type=str, default='0', help='which devices this model run')
    parser.add_argument("--pretrained_model", type=str, default='bert-base-uncased', help='huggingface pretrained model')

    parser.add_argument('--device', default='auto')
    parser.add_argument('--description_model', default=None)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    args = parser.parse_args()
    import re
    match = re.fullmatch(r'(fewrel|wikizsl)_split_(.+)_unseen_(\d+)_f1_.+', args.ckpt_name)
    if match is None:
        parser.error('ckpt_name must follow DATASET_split_SEED_unseen_N_f1_SCORE')
    args.dataset = match[1]
    args.ckpt_load_path = os.path.join(args.ckpt_save_path, args.ckpt_name)
    args.rel_split_seed = match[2]
    args.unseen = int(match[3])
    print(f'dataset:{args.dataset}, rel_split_seed : {args.rel_split_seed}, unseen: {str(args.unseen)}')

    # dataset ,relation_description, rel2id path
    args.dataset_file = os.path.join(args.dataset_path, args.dataset, f'{args.dataset}_dataset.json')
    args.relation_description_file = os.path.join(args.dataset_path, args.dataset, 'relation_description',
                                                  args.relation_description)
    args.rel2id_file = os.path.join(args.rel2id_path, f'{args.dataset}_rel2id',
                                    f'{args.dataset}_rel2id_{str(args.unseen)}_{args.rel_split_seed}.json')

    # cuda set
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_device
    args.device = str(resolve_device(args.device))


    set_seed(args.seed)

    # load rel2id and id2rel
    with open(args.rel2id_file, 'r', encoding='utf-8') as r2id:
        relation2idx = json.load(r2id)
        train_relation2idx, test_relation2idx = relation2idx['train'], relation2idx['test']
        train_idx2relation, test_idx2relation, = dict((v, k) for k, v in train_relation2idx.items()), \
                                                 dict((v, k) for k, v in test_relation2idx.items())

    train_label, test_label = ordered_labels(train_relation2idx), ordered_labels(test_relation2idx)

    # load rel_description
    with open(args.relation_description_file, 'r', encoding='utf-8') as rd:
        relation_desc = json.load(rd)
        train_desc = [i for i in relation_desc if i['relation'] in train_label]
        test_desc = [i for i in relation_desc if i['relation'] in test_label]
    # load data
    with open(args.dataset_file, 'r', encoding='utf-8') as d:
        raw_data = json.load(d)
        test_data = [i for i in raw_data if i['relation'] in test_label]

    # print info
    print('there are {} kinds of relation in test.'.format(len(set(test_label))))

    # load description
    train_rel2vec, test_rel2vec = data_process.generate_attribute(args, train_desc, test_desc)

    # load model
    config = AutoConfig.from_pretrained(args.pretrained_model, num_labels=len(set(train_label)))
    if config.model_type != 'bert':
        raise ValueError('REMatchingModel requires a BERT backbone.')
    config.pretrained_model = args.pretrained_model
    config.margin = args.gamma
    config.alpha = args.alpha
    if any(vector.shape[-1] != config.hidden_size
           for vectors in test_rel2vec.values() for vector in vectors):
        raise ValueError('Description encoder dimension must match BERT hidden_size.')
    config.lambda_ = args.lambda_
    model = REMatchingModel.from_pretrained(args.pretrained_model, config=config)
    load_checkpoint(model, args.ckpt_load_path)
    model = model.to(args.device)

    # To evaluate the inference time
    test_batchsize = args.eval_batch_size
    testset = data_process.FewRelDataset(args, 'test', test_data, test_rel2vec, test_relation2idx)
    testloader = DataLoader(testset, batch_size=test_batchsize,
                            collate_fn=data_process.create_mini_batch, shuffle=False)

    test_y_attr, test_y, test_y_e1, test_y_e2 = [], [], [], []

    for i, test in enumerate(test_data):
        label = int(test_relation2idx[test['relation']])
        test_y.append(label)

    for i in test_label:
        test_y_attr.append(test_rel2vec[i][0])
        test_y_e1.append(test_rel2vec[i][1])
        test_y_e2.append(test_rel2vec[i][2])

    test_y, test_y_attr, test_y_e1, test_y_e2 = np.array(test_y), np.array(test_y_attr), np.array(test_y_e1), np.array(
        test_y_e2)

    print('============== EVALUATION ON Test DATA ==============')
    preds, e1_hs, e2_hs = extract_relation_emb(model, testloader, 'test')
    test_pt, test_rt, test_f1t = evaluate(args, preds.cpu(), e1_hs.cpu(), e2_hs.cpu(), test_y_attr, test_y_e1,
                                          test_y_e2, test_y, test_idx2relation)

    print(f'[test] (macro) final precision: {test_pt:.4f}, recall: {test_rt:.4f}, f1 score: {test_f1t:.4f}')


if __name__ == "__main__":
    main()
