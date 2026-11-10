import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=0.5):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


def dot_attention(q, k, v, v_mask=None, dropout=None):
    attention_weights = torch.matmul(q, k.transpose(-1, -2))
    extended_v_mask = None
    if v_mask is not None:
        extended_v_mask = (1.0 - v_mask.unsqueeze(1)) * -100000.0
    if extended_v_mask is not None:
        attention_weights += extended_v_mask
    attention_weights = F.softmax(attention_weights, -1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    output = torch.matmul(attention_weights, v).squeeze(-2)
    return output


def emb_proj(fp, fc):
    # fp 和 fc 都是batch * tokens 的二维tensor

    x_y = (fp * fc).sum(dim=-1)
    x_y_y = x_y.view(-1, 1) * fc
    y_length = (fc * fc).sum(dim=-1).view(-1, 1)
    proj = x_y_y / y_length.clamp_min(torch.finfo(fc.dtype).eps)

    return proj


def extract_entity(sequence_output, e_mask):
    extended_e_mask = e_mask.unsqueeze(-1)
    extended_e_mask = extended_e_mask.float() * sequence_output
    extended_e_mask, _ = extended_e_mask.max(dim=-2)
    # extended_e_mask = torch.stack([sequence_output[i,j,:] for i,j in enumerate(e_mask)])
    return extended_e_mask.float()


# class REMatchingModel(nn.Module):
class REMatchingModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.relation_emb_dim = config.hidden_size
        self.margin = float(config.margin)
        self.alpha = config.alpha
        #self.bert = AutoModel.from_pretrained(args.pretrained_model)
        self.bert = BertModel(config)
        self.fclayer = nn.Linear(self.relation_emb_dim * 2, self.relation_emb_dim)
        self.code = nn.Embedding(1, self.relation_emb_dim)
        self.classifier = nn.Sequential(GradientReversal(),
                                        nn.Linear(self.relation_emb_dim, self.relation_emb_dim),
                                        nn.Tanh(),
                                        nn.Linear(self.relation_emb_dim, self.num_labels)
                                        )
        self.rev_loss = nn.CrossEntropyLoss()
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            e1_mask=None,
            e2_mask=None,
            marked_head=None,
            marked_tail=None,
            input_relation_emb=None,
            input_relation_head_emb=None,
            input_relation_tail_emb=None,
            labels=None,
            num_neg_sample=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]  # Sequence of hidden-states of the last layer
        common_emb = dot_attention(self.code(torch.zeros(1, dtype=torch.long, device=input_ids.device)), sequence_output.detach(),
                                   sequence_output.detach(), attention_mask)

        e1_h = extract_entity(sequence_output, marked_head)
        e2_h = extract_entity(sequence_output, marked_tail)
        e1_mark = extract_entity(sequence_output, e1_mask)
        e2_mark = extract_entity(sequence_output, e2_mask)
        pooled_output = torch.cat([e1_mark, e2_mark], dim=-1)
        sentence_embeddings = self.fclayer(pooled_output)

        sentence_proj = emb_proj(sentence_embeddings, common_emb)
        sentence_embeddings = emb_proj(sentence_embeddings, sentence_embeddings - sentence_proj)

        sentence_embeddings = torch.tanh(sentence_embeddings)
        e1_h = torch.tanh(e1_h)
        e2_h = torch.tanh(e2_h)
        outputs = (outputs,)
        if labels is not None:
            rev_logit = self.classifier(common_emb)
            revloss = self.rev_loss(rev_logit.view(-1, self.num_labels), labels.view(-1))
            gamma = self.margin
            loss = sentence_embeddings.new_zeros(())
            zeros = sentence_embeddings.new_zeros(())
            for a, b in enumerate(zip(sentence_embeddings, e1_h, e2_h)):
                max_val = sentence_embeddings.new_zeros(())
                matched_sentence_pair = input_relation_emb[a]
                matched_head_pair = input_relation_head_emb[a]
                matched_tail_pair = input_relation_tail_emb[a]

                pos_s = torch.cosine_similarity(matched_sentence_pair, b[0], dim=-1)
                pos_h = torch.cosine_similarity(matched_head_pair, b[1], dim=-1).max()
                pos_t = torch.cosine_similarity(matched_tail_pair, b[2], dim=-1).max()
                pos = (1 - 2 * self.alpha) * pos_s + self.alpha * pos_h + self.alpha * pos_t

                # randomly sample relation_emb
                rand = random.sample(range(len(input_relation_emb)), min(num_neg_sample, len(input_relation_emb)))
                if not rand:
                    loss = loss + torch.maximum(zeros, max_val - pos + gamma)
                    continue
                neg_relation_emb = torch.stack([input_relation_emb[i] for i in rand])
                neg_relation_head_emb = torch.stack([input_relation_head_emb[i] for i in rand])
                neg_relation_tail_emb = torch.stack([input_relation_tail_emb[i] for i in rand])
                for i, j in enumerate(zip(neg_relation_emb, neg_relation_head_emb, neg_relation_tail_emb)):
                    tmp_s = torch.cosine_similarity(b[0], j[0], dim=-1)
                    tmp_h = torch.cosine_similarity(b[1], j[1], dim=-1).max()
                    tmp_t = torch.cosine_similarity(b[2], j[2], dim=-1).max()
                    tmp = (1 - 2 * self.alpha) * tmp_s + self.alpha * tmp_h + self.alpha * tmp_t
                    if tmp > max_val:
                        if (matched_sentence_pair == j[0]).all():
                            continue
                        else:
                            max_val = tmp

                neg = max_val
                loss = loss + torch.maximum(zeros, neg - pos + gamma)
            outputs = (loss, revloss)
            #outputs = (loss, )
        return outputs, sentence_embeddings, e1_h, e2_h
