from collections import OrderedDict

import torch
import torch.nn as nn
import math
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

device = torch.device("mps")


class Config(object):
    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = dropout_prob
        self.attention_probs_dropout_prob = dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, dict_object):
        config = Config(vocab_size=None)
        for key, value in dict_object.items():
            config.__dict__[key] = value
        return config


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class PositionWiseMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(PositionWiseMLP, self).__init__()
        self.expansion = nn.Linear(hidden_size, intermediate_size)
        self.contraction = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        x = self.expansion(x)
        x = self.contraction(torch.nn.functional.gelu(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.attn_out = nn.Linear(config.hidden_size, config.hidden_size)
        self.ln1 = LayerNorm(config.hidden_size)

        self.mlp = PositionWiseMLP(config.hidden_size, config.intermediate_size)
        self.ln2 = LayerNorm(config.hidden_size)

    @staticmethod
    def split_heads(tensor, num_heads, attention_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attention_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)

    @staticmethod
    def merge_heads(tensor, num_heads, attention_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attention_head_size,)
        return tensor.view(new_shape)

    def attention(self, q, k, v, attention_mask):
        mask = attention_mask == 1
        mask = mask.unsqueeze(1).unsqueeze(2)

        weights = torch.matmul(q, k.transpose(-2, -1))
        weights = weights / math.sqrt(self.attention_head_size)
        weights = torch.where(
            mask, weights, torch.tensor(float(-1e9), device=attention_mask.device)
        )

        logit = torch.nn.functional.softmax(weights, dim=-1)
        logit = self.dropout(logit)

        scores = torch.matmul(logit, v)
        return scores

    def forward(self, x, attention_mask):
        q, k, v = self.query(x), self.key(x), self.value(x)

        q = self.split_heads(q, self.num_attention_heads, self.attention_head_size)
        k = self.split_heads(k, self.num_attention_heads, self.attention_head_size)
        v = self.split_heads(v, self.num_attention_heads, self.attention_head_size)

        a = self.attention(q, k, v, attention_mask)
        a = self.merge_heads(a, self.num_attention_heads, self.attention_head_size)
        a = self.attn_out(a)
        a = self.dropout(a)
        a = self.ln1(a + x)

        m = self.mlp(a)
        m = self.dropout(m)
        m = self.ln2(m + a)

        return m


class MiniBert(nn.Module):
    def __init__(self, config_dict):
        super(MiniBert, self).__init__()
        self.config = Config.from_dict(config_dict)
        self.embeddings = nn.ModuleDict(
            {
                "token": nn.Embedding(
                    self.config.vocab_size, self.config.hidden_size, padding_idx=0
                ),
                "position": nn.Embedding(
                    self.config.max_position_embeddings, self.config.hidden_size
                ),
                "token_type": nn.Embedding(
                    self.config.type_vocab_size, self.config.hidden_size
                ),
            }
        )

        self.ln = LayerNorm(self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.layers = nn.ModuleList(
            [Transformer(self.config) for _ in range(self.config.num_hidden_layers)]
        )

        # This is a pooling layer for Bert's last layer.
        self.pooler = nn.Sequential(
            OrderedDict(
                [
                    (
                        "dense",
                        nn.Linear(self.config.hidden_size, self.config.hidden_size),
                    ),
                    ("activation", nn.Tanh()),
                ]
            )
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        position_ids = torch.arange(
            input_ids.size(1), dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        x = (
            self.embeddings.token(input_ids)
            + self.embeddings.position(position_ids)
            + self.embeddings.token_type(token_type_ids)
        )
        x = self.dropout(self.ln(x))

        for layer in self.layers:
            x = layer(x, attention_mask)

        o = self.pooler(x[:, 0])
        return x, o


# BERT fine-tune task: https://arxiv.org/pdf/1705.02364
class BertFineTuneTask(nn.Module):
    def __init__(self, num_labels, bert_config):
        super(BertFineTuneTask, self).__init__()
        self.bert = MiniBert(bert_config)
        self.linear1 = torch.nn.Linear(3 * self.bert.config.hidden_size, num_labels)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, ground_truth):
        input_ids = input_ids.view(input_ids.size(0) * 2, input_ids.size(2))
        attention_mask = attention_mask.view(
            attention_mask.size(0) * 2, attention_mask.size(2)
        )
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        embeds = bert_output[1].view(
            bert_output[1].size(0) // 2, 2, bert_output[1].size(1)
        )

        concat = torch.cat(
            (embeds[:, 0], embeds[:, 1], torch.abs(embeds[:, 0] - embeds[:, 1])), dim=-1
        )
        lin = self.linear1(concat)
        loss = self.loss_func(lin, ground_truth)
        return loss


def tokenize_sentence_pair_dataset(dataset, tokenizer, max_length=512):
    tokenized_dataset = []
    for i in range(0, len(dataset[0])):
        tokenized_dataset.append(
            (
                tokenizer(
                    [dataset[0][i], dataset[1][i]],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                ),
                torch.tensor(dataset[2][i], dtype=torch.float32),
            )
        )

    return tokenized_dataset


def train_loop(model, optimizer, train_dataloader, num_epochs, val_dataloader):
    model.to(device)
    model.train(True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} out of {num_epochs}")
        epoch_loss = 0
        for item in iter(train_dataloader):
            optimizer.zero_grad()

            ids = item[0]["input_ids"].to(device)
            mask = item[0]["attention_mask"].to(device)
            truth = item[1].to(device)

            loss = model(ids, mask, truth)
            epoch_loss += loss
            loss.backward()
            optimizer.step()

        print(f"epoch loss: {epoch_loss}")
        dev_loss = 0
        with torch.no_grad():
            for item in iter(val_dataloader):
                ids = item[0]["input_ids"].to(device)
                mask = item[0]["attention_mask"].to(device)
                truth = item[1].to(device)

                loss = model(ids, mask, truth)
                dev_loss += loss
        print(f"validation loss: {dev_loss}")
        print()


def train(num_epochs, batch_size, num_labels, bert_config, lr, dataset):
    model_name = "prajjwal1/bert-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_train = tokenize_sentence_pair_dataset(
        dataset["train"][:50000], tokenizer, max_length=128
    )
    tokenized_val = tokenize_sentence_pair_dataset(
        dataset["dev"], tokenizer, max_length=128
    )

    train_dataloader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(tokenized_val, batch_size=batch_size, shuffle=False)

    bert_model = BertFineTuneTask(num_labels, bert_config)
    optimizer = torch.optim.AdamW(bert_model.parameters(), lr=lr)
    train_loop(bert_model, optimizer, train_dataloader, num_epochs, val_dataloader)
