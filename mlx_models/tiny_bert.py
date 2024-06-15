import math

import mlx.optimizers
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

import mlx.nn as nn
import mlx.core as mx

from pytorch_models.tiny_bert import Config


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = mx.zeros(hidden_size)
        self.beta = mx.zeros(hidden_size)
        self.variance_epsilon = variance_epsilon

    def __call__(self, x):
        u = mx.mean(x, -1, keepdims=True)
        s = mx.power((x - u), 2).mean(-1, keepdims=True)
        x = (x - u) / mx.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class PositionWiseMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(PositionWiseMLP, self).__init__()
        self.expansion = nn.Linear(hidden_size, intermediate_size)
        self.contraction = nn.Linear(intermediate_size, hidden_size)
        self.gelu = nn.GELU()

    def __call__(self, x):
        x = self.expansion(x)
        x = self.contraction(self.gelu(x))
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
        new_shape = tensor.shape[:-1] + (num_heads, attention_head_size)
        tensor = tensor.reshape(new_shape)
        return tensor.transpose(0, 2, 1, 3)

    @staticmethod
    def merge_heads(tensor, num_heads, attention_head_size):
        tensor = tensor.transpose(0, 2, 1, 3)
        new_shape = tensor.shape[:-2] + (num_heads * attention_head_size,)
        return tensor.reshape(new_shape)

    def attention(self, q, k, v, attention_mask):
        mask = attention_mask == 1
        mask = mx.expand_dims(mask, 1)
        mask = mx.expand_dims(mask, 2)

        weights = mx.matmul(q, k.transpose(0, 1, 3, 2))
        weights = weights / math.sqrt(self.attention_head_size)
        weights = mx.where(mask, weights, float(-1e9))

        logit = nn.softmax(weights, axis=-1)
        logit = self.dropout(logit)

        scores = mx.matmul(logit, v)
        return scores

    def __call__(self, x, attention_mask):
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

        self.token_embedding = nn.Embedding(
            self.config.vocab_size, self.config.hidden_size
        )
        self.position_embedding = nn.Embedding(
            self.config.max_position_embeddings, self.config.hidden_size
        )
        self.token_type_embedding = nn.Embedding(
            self.config.type_vocab_size, self.config.hidden_size
        )

        self.ln = LayerNorm(self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.layers = [
            Transformer(self.config) for _ in range(self.config.num_hidden_layers)
        ]

        self.pooler = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size), nn.Tanh()
        )

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None):
        position_ids = mx.arange(
            input_ids.shape[1],
            dtype=mx.int32,
        )
        position_ids = mx.expand_dims(position_ids, 0)
        position_ids = mx.repeat(position_ids, input_ids.shape[0], 0)

        if token_type_ids is None:
            token_type_ids = mx.zeros_like(input_ids)

        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(position_ids)
            + self.token_type_embedding(token_type_ids)
        )
        x = self.dropout(self.ln(x))

        for layer in self.layers:
            x = layer(x, attention_mask)

        o = self.pooler(x[:, 0])
        return x, o


class BertFineTuneTask(nn.Module):
    def __init__(self, num_labels, bert_config):
        super(BertFineTuneTask, self).__init__()
        self.bert = MiniBert(bert_config)
        self.linear1 = nn.Linear(4 * self.bert.config.hidden_size, num_labels)

    def __call__(self, input_ids, attention_mask):
        input_ids = input_ids.reshape(input_ids.shape[0] * 2, input_ids.shape[2])
        attention_mask = attention_mask.reshape(
            attention_mask.shape[0] * 2, attention_mask.shape[2]
        )
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        embeds = bert_output[1].reshape(
            bert_output[1].shape[0] // 2, 2, bert_output[1].shape[1]
        )

        set_1_embeds = embeds[:, 0]
        set_2_embeds = embeds[:, 1]
        concat = mx.concatenate(
            (
                set_1_embeds,
                set_2_embeds,
                mx.abs(set_1_embeds - set_2_embeds),
                mx.multiply(set_1_embeds, set_2_embeds),
            ),
            axis=-1,
        )

        lin = self.linear1(concat)
        return lin


def tokenize_sentence_pair_dataset(dataset, tokenizer, max_length=512):
    tokenized_dataset = []
    for i in range(0, len(dataset[0])):
        tokenized_dataset.append(
            (
                tokenizer(
                    [dataset[0][i], dataset[1][i]],
                    return_tensors="np",
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                ),
                np.array(dataset[2][i]),
            )
        )
    return tokenized_dataset


def train_loop(model, optimizer, train_dataloader, num_epochs, val_dataloader):
    mx.eval(model.parameters())
    state = [model.state, optimizer.state]

    def loss_fn(model, x, y):
        input_ids, attention_mask = x
        embeds = model(input_ids, attention_mask)
        losses = nn.losses.cross_entropy(embeds, y, reduction="mean")
        return losses

    def step(inputs, targets):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, inputs, targets)
        optimizer.update(model, grads)
        return loss

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} out of {num_epochs}")
        epoch_loss = 0
        for item in iter(train_dataloader):
            # From Pytorch tensor to numpy, there is no copy.
            ids = mx.array(item[0]["input_ids"].numpy())
            mask = mx.array(item[0]["attention_mask"].numpy())
            truth = mx.array(item[1].numpy())

            loss = step((ids, mask), truth)
            mx.eval(state)
            epoch_loss += loss.item()

        print(f"epoch loss: {epoch_loss}")
        dev_loss = 0

        for item in iter(val_dataloader):
            ids = mx.array(item[0]["input_ids"].numpy())
            mask = mx.array(item[0]["attention_mask"].numpy())
            truth = mx.array(item[1].numpy())

            loss = loss_fn(model, (ids, mask), truth)
            dev_loss += loss.item()
        print(f"validation loss: {dev_loss}")
        print()


def train(num_epochs, batch_size, num_labels, bert_config, lr, dataset):
    mx.set_default_device(mx.gpu)
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
    optimizer = mlx.optimizers.AdamW(learning_rate=lr)
    train_loop(bert_model, optimizer, train_dataloader, num_epochs, val_dataloader)
