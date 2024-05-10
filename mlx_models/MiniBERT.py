import mlx.nn as nn
import mlx.core as mx


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
