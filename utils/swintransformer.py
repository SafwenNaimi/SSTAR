
import tensorflow as tf
import numpy as np
import copy

def window_partition_1d(x, window_size):
    batch_size, seq_len, channels = x.shape
    windows = tf.reshape(x, (batch_size, seq_len // window_size, window_size, channels))
    return windows

def window_merge_1d(windows, window_size, seq_len, channels):
    batch_size = tf.shape(windows)[0]
    x = tf.reshape(windows, (batch_size, seq_len // window_size, window_size, channels))
    x = tf.reshape(x, (batch_size, seq_len, channels))
    return x

class WindowAttention1D(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, window_size, **kwargs):
        super(WindowAttention1D, self).__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = tf.keras.layers.Dense(dim * 3, use_bias=False)
        self.proj = tf.keras.layers.Dense(dim)

    def call(self, x, mask=None):
        batch_size, num_windows, window_size, _ = x.shape

        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (batch_size, num_windows, window_size, 3, self.num_heads, self.dim // self.num_heads))
        qkv = tf.transpose(qkv, [3, 0, 4, 1, 2, 5])
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ tf.transpose(k, [0, 1, 2, 4, 3])) * self.scale
        if mask is not None:
            nW = mask.shape[0]
            attn = tf.reshape(attn, (batch_size // nW, nW, self.num_heads, window_size, window_size)) + mask
            attn = tf.reshape(attn, (-1, self.num_heads, window_size, window_size))
            attn = tf.nn.softmax(attn, axis=-1)
        else:
            attn = tf.nn.softmax(attn, axis=-1)

        attn = tf.reshape(attn @ v, (batch_size, num_windows, window_size, self.dim))
        x = self.proj(attn)
        return x
    


class SwinTransformerBlock1D(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, window_size, shift_size=0, **kwargs):
        super(SwinTransformerBlock1D, self).__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention1D(dim, num_heads, window_size)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(4 * dim, activation=tf.nn.gelu),
            tf.keras.layers.Dense(dim)
        ])

    def call(self, x, mask=None):
        seq_len = tf.shape(x)[1]

        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=-self.shift_size, axis=1)
        else:
            shifted_x = x

        x_windows = window_partition_1d(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows, (-1, self.window_size, self.dim))
        attn_windows = self.attn(x_windows, mask)

        attn_windows = tf.reshape(attn_windows, (-1, seq_len // self.window_size, self.window_size, self.dim))
        shifted_x = window_merge_1d(attn_windows, self.window_size, seq_len, self.dim)

        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=self.shift_size, axis=1)
        else:
            x = shifted_x

        x = x + self.mlp(self.norm2(x))
        return x



class PatchMerging1D(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super(PatchMerging1D, self).__init__(**kwargs)
        self.dim = dim

    def call(self, x):
        batch_size, seq_len, channels = x.shape
        x = tf.reshape(x, (batch_size, seq_len // 2, 2, channels))
        x = tf.reshape(x, (batch_size, seq_len // 2, 2 * channels))
        x = tf.keras.layers.Dense(2 * self.dim)(x)
        return x


class SwinTransformer1D(tf.keras.models.Model):
    def __init__(self, seq_len, in_channels, num_classes, embed_dim, depths, num_heads, window_size=7, **kwargs):
        super(SwinTransformer1D, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.window_size = window_size

        self.patch_embed = tf.keras.Sequential([
            tf.keras.layers.Dense(embed_dim),
            tf.keras.layers.Reshape((-1, embed_dim))
        ])

        self.pos_embed = self.add_weight(shape=(1, (seq_len // window_size) * window_size, embed_dim), initializer='zeros', trainable=True)

        self.blocks = []
        for i in range(len(depths)):
            for j in range(depths[i]):
                self.blocks.append(SwinTransformerBlock1D(embed_dim * (2 ** i), num_heads[i], window_size, shift_size=(window_size // 2) if (j % 2 == 1) else 0))
            if i < len(depths) - 1:
                self.blocks.append(PatchMerging1D(embed_dim * (2 ** (i + 1))))

        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.head = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        x = self.patch_embed(x)
        x += self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = tf.reduce_mean(x, axis=1)
        x = self.head(x)
        return x

