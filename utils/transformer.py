import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
import numpy as np

def drop_path(x, drop_prob=0., training=False, scale_by_keep=True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect implementation I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (tf.shape(x)[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
    binary_mask = tf.floor(random_tensor)

    if scale_by_keep:
        binary_mask /= keep_prob

    x = x * binary_mask
    return x


class DropPath(tf.keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def call(self, x, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        return drop_path(x, self.drop_prob, training, self.scale_by_keep)



class Mlp(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=tf.nn.gelu, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = tf.keras.layers.Dense(hidden_features)
        self.act = act_layer
        self.fc2 = tf.keras.layers.Dense(out_features)
        self.drop = tf.keras.layers.Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x, training=True)
        x = self.fc2(x)
        x = self.drop(x, training=True)
        return x


def window_partition(x, window_size):
    B, L, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
    a = x.shape[0]
    b = x.shape[1]
    c = x.shape[2]
    #print("B_partition", a)
    #print("L_partition", b)
    #print("C_partition", c)
    #if L % window_size != 0:
    #    raise ValueError("Length of the input should be divisible by window size.")
    x = tf.reshape(x, (B, b // window_size, window_size, c))
    #print("x.shape in window_partition", x.shape)
    windows = tf.reshape(x, (-1, window_size, c))
    #print("windows.shape in window_partition", windows.shape)
    return windows



def window_reverse(windows, window_size, L):
    #print("windonws shape:",windows.shape)
    K = windows.shape[0]
    #print(("K :", K))
    num_windows = tf.shape(windows)[0]
    B = num_windows // (L // window_size)
    ##print("B:", B)
    x = tf.reshape(windows, [B, L // window_size, window_size, -1])
    #print("x.shape:", x.shape)
    x = tf.reshape(x, [B, L, -1])
    #print("x.shape after window_reverse:", x.shape)
    return x



class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = tf.Variable(
            tf.zeros((2 * window_size - 1, num_heads)), trainable=True)  # 2*window_size - 1, nH

        coords_w = tf.range(self.window_size)
        relative_coords = coords_w[:, None] - coords_w[None, :]  # W, W
        relative_coords += self.window_size - 1  # shift to start from 0
        self.relative_position_index = tf.Variable(relative_coords, trainable=False)  # (W, W): range of 0 -- 2*(W-1)

        self.qkv = tf.keras.layers.Dense(dim * 3, use_bias=qkv_bias)
        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj = tf.keras.layers.Dense(dim)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)

        self.softmax = tf.nn.softmax

    @tf.function
    def call(self, x, mask=None):
        #B_, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        #x.set_shape((None, None, self.dim)) #128 # Ensure the last dimension is defined
        #print("x shape:", x.shape)
        B_, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        #print(B_)
        #print(N)
        #print(C)
        w = x.shape[0]
        z = x.shape[1]
        y = x.shape[2]
        #print("w shape:", w)
        #print("z shape:", z)
        #print("y shape:", y)
        qkv = self.qkv(x)
        #print("qkv shape:", qkv.shape)
        qkv = tf.reshape(qkv, (B_, z, 3, self.num_heads, y // self.num_heads))
        #print("qkv shape after reshape:", qkv.shape)
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = tf.split(qkv, 3, axis=0)
        #print("q shape:", q.shape)
        #print("k shape:", k.shape)
        #print("v shape:", v.shape)

        q = q * self.scale
        attn = tf.matmul(q, k, transpose_b=True)
        #print("attn", attn)

        relative_position_bias = tf.gather(tf.reshape(self.relative_position_bias_table, (-1,)), tf.reshape(self.relative_position_index, (-1,)))
        relative_position_bias = tf.reshape(relative_position_bias, (self.window_size, self.window_size, -1))
        relative_position_bias = tf.transpose(relative_position_bias, (2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, 0)

        if mask is not None:
            nW = mask.shape[0]
            attn = tf.reshape(attn, (B_ // nW, nW, self.num_heads, N, N)) + tf.expand_dims(tf.expand_dims(mask, 1), 0)
            attn = tf.reshape(attn, (-1, self.num_heads, N, N))
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        #print("attn before transpose ", attn.shape)
        x = tf.transpose(tf.matmul(attn, v), perm=[0, 2, 1, 3, 4])
        #print("x shape after transpose ", x.shape)
        x = tf.reshape(x, (B_, z, y))
        #print("x shape after reshape ", x.shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=tf.nn.gelu, norm_layer=tf.keras.layers.LayerNormalization):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window_size"

        self.norm1 = norm_layer()
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = tf.keras.layers.Dropout(drop_path) if drop_path > 0. else tf.keras.layers.Lambda(lambda x: x)
        self.norm2 = norm_layer()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        attn_mask = None
        self.attn_mask = tf.Variable(attn_mask, trainable=False) if attn_mask is not None else None
    @tf.function
    def call(self, x):
        B, L, C = x.shape
        #print("L", L)
        assert L >= self.window_size, f'input length ({L}) must be >= window size ({self.window_size})'
        assert L % self.window_size == 0, f'input length ({L}) must be divisible by window size ({self.window_size})'

        shortcut = x
        x = self.norm1(x)
        #print("x in SwinTransformerBlock", x.shape)

        # zero-padding shift
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=-self.shift_size, axis=1)
            shifted_x = tf.concat([shifted_x[:, :L-self.shift_size], tf.zeros_like(shifted_x[:, L-self.shift_size:])], axis=1)
            #shifted_x[:, -self.shift_size:] = 0.
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, C
        else:
            shifted_x = x
            # partition windows
        #print("shifted_x.shape", shifted_x.shape)
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, C
        #print("x_windows.shape after window_partition", x_windows.shape)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size, C
        #print("attn_windows:", attn_windows.shape)
        #print("L:", L)
        # merge windows
        shifted_x = window_reverse(attn_windows, self.window_size, L)  # (B, L, C)
        #print("shifted_x shape after window_reverse:", shifted_x.shape)

        # reverse zero-padding shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=self.shift_size, axis=1)
            x = tf.concat([x[:, :L-self.shift_size], tf.zeros_like(x[:, L-self.shift_size:])], axis=1)
            #x[:, :self.shift_size] = 0.  # remove invalid embs
        else:
            x = shifted_x

        x = shortcut + self.drop_path(x)
        #print("final x:", x.shape)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def flops(self, L):
        flops = 0
        # norm1
        flops += self.dim * L
        # W-MSA/SW-MSA
        nW = L / self.window_size
        flops += nW * self.attn.flops(self.window_size)
        # mlp
        flops += 2 * L * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * L
        return flops



class SwinTransformerLayer(tf.keras.layers.Layer):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=tf.keras.layers.LayerNormalization, use_checkpoint=False):
        super(SwinTransformerLayer, self).__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = []
        for i in range(depth):
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            block = SwinTransformerBlock(dim=dim,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         shift_size=shift_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias,
                                         qk_scale=qk_scale,
                                         drop=drop,
                                         attn_drop=attn_drop,
                                         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                         norm_layer=norm_layer)
            self.blocks.append(block)
    @tf.function
    def call(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = tf.recompute_grad(blk)(x)
            else:
                x = blk(x)
                #print("real final", x.shape)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops
    
    def get_config(self):
        config ={
            'dim': self.dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'window_size': self.window_size,

        }
        return config
    
class PatchMerging(tf.keras.layers.Layer):
    """Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
    """
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = tf.keras.layers.Dense(2 * dim, use_bias=False)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        
        tf.debugging.assert_equal(L, H * W, message="input feature has wrong size")
        tf.debugging.assert_equal(H % 2, 0, message=f"H size ({H}) is not even.")
        tf.debugging.assert_equal(W % 2, 0, message=f"W size ({W}) is not even.")
        
        x = tf.reshape(x, [B, H, W, C])
        
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        
        x = tf.concat([x0, x1, x2, x3], axis=-1)  # B H/2 W/2 4*C
        x = tf.reshape(x, [B, -1, 4 * C])  # B H/2*W/2 4*C
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_resolution": self.input_resolution,
            "dim": self.dim
        })
        return config

    
class PatchClassEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, n_patches, pos_emb=None, kernel_initializer='he_normal', **kwargs):
        super(PatchClassEmbedding, self).__init__(**kwargs)
        self.d_model = d_model
        self.n_tot_patches = n_patches + 1
        self.pos_emb = pos_emb

        self.class_embed = self.add_weight(shape=(1, 1, d_model), initializer=kernel_initializer, name="class_token")
        if self.pos_emb is not None:
            self.pos_emb = tf.convert_to_tensor(np.load(self.pos_emb))
            self.lap_position_embedding = tf.keras.layers.Embedding(input_dim=self.pos_emb.shape[0], output_dim=d_model)
        else:
            self.position_embedding = tf.keras.layers.Embedding(input_dim=self.n_tot_patches, output_dim=d_model)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        class_token = tf.tile(self.class_embed, [batch_size, 1, 1])
        x = tf.concat([class_token, inputs], axis=1)

        if self.pos_emb is None:
            positions = tf.range(start=0, limit=self.n_tot_patches, delta=1)
            position_embeddings = self.position_embedding(positions)
        else:
            pe = self.pos_emb
            pe = tf.reshape(pe, [1, -1])
            position_embeddings = self.lap_position_embedding(pe)

        encoded = x + position_embeddings
        return encoded

if __name__ == '__main__':
    dim = 128
    depth = 4
    num_heads = 4
    window_size = 1

    BS = 10
    L = window_size * 10
    x = tf.random.uniform((BS, L, dim))

    swin_t = SwinTransformerLayer(dim=dim, depth=depth, num_heads=num_heads, window_size=window_size)

    # forward
    out = swin_t(x)

    #print(x.shape)  # (BS, L, dim)
    #print(out.shape)  # (BS, L, dim)
