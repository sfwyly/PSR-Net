
"""

    @Author: Junjie Jin
    @Code: Junjie Jin
    @Description:
        Cross Feature Reconstruction
        transfer attention from high-level to low-level features

"""

import tensorflow as tf
import tensorflow.keras.layers as layers

class AtnConv(layers.Layer):
    def __init__(self, input_channels=256, output_channels=256, groups=4, ksize=3, stride=1, rate=1, softmax_scale=10,
                 fuse=True, rates=[1, 2, 4, 8]):
        super(AtnConv, self).__init__()

        self.kernel = ksize
        self.strides = stride
        self.rate = rate
        self.softmax_scale = softmax_scale
        self.groups = groups
        self.fuse = fuse
        self.att_scores_prev = None
        self.masks_prev = None
        self.ratio = tf.ones(1)
        # if(self.fuse):
        #    self.group_blocks = []
        #    for i in range(groups):
        #        self.group_blocks.append(
        #            models.Sequential([layers.Conv2D(output_channels//groups,kernel_size=3,dilation_rate=rates[i],padding="same"),layers.ReLU()])
        #        )

    # x1 lower-level  x2: high-level
    def call(self, x1, x2, mask):

        x1s = x1.shape
        x2s = x2.shape
        bz, height, width, channels = x1s
        kernel = 2 * self.rate
        raw_w = tf.image.extract_patches(x1, [1, self.kernel, self.kernel, 1],
                                         [1, self.rate * self.strides, self.rate * self.strides, 1], [1, 1, 1, 1],
                                         padding='SAME')
        raw_w = tf.reshape(raw_w, [x1s[0], -1, self.kernel, self.kernel, x1s[-1]])
        raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
        raw_w_groups = tf.split(raw_w, x1s[0], axis=0)

        f_groups = tf.split(x2, x2s[0], axis=0)

        w = tf.image.extract_patches(x2, [1, self.kernel, self.kernel, 1], [1, self.strides, self.strides, 1],
                                     [1, 1, 1, 1], padding='SAME')
        w = tf.reshape(w, [x2s[0], -1, self.kernel, self.kernel, x2s[-1]])
        w = tf.transpose(w, [0, 2, 3, 4, 1])
        w_groups = tf.split(w, x2s[0], axis=0)

        ms = mask.shape
        if (mask is not None):
            mask = tf.image.resize(mask, x2s[1:3], "bilinear")
        else:
            mask = tf.zeros((x2s[0], x2s[1], x2s[2], x2s[3]))
        m = tf.image.extract_patches(mask, [1, self.kernel, self.kernel, 1], [1, self.strides, self.strides, 1],
                                     [1, 1, 1, 1], padding='SAME')
        m = tf.reshape(m, [ms[0], -1, self.kernel, self.kernel, ms[-1]])
        m = tf.transpose(m, [0, 2, 3, 4, 1])  # b k k c hw
        m = tf.cast(tf.equal(tf.reduce_mean(m, axis=[1, 2, 3], keepdims=True), 1.), tf.float32)
        mm = tf.squeeze(m, axis=1)  # b 1 1 hw
        mm_groups = tf.split(mm, ms[0], axis=0)

        y = []
        att_score = []
        scale = self.softmax_scale
        for i in range(bz):

            xi, wi, raw_wi, mi = x2[i:i + 1], w[i], raw_w[i], mm[i]  # k k c hw
            escape_NaN = 1e-4
            wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0, 1, 2])), escape_NaN)  # k k c hw

            yi = tf.nn.conv2d(xi, wi_normed, strides=1, padding="SAME")  # 1 h w  hw

            yi = tf.reshape(yi, [1, x2s[1], x2s[2], x2s[1] // self.strides * x2s[2] // self.strides])

            yi = tf.nn.avg_pool2d(yi, 3, 1, padding="SAME") * 9

            attention_scores = tf.nn.softmax(yi * scale, axis=-1)

            if (self.att_scores_prev is not None):
                attention_scores = (self.att_scores_prev[i:i + 1] * self.masks_prev[i:i + 1] + attention_scores * (
                            tf.abs(self.ratio) + 1e-7)) / (self.masks_prev[i:i + 1] + (tf.abs(self.ratio) + 1e-7))
                # pass
            att_score.append(attention_scores)
            yi = tf.nn.conv2d_transpose(attention_scores, raw_wi, tf.concat([[1], x1s[1:]], axis=0),
                                        strides=[1, self.rate, self.rate, 1], padding="SAME")  # /4.
            y.append(yi)

        y = tf.concat(y, axis=0)
        self.att_scores_prev = tf.reshape(tf.concat(att_score, axis=0), [bz, height, width, height * width])
        self.masks_prev = tf.reshape(mm, [bz, 1, 1, height * width])
        # if(self.fuse):
        #    tmp = []
        #    for i in range(self.groups):
        #        tmp.append(self.group_blocks[i](y))
        #    y = tf.concat(tmp,axis= -1)

        return y

class AttentionModule(layers.Layer):

    def __init__(self, inchannel=256, patch_size_list=[1, 2, 4, 8], **kwargs):
        super(AttentionModule, self).__init__(**kwargs)
        self.att = AtnConv(256, 256)
        self.num_of_modules = len(patch_size_list)
        self.combiner = layers.Conv2D(inchannel, kernel_size=1, padding="same")
        self.blocks = []
        for rate in patch_size_list:
            self.blocks.append(
                layers.Conv2D(inchannel, kernel_size=3, activation="relu", dilation_rate=rate, padding="same"))
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()

        self.mlp1 = layers.Dense(max(inchannel // 8, 8), activation="relu")
        self.mlp2 = layers.Dense(inchannel * len(patch_size_list))

        # self.dropout1 = layers.Dropout(0.5)
        # self.dropout2 = layers.Dropout(0.5)

    def build(self, input_shape):
        super(AttentionModule, self).build(input_shape)

    def call(self, x1, x2, mask):
        bz, h, w, c = x1.shape
        outputs = self.att(x1, x2, mask)
        # return outputs
        patch_list = []
        for block in self.blocks:
            patch_list.append(block(outputs)[tf.newaxis, ...])
        patch_block = tf.concat(patch_list, axis=0)
        block_sum = tf.reduce_sum(patch_block, axis=0)

        avg_feat = self.avg_pool(block_sum)
        max_feat = self.max_pool(block_sum)
        avg_feat = self.mlp1(avg_feat)
        # avg_feat = self.dropout1(avg_feat)
        avg_feat = self.mlp2(avg_feat)

        max_feat = self.mlp1(max_feat)
        # max_feat = self.dropout2(max_feat)
        max_feat = self.mlp2(max_feat)

        feat = avg_feat + max_feat

        feat = tf.reshape(feat, [bz, 1, 1, c, self.num_of_modules])
        feat = tf.transpose(feat, [4, 0, 1, 2, 3])
        feat = tf.nn.softmax(feat, axis=0)
        outputs = tf.reduce_sum(patch_block * feat, axis=0)

        outputs = tf.concat([outputs, x2], axis=-1)
        outputs = self.combiner(outputs)

        return outputs