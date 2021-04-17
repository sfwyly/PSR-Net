

"""

    @Title: Progressive Semantic Reasoning for Image Inpainting
    @Author: Jin et al.
    @Code: Junjie Jin

"""

from config import *
from .cfr import *
import tensorflow.keras.models as models


class Bottleneck(layers.Layer):
    """
        Bottleneck

    """
    def __init__(self, output_dim, strides=1, training=True, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.conv1 = layers.Conv2D(output_dim // 4, kernel_size=1, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(output_dim // 4, kernel_size=3, strides=strides, padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(output_dim, kernel_size=1, padding="same", use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.training = training

    def build(self, input_shape):
        super(Bottleneck, self).build(input_shape)

    def call(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out, training=self.training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=self.training)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, training=self.training)

        out = out + residual
        out = self.relu(out)

        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class PartialConv(layers.Layer):
    """
        Partial Convolution
    """
    def __init__(self, kernel=3, dilation_rate=1, strides=2, in_channels=64, out_channels=64, activation="relu",
                 flag=True, mul=True, training=True, **kwargs):
        super(PartialConv, self).__init__(**kwargs)
        self.slide_window = kernel ** 2
        self.kernel = kernel
        self.strides = strides
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.dilation_rate = dilation_rate
        self.dense2 = tf.keras.layers.Conv2D(self.out_channels, kernel_size=self.kernel,
                                             dilation_rate=self.dilation_rate, strides=self.strides, padding="same",
                                             use_bias=False, trainable=True)  #
        self.flag = flag
        self.mul = mul
        # self.bias = self.add_weight(shape=(self.channels,),initializer=tf.constant_initializer(0.0),trainable=True)
        if (self.flag):
            self.leaky_relu = tf.nn.leaky_relu
            self.bn = layers.BatchNormalization()
        self.weights_updater = tf.ones((self.kernel, self.kernel, self.in_channels, self.out_channels))
        self.training = training

    def call(self, input, mask):

        update_mask = tf.nn.conv2d(mask, self.weights_updater, strides=self.strides, padding="SAME")
        mask_ratio = (self.slide_window * self.in_channels) / (update_mask + 1e-8)
        update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
        mask_ratio = mask_ratio * update_mask

        output = self.dense2(input)  # -self.bias
        if (self.mul):
            output = output * mask_ratio
            # output = self.relu(output*mask_ratio+self.bias)
            output = output * update_mask
        if (self.flag):
            return self.leaky_relu(self.bn(output, training=self.training)), update_mask
        return output, update_mask

class PFRNet(models.Model):
    """
        Our Progressive Semantic Reasoning Module
        Three Share Parameters modules

    """
    def __init__(self,training = True):
        super(PFRNet, self).__init__()

        self.training = training
        self.Pconv01 = PartialConv(in_channels=3, out_channels=64, kernel=7, strides=2, flag=False,training=training)
        self.Pconv02 = PartialConv(in_channels=64, out_channels=64, kernel=7, strides=1,training=training)

        self.Pconv11 = PartialConv(in_channels=64, out_channels=128, kernel=7, strides=2,training=training)
        self.Pconv12 = PartialConv(in_channels=128, out_channels=256, kernel=5, strides=2, flag=False,training=training)

        self.u_Pconv11 = PartialConv(in_channels=384, out_channels=128, kernel=3, strides=1,training=training)
        self.u_Pconv10 = PartialConv(in_channels=192, out_channels=64, kernel=3, strides=1,training=training)

        self.up = layers.UpSampling2D(size=(2, 2))

        self.Pconv23 = PartialConv(in_channels=256, out_channels=512, kernel=3, strides=2, mul=False,training=training)
        self.Pconv24 = PartialConv(in_channels=512, out_channels=512, kernel=3, strides=2, mul=False,training=training)

        self.u_Pconv22 = PartialConv(in_channels=768, out_channels=256, kernel=3, strides=1, mul=False,training=training)
        self.u_Pconv23 = PartialConv(in_channels=1024, out_channels=512, kernel=3, strides=1, mul=False,training=training)

        self.Pconv35 = PartialConv(in_channels=512, out_channels=512, kernel=3, strides=2, mul=False,training=training)
        self.Pconv36 = PartialConv(in_channels=512, out_channels=512, kernel=3, strides=2, mul=False,training=training)

        self.u_Pconv34 = PartialConv(in_channels=1024, out_channels=512, kernel=3, strides=1, mul=False,training=training)
        self.u_Pconv35 = PartialConv(in_channels=1024, out_channels=512, kernel=3, strides=1, mul=False,training=training)

        self.atn = AttentionModule()

        self.conv = layers.Conv2D(64, kernel_size=1, strides=1, padding="same", use_bias=False)
        self.conv2 = layers.Conv2D(64, kernel_size=1, activation="sigmoid", strides=1, padding="same", use_bias=False)

        self.Tconv = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.tail1 = PartialConv(in_channels=67, out_channels=32, kernel=3, strides=1, flag=False,training=training)
        self.tail2 = Bottleneck(output_dim=32, strides=1)
        self.out = layers.Conv2D(3, kernel_size=3, activation="sigmoid", strides=1, padding="same")

    def call(self, input, mask):

        self.atn.att.att_scores_prev = None
        self.atn.att.masks_prev = None

        x, m = self.Pconv01(input, mask)
        x = tf.nn.relu(x)
        x, m = self.Pconv02(x, m)

        x1_0, m1_0 = x, m
        x1_1, m1_1 = self.Pconv11(x1_0, m1_0)
        x1_1 = tf.nn.relu(x1_1)
        x1_2, m1_2 = self.Pconv12(x1_1, m1_1)
        x1_2 = tf.nn.relu(x1_2)

        u_x1_1, u_m1_1 = self.u_Pconv11(tf.concat([self.up(x1_2), x1_1], axis=-1),
                                        tf.concat([self.up(m1_2), m1_1], axis=-1))
        u_x1_0, u_m1_0 = self.u_Pconv10(tf.concat([self.up(u_x1_1), x1_0], axis=-1), tf.concat([self.up(u_m1_1), m1_0],
                                                                                               axis=-1))

        x2_0, m2_0 = u_x1_0 * u_m1_0, u_m1_0

        x2_1, m2_1 = self.Pconv11(x2_0, m2_0)
        x2_1 = tf.nn.relu(x2_1)
        x2_2, m2_2 = self.Pconv12(x2_1, m2_1)
        x2_2 = tf.nn.relu(x2_2)
        x2_3 = self.atn(x1_2, x2_2, m2_2)

        x2_3, m2_3 = self.Pconv23(x2_3, m2_2)
        x2_3 = tf.nn.relu(x2_3)
        x2_4, m2_4 = self.Pconv24(x2_3, m2_3)
        x2_4 = tf.nn.relu(x2_4)

        u_x2_3, u_m2_3 = self.u_Pconv23(tf.concat([self.up(x2_4), x2_3], axis=-1), tf.concat([self.up(m2_4), m2_3],
                                                                                             axis=-1))
        u_x2_2, u_m2_2 = self.u_Pconv22(tf.concat([self.up(u_x2_3), x2_2], axis=-1), tf.concat([self.up(u_m2_3), m2_2],
                                                                                               axis=-1))
        u_x2_1, u_m2_1 = self.u_Pconv11(tf.concat([self.up(u_x2_2), x2_1], axis=-1),
                                        tf.concat([self.up(m2_2), m2_1], axis=-1))
        u_x2_0, u_m2_0 = self.u_Pconv10(tf.concat([self.up(u_x2_1), x2_0], axis=-1), tf.concat([self.up(u_m2_1), m2_0],
                                                                                               axis=-1))
        x3_0, m3_0 = u_x2_0 * u_m2_0, u_m2_0

        x3_1, m3_1 = self.Pconv11(x3_0, m3_0)
        x3_1 = tf.nn.relu(x3_1)
        x3_2, m3_2 = self.Pconv12(x3_1, m3_1)
        x3_2 = tf.nn.relu(x3_2)
        x3_3 = self.atn(x2_2, x3_2, m3_2)

        x3_3, m3_3 = self.Pconv23(x3_3, m3_2)
        x3_3 = tf.nn.relu(x3_3)
        x3_4, m3_4 = self.Pconv24(x3_3, m3_3)
        x3_4 = tf.nn.relu(x3_4)
        x3_5, m3_5 = self.Pconv35(x3_4, m3_4)
        x3_5 = tf.nn.relu(x3_5)
        x3_6, m3_6 = self.Pconv36(x3_5, m3_5)
        x3_6 = tf.nn.relu(x3_6)

        u_x3_5, u_m3_5 = self.u_Pconv35(tf.concat([self.up(x3_6), x3_5], axis=-1),
                                        tf.concat([self.up(m3_6), m3_5], axis=-1))
        u_x3_4, u_m3_4 = self.u_Pconv34(tf.concat([self.up(u_x3_5), x3_4], axis=-1), tf.concat([self.up(u_m3_5), m3_4],
                                                                                               axis=-1))
        u_x3_3, u_m3_3 = self.u_Pconv23(tf.concat([self.up(u_x3_4), x3_3], axis=-1), tf.concat([self.up(u_m3_4), m3_3],
                                                                                               axis=-1))
        u_x3_2, u_m3_2 = self.u_Pconv22(tf.concat([self.up(u_x3_3), x3_2], axis=-1), tf.concat([self.up(u_m3_3), m3_2],
                                                                                               axis=-1))
        u_x3_1, u_m3_1 = self.u_Pconv11(tf.concat([self.up(u_x3_2), x3_1], axis=-1),
                                        tf.concat([self.up(m3_2), m3_1], axis=-1))
        u_x3_0, u_m3_0 = self.u_Pconv10(tf.concat([self.up(u_x3_1), x3_0], axis=-1), tf.concat([self.up(u_m3_1), m3_0],
                                                                                               axis=-1))

        x3, m3 = u_x3_0 * u_m3_0, u_m3_0

        c_x = tf.concat([x2_0, x3_0, x3], axis=-1)
        c_m = tf.concat([m2_0, m3_0, m3], axis=-1)
        c_x = self.conv(c_x)
        c_m = self.conv2(tf.concat([c_x, c_m], axis=-1))

        c_x = c_x * c_m
        x3, m3 = c_x, c_m

        x4 = self.Tconv(x3)
        x4 = tf.nn.leaky_relu(self.bn3(x4,training=self.training))
        m4 = tf.image.resize(m3, (m3.shape[1] * 2, m3.shape[2] * 2), "bilinear")
        x5 = tf.concat([input, x4], axis=-1)  # 这里是c
        m5 = tf.concat([mask, m4], axis=-1)

        x5, _ = self.tail1(x5, m5)
        x5 = tf.nn.leaky_relu(x5)
        x6 = self.tail2(x5)
        x6 = tf.concat([x5, x6], axis=-1)
        output = self.out(x6)

        return output

def build_model(mode = "training"):


    inputs = layers.Input(batch_shape=(batch_size, image_size, image_size, 3))
    masks = layers.Input(batch_shape=(batch_size, image_size, image_size, 1))

    if(mode=="tuning"):
        outputs = PFRNet(training=False)(inputs, masks)
    else:
        outputs = PFRNet(training=True)(inputs, masks)

    return models.Model(inputs,outputs)