
"""
    @Author: Junjie Jin
    @Code: Junjie Jin
    @Description: all loss function

"""

import tensorflow as tf
from config import *

style_layers = ['block1_pool', 'block2_pool', 'block3_pool']
content_layers = []
# num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

    return x_var, y_var


def total_variation_loss(image, mask_list):

    kernel = tf.ones((3, 3, mask_list.shape[3], mask_list.shape[3]))
    dilated_mask = tf.nn.conv2d(1 - mask_list, kernel, strides=[1, 1, 1, 1], padding="SAME")
    dilated_ratio = 9. * 3 / (dilated_mask + 10e-6)
    dilated_mask = tf.cast(tf.greater(dilated_mask, 0), "float32")
    image = dilated_mask * image
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_mean(tf.abs(x_deltas)) + tf.reduce_mean(tf.abs(y_deltas))


def border_loss(image, mask_list):
    x_var = (image[:, :, 1:, :] - image[:, :, :-1, :]) * (mask_list[:, :, 1:, :] - mask_list[:, :, :-1, :])
    y_var = (image[:, 1:, :, :] - image[:, :-1, :, :]) * (mask_list[:, 1:, :, :] - mask_list[:, :-1, :, :])
    return tf.reduce_mean(tf.abs(x_var)) + tf.reduce_mean(tf.abs(y_var))


def vgg_layers(layer_names):

    vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2] * input_shape[3], tf.float32)
    return result / (num_locations)

class StyleContentModel(tf.keras.models.Model):

    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
        # self.vgg.summary()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def call(self, inputs):
        # inputs = tf.image.resize(inputs, (224, 224))
        outputs = self.vgg(inputs)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        # style层的原始输出
        perceptual_dict = {style_name: value
                           for style_name, value
                           in zip(self.style_layers, style_outputs)}
        style_dict = {style_name: gram_matrix(value)
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'perceptual': perceptual_dict, 'style': style_dict}


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=10e-8, clip_value_max=1.0 - 10e-8)

def style_content_loss(outputs, style_targets):
    # print("样式")
    style_outputs = outputs['style']
    style_targets = style_targets['style']

    style_loss = tf.add_n([tf.reduce_mean(tf.abs(style_outputs[name] - style_targets[name]) ** 2)
                           for name in style_outputs.keys()])
    return style_loss

def l1_loss(y_pred, y_true, mask_list):

    y_pred = tf.cast(y_pred, dtype=tf.float32)
    y_true = tf.cast(y_true, dtype=tf.float32)
    return 1. * tf.reduce_mean(tf.abs(y_pred - y_true)) + 5. * tf.reduce_mean(tf.abs(y_pred - y_true) * (1 - mask_list))


def cal_perceptual(outputs, style_targets):

    style_outputs = outputs['perceptual']
    style_targets = style_targets['perceptual']

    result = tf.add_n(
        [tf.reduce_mean(tf.abs(style_outputs[name] - style_targets[name])) for name in style_outputs.keys()])
    return result

extractor = StyleContentModel(style_layers, content_layers)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.999)

@tf.function()
def train_step(generator, image_list, mask_image_list, mask_list):
    with tf.GradientTape() as tape, tf.GradientTape() as disc_tape:
        B, H, W, C = image_list.shape

        image_list = tf.cast(image_list, dtype=tf.float32)
        mask_image_list = tf.cast(mask_image_list, dtype=tf.float32)
        mask_list = tf.cast(mask_list, dtype=tf.float32)
        gen_image_list = generator([mask_image_list, mask_list])

        gen_image_list = tf.cast(gen_image_list, dtype=tf.float32)

        style_targets = extractor(image_list)
        outputs = extractor(gen_image_list)

        comp = image_list * mask_list + gen_image_list * (1 - mask_list)
        comp_outputs = extractor(comp)

        style_loss = 120. * (
                    style_content_loss(outputs, style_targets) + style_content_loss(comp_outputs, style_targets))
        perceptual_loss = 0.05 * (cal_perceptual(outputs, style_targets) + cal_perceptual(comp_outputs, style_targets))

        L1_loss = l1_loss(gen_image_list, image_list, mask_list)
        tvl_loss = 0.01 * total_variation_loss(comp, mask_list)

        loss = L1_loss + tvl_loss + style_loss + perceptual_loss
    grads = tape.gradient(loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

    return loss, style_loss, L1_loss, tvl_loss, perceptual_loss