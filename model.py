#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    2023/10/13 22:09:17
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

def create_model():
    input_1_unnormalized = keras.Input(shape=(720,1280,3), name="input_1_unnormalized")
    input_1 = SubtractConstantLayer((720,1280,3), name="input_1_")(input_1_unnormalized)
    conv1_prepadded = layers.ZeroPadding2D(padding=((3,3),(3,3)))(input_1)
    conv1 = layers.Conv2D(64, (7,7), strides=(2,2), name="conv1_")(conv1_prepadded)
    bn_conv1 = layers.BatchNormalization(epsilon=0.001000, name="bn_conv1_")(conv1)
    activation_1_relu = layers.ReLU()(bn_conv1)
    max_pooling2d_1 = layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same")(activation_1_relu)
    res2a_branch2a = layers.Conv2D(64, (1,1), name="res2a_branch2a_")(max_pooling2d_1)
    bn2a_branch2a = layers.BatchNormalization(epsilon=0.001000, name="bn2a_branch2a_")(res2a_branch2a)
    activation_2_relu = layers.ReLU()(bn2a_branch2a)
    res2a_branch2b = layers.Conv2D(64, (3,3), padding="same", name="res2a_branch2b_")(activation_2_relu)
    bn2a_branch2b = layers.BatchNormalization(epsilon=0.001000, name="bn2a_branch2b_")(res2a_branch2b)
    activation_3_relu = layers.ReLU()(bn2a_branch2b)
    res2a_branch2c = layers.Conv2D(256, (1,1), name="res2a_branch2c_")(activation_3_relu)
    res2a_branch1 = layers.Conv2D(256, (1,1), name="res2a_branch1_")(max_pooling2d_1)
    bn2a_branch2c = layers.BatchNormalization(epsilon=0.001000, name="bn2a_branch2c_")(res2a_branch2c)
    bn2a_branch1 = layers.BatchNormalization(epsilon=0.001000, name="bn2a_branch1_")(res2a_branch1)
    add_1 = layers.Add()([bn2a_branch2c, bn2a_branch1])
    activation_4_relu = layers.ReLU()(add_1)
    res2b_branch2a = layers.Conv2D(64, (1,1), name="res2b_branch2a_")(activation_4_relu)
    bn2b_branch2a = layers.BatchNormalization(epsilon=0.001000, name="bn2b_branch2a_")(res2b_branch2a)
    activation_5_relu = layers.ReLU()(bn2b_branch2a)
    res2b_branch2b = layers.Conv2D(64, (3,3), padding="same", name="res2b_branch2b_")(activation_5_relu)
    bn2b_branch2b = layers.BatchNormalization(epsilon=0.001000, name="bn2b_branch2b_")(res2b_branch2b)
    activation_6_relu = layers.ReLU()(bn2b_branch2b)
    res2b_branch2c = layers.Conv2D(256, (1,1), name="res2b_branch2c_")(activation_6_relu)
    bn2b_branch2c = layers.BatchNormalization(epsilon=0.001000, name="bn2b_branch2c_")(res2b_branch2c)
    add_2 = layers.Add()([bn2b_branch2c, activation_4_relu])
    activation_7_relu = layers.ReLU()(add_2)
    res2c_branch2a = layers.Conv2D(64, (1,1), name="res2c_branch2a_")(activation_7_relu)
    bn2c_branch2a = layers.BatchNormalization(epsilon=0.001000, name="bn2c_branch2a_")(res2c_branch2a)
    activation_8_relu = layers.ReLU()(bn2c_branch2a)
    res2c_branch2b = layers.Conv2D(64, (3,3), padding="same", name="res2c_branch2b_")(activation_8_relu)
    bn2c_branch2b = layers.BatchNormalization(epsilon=0.001000, name="bn2c_branch2b_")(res2c_branch2b)
    activation_9_relu = layers.ReLU()(bn2c_branch2b)
    res2c_branch2c = layers.Conv2D(256, (1,1), name="res2c_branch2c_")(activation_9_relu)
    bn2c_branch2c = layers.BatchNormalization(epsilon=0.001000, name="bn2c_branch2c_")(res2c_branch2c)
    add_3 = layers.Add()([bn2c_branch2c, activation_7_relu])
    activation_10_relu = layers.ReLU()(add_3)
    res3a_branch2a = layers.Conv2D(128, (1,1), strides=(2,2), name="res3a_branch2a_")(activation_10_relu)
    bn3a_branch2a = layers.BatchNormalization(epsilon=0.001000, name="bn3a_branch2a_")(res3a_branch2a)
    activation_11_relu = layers.ReLU()(bn3a_branch2a)
    res3a_branch2b = layers.Conv2D(128, (3,3), padding="same", name="res3a_branch2b_")(activation_11_relu)
    bn3a_branch2b = layers.BatchNormalization(epsilon=0.001000, name="bn3a_branch2b_")(res3a_branch2b)
    activation_12_relu = layers.ReLU()(bn3a_branch2b)
    res3a_branch2c = layers.Conv2D(512, (1,1), name="res3a_branch2c_")(activation_12_relu)
    res3a_branch1 = layers.Conv2D(512, (1,1), strides=(2,2), name="res3a_branch1_")(activation_10_relu)
    bn3a_branch2c = layers.BatchNormalization(epsilon=0.001000, name="bn3a_branch2c_")(res3a_branch2c)
    bn3a_branch1 = layers.BatchNormalization(epsilon=0.001000, name="bn3a_branch1_")(res3a_branch1)
    add_4 = layers.Add()([bn3a_branch2c, bn3a_branch1])
    activation_13_relu = layers.ReLU()(add_4)
    res3b_branch2a = layers.Conv2D(128, (1,1), name="res3b_branch2a_")(activation_13_relu)
    bn3b_branch2a = layers.BatchNormalization(epsilon=0.001000, name="bn3b_branch2a_")(res3b_branch2a)
    activation_14_relu = layers.ReLU()(bn3b_branch2a)
    res3b_branch2b = layers.Conv2D(128, (3,3), padding="same", name="res3b_branch2b_")(activation_14_relu)
    bn3b_branch2b = layers.BatchNormalization(epsilon=0.001000, name="bn3b_branch2b_")(res3b_branch2b)
    activation_15_relu = layers.ReLU()(bn3b_branch2b)
    res3b_branch2c = layers.Conv2D(512, (1,1), name="res3b_branch2c_")(activation_15_relu)
    bn3b_branch2c = layers.BatchNormalization(epsilon=0.001000, name="bn3b_branch2c_")(res3b_branch2c)
    add_5 = layers.Add()([bn3b_branch2c, activation_13_relu])
    activation_16_relu = layers.ReLU()(add_5)
    res3c_branch2a = layers.Conv2D(128, (1,1), name="res3c_branch2a_")(activation_16_relu)
    bn3c_branch2a = layers.BatchNormalization(epsilon=0.001000, name="bn3c_branch2a_")(res3c_branch2a)
    activation_17_relu = layers.ReLU()(bn3c_branch2a)
    res3c_branch2b = layers.Conv2D(128, (3,3), padding="same", name="res3c_branch2b_")(activation_17_relu)
    bn3c_branch2b = layers.BatchNormalization(epsilon=0.001000, name="bn3c_branch2b_")(res3c_branch2b)
    activation_18_relu = layers.ReLU()(bn3c_branch2b)
    res3c_branch2c = layers.Conv2D(512, (1,1), name="res3c_branch2c_")(activation_18_relu)
    bn3c_branch2c = layers.BatchNormalization(epsilon=0.001000, name="bn3c_branch2c_")(res3c_branch2c)
    add_6 = layers.Add()([bn3c_branch2c, activation_16_relu])
    activation_19_relu = layers.ReLU()(add_6)
    res3d_branch2a = layers.Conv2D(128, (1,1), name="res3d_branch2a_")(activation_19_relu)
    bn3d_branch2a = layers.BatchNormalization(epsilon=0.001000, name="bn3d_branch2a_")(res3d_branch2a)
    activation_20_relu = layers.ReLU()(bn3d_branch2a)
    res3d_branch2b = layers.Conv2D(128, (3,3), padding="same", name="res3d_branch2b_")(activation_20_relu)
    bn3d_branch2b = layers.BatchNormalization(epsilon=0.001000, name="bn3d_branch2b_")(res3d_branch2b)
    activation_21_relu = layers.ReLU()(bn3d_branch2b)
    res3d_branch2c = layers.Conv2D(512, (1,1), name="res3d_branch2c_")(activation_21_relu)
    bn3d_branch2c = layers.BatchNormalization(epsilon=0.001000, name="bn3d_branch2c_")(res3d_branch2c)
    add_7 = layers.Add()([bn3d_branch2c, activation_19_relu])
    activation_22_relu = layers.ReLU()(add_7)
    res4a_branch2a = layers.Conv2D(256, (1,1), strides=(2,2), padding="same", name="res4a_branch2a_")(activation_22_relu)
    bn4a_branch2a = layers.BatchNormalization(epsilon=0.001000, name="bn4a_branch2a_")(res4a_branch2a)
    activation_23_relu = layers.ReLU()(bn4a_branch2a)
    res4a_branch2b = layers.Conv2D(256, (3,3), padding="same", name="res4a_branch2b_")(activation_23_relu)
    bn4a_branch2b = layers.BatchNormalization(epsilon=0.001000, name="bn4a_branch2b_")(res4a_branch2b)
    activation_24_relu = layers.ReLU()(bn4a_branch2b)
    res4a_branch2c = layers.Conv2D(1024, (1,1), name="res4a_branch2c_")(activation_24_relu)
    res4a_branch1 = layers.Conv2D(1024, (1,1), strides=(2,2), padding="same", name="res4a_branch1_")(activation_22_relu)
    bn4a_branch2c = layers.BatchNormalization(epsilon=0.001000, name="bn4a_branch2c_")(res4a_branch2c)
    bn4a_branch1 = layers.BatchNormalization(epsilon=0.001000, name="bn4a_branch1_")(res4a_branch1)
    add_8 = layers.Add()([bn4a_branch2c, bn4a_branch1])
    activation_25_relu = layers.ReLU()(add_8)
    res4b_branch2a = layers.Conv2D(256, (1,1), name="res4b_branch2a_")(activation_25_relu)
    bn4b_branch2a = layers.BatchNormalization(epsilon=0.001000, name="bn4b_branch2a_")(res4b_branch2a)
    activation_26_relu = layers.ReLU()(bn4b_branch2a)
    res4b_branch2b = layers.Conv2D(256, (3,3), padding="same", name="res4b_branch2b_")(activation_26_relu)
    bn4b_branch2b = layers.BatchNormalization(epsilon=0.001000, name="bn4b_branch2b_")(res4b_branch2b)
    activation_27_relu = layers.ReLU()(bn4b_branch2b)
    res4b_branch2c = layers.Conv2D(1024, (1,1), name="res4b_branch2c_")(activation_27_relu)
    bn4b_branch2c = layers.BatchNormalization(epsilon=0.001000, name="bn4b_branch2c_")(res4b_branch2c)
    add_9 = layers.Add()([bn4b_branch2c, activation_25_relu])
    activation_28_relu = layers.ReLU()(add_9)
    res4c_branch2a = layers.Conv2D(256, (1,1), name="res4c_branch2a_")(activation_28_relu)
    bn4c_branch2a = layers.BatchNormalization(epsilon=0.001000, name="bn4c_branch2a_")(res4c_branch2a)
    activation_29_relu = layers.ReLU()(bn4c_branch2a)
    res4c_branch2b = layers.Conv2D(256, (3,3), padding="same", name="res4c_branch2b_")(activation_29_relu)
    bn4c_branch2b = layers.BatchNormalization(epsilon=0.001000, name="bn4c_branch2b_")(res4c_branch2b)
    activation_30_relu = layers.ReLU()(bn4c_branch2b)
    res4c_branch2c = layers.Conv2D(1024, (1,1), name="res4c_branch2c_")(activation_30_relu)
    bn4c_branch2c = layers.BatchNormalization(epsilon=0.001000, name="bn4c_branch2c_")(res4c_branch2c)
    add_10 = layers.Add()([bn4c_branch2c, activation_28_relu])
    activation_31_relu = layers.ReLU()(add_10)
    res4d_branch2a = layers.Conv2D(256, (1,1), name="res4d_branch2a_")(activation_31_relu)
    bn4d_branch2a = layers.BatchNormalization(epsilon=0.001000, name="bn4d_branch2a_")(res4d_branch2a)
    activation_32_relu = layers.ReLU()(bn4d_branch2a)
    res4d_branch2b = layers.Conv2D(256, (3,3), padding="same", name="res4d_branch2b_")(activation_32_relu)
    bn4d_branch2b = layers.BatchNormalization(epsilon=0.001000, name="bn4d_branch2b_")(res4d_branch2b)
    activation_33_relu = layers.ReLU()(bn4d_branch2b)
    res4d_branch2c = layers.Conv2D(1024, (1,1), name="res4d_branch2c_")(activation_33_relu)
    bn4d_branch2c = layers.BatchNormalization(epsilon=0.001000, name="bn4d_branch2c_")(res4d_branch2c)
    add_11 = layers.Add()([bn4d_branch2c, activation_31_relu])
    activation_34_relu = layers.ReLU()(add_11)
    res4e_branch2a = layers.Conv2D(256, (1,1), name="res4e_branch2a_")(activation_34_relu)
    bn4e_branch2a = layers.BatchNormalization(epsilon=0.001000, name="bn4e_branch2a_")(res4e_branch2a)
    activation_35_relu = layers.ReLU()(bn4e_branch2a)
    res4e_branch2b = layers.Conv2D(256, (3,3), padding="same", name="res4e_branch2b_")(activation_35_relu)
    bn4e_branch2b = layers.BatchNormalization(epsilon=0.001000, name="bn4e_branch2b_")(res4e_branch2b)
    activation_36_relu = layers.ReLU()(bn4e_branch2b)
    res4e_branch2c = layers.Conv2D(1024, (1,1), name="res4e_branch2c_")(activation_36_relu)
    bn4e_branch2c = layers.BatchNormalization(epsilon=0.001000, name="bn4e_branch2c_")(res4e_branch2c)
    add_12 = layers.Add()([bn4e_branch2c, activation_34_relu])
    activation_37_relu = layers.ReLU()(add_12)
    res4f_branch2a = layers.Conv2D(256, (1,1), name="res4f_branch2a_")(activation_37_relu)
    bn4f_branch2a = layers.BatchNormalization(epsilon=0.001000, name="bn4f_branch2a_")(res4f_branch2a)
    activation_38_relu = layers.ReLU()(bn4f_branch2a)
    res4f_branch2b = layers.Conv2D(256, (3,3), padding="same", name="res4f_branch2b_")(activation_38_relu)
    bn4f_branch2b = layers.BatchNormalization(epsilon=0.001000, name="bn4f_branch2b_")(res4f_branch2b)
    activation_39_relu = layers.ReLU()(bn4f_branch2b)
    res4f_branch2c = layers.Conv2D(1024, (1,1), name="res4f_branch2c_")(activation_39_relu)
    bn4f_branch2c = layers.BatchNormalization(epsilon=0.001000, name="bn4f_branch2c_")(res4f_branch2c)
    add_13 = layers.Add()([bn4f_branch2c, activation_37_relu])
    activation_40_relu = layers.ReLU()(add_13)
    res5a_branch2a = layers.Conv2D(512, (1,1), name="res5a_branch2a_")(activation_40_relu)
    bn5a_branch2a = layers.BatchNormalization(epsilon=0.001000, name="bn5a_branch2a_")(res5a_branch2a)
    activation_41_relu = layers.ReLU()(bn5a_branch2a)
    res5a_branch2b = layers.Conv2D(512, (3,3), padding="same", dilation_rate=(2,2), name="res5a_branch2b_")(activation_41_relu)
    bn5a_branch2b = layers.BatchNormalization(epsilon=0.001000, name="bn5a_branch2b_")(res5a_branch2b)
    activation_42_relu = layers.ReLU()(bn5a_branch2b)
    res5a_branch2c = layers.Conv2D(2048, (1,1), name="res5a_branch2c_")(activation_42_relu)
    res5a_branch1 = layers.Conv2D(2048, (1,1), name="res5a_branch1_")(activation_40_relu)
    bn5a_branch2c = layers.BatchNormalization(epsilon=0.001000, name="bn5a_branch2c_")(res5a_branch2c)
    bn5a_branch1 = layers.BatchNormalization(epsilon=0.001000, name="bn5a_branch1_")(res5a_branch1)
    add_14 = layers.Add()([bn5a_branch2c, bn5a_branch1])
    activation_43_relu = layers.ReLU()(add_14)
    res5b_branch2a = layers.Conv2D(512, (1,1), name="res5b_branch2a_")(activation_43_relu)
    bn5b_branch2a = layers.BatchNormalization(epsilon=0.001000, name="bn5b_branch2a_")(res5b_branch2a)
    activation_44_relu = layers.ReLU()(bn5b_branch2a)
    res5b_branch2b = layers.Conv2D(512, (3,3), padding="same", dilation_rate=(2,2), name="res5b_branch2b_")(activation_44_relu)
    bn5b_branch2b = layers.BatchNormalization(epsilon=0.001000, name="bn5b_branch2b_")(res5b_branch2b)
    activation_45_relu = layers.ReLU()(bn5b_branch2b)
    res5b_branch2c = layers.Conv2D(2048, (1,1), name="res5b_branch2c_")(activation_45_relu)
    bn5b_branch2c = layers.BatchNormalization(epsilon=0.001000, name="bn5b_branch2c_")(res5b_branch2c)
    add_15 = layers.Add()([bn5b_branch2c, activation_43_relu])
    activation_46_relu = layers.ReLU()(add_15)
    res5c_branch2a = layers.Conv2D(512, (1,1), name="res5c_branch2a_")(activation_46_relu)
    bn5c_branch2a = layers.BatchNormalization(epsilon=0.001000, name="bn5c_branch2a_")(res5c_branch2a)
    activation_47_relu = layers.ReLU()(bn5c_branch2a)
    res5c_branch2b = layers.Conv2D(512, (3,3), padding="same", dilation_rate=(2,2), name="res5c_branch2b_")(activation_47_relu)
    bn5c_branch2b = layers.BatchNormalization(epsilon=0.001000, name="bn5c_branch2b_")(res5c_branch2b)
    activation_48_relu = layers.ReLU()(bn5c_branch2b)
    res5c_branch2c = layers.Conv2D(2048, (1,1), name="res5c_branch2c_")(activation_48_relu)
    bn5c_branch2c = layers.BatchNormalization(epsilon=0.001000, name="bn5c_branch2c_")(res5c_branch2c)
    add_16 = layers.Add()([bn5c_branch2c, activation_46_relu])
    activation_49_relu = layers.ReLU()(add_16)
    aspp_Conv_1 = layers.Conv2D(256, (1,1), padding="same", name="aspp_Conv_1_")(activation_49_relu)
    aspp_BatchNorm_1 = layers.BatchNormalization(epsilon=0.000010, name="aspp_BatchNorm_1_")(aspp_Conv_1)
    aspp_Relu_1 = layers.ReLU()(aspp_BatchNorm_1)
    aspp_Conv_2 = layers.Conv2D(256, (3,3), padding="same", dilation_rate=(6,6), name="aspp_Conv_2_")(activation_49_relu)
    aspp_BatchNorm_2 = layers.BatchNormalization(epsilon=0.000010, name="aspp_BatchNorm_2_")(aspp_Conv_2)
    aspp_Relu_2 = layers.ReLU()(aspp_BatchNorm_2)
    aspp_Conv_3 = layers.Conv2D(256, (3,3), padding="same", dilation_rate=(12,12), name="aspp_Conv_3_")(activation_49_relu)
    aspp_BatchNorm_3 = layers.BatchNormalization(epsilon=0.000010, name="aspp_BatchNorm_3_")(aspp_Conv_3)
    aspp_Relu_3 = layers.ReLU()(aspp_BatchNorm_3)
    aspp_Conv_4 = layers.Conv2D(256, (3,3), padding="same", dilation_rate=(18,18), name="aspp_Conv_4_")(activation_49_relu)
    aspp_BatchNorm_4 = layers.BatchNormalization(epsilon=0.000010, name="aspp_BatchNorm_4_")(aspp_Conv_4)
    aspp_Relu_4 = layers.ReLU()(aspp_BatchNorm_4)
    catAspp = layers.Concatenate(axis=-1)([aspp_Relu_1, aspp_Relu_2, aspp_Relu_3, aspp_Relu_4])
    dec_c1 = layers.Conv2D(256, (1,1), name="dec_c1_")(catAspp)
    dec_bn1 = layers.BatchNormalization(epsilon=0.000010, name="dec_bn1_")(dec_c1)
    dec_relu1 = layers.ReLU()(dec_bn1)
    dec_upsample1 = layers.Conv2DTranspose(256, (8,8), strides=(4,4), name="dec_upsample1_")(dec_relu1)
    dec_upsample1 = layers.Cropping2D(cropping=((2,2),(2,2)))(dec_upsample1)
    dec_c2 = layers.Conv2D(48, (1,1), name="dec_c2_")(activation_10_relu)
    dec_bn2 = layers.BatchNormalization(epsilon=0.000010, name="dec_bn2_")(dec_c2)
    dec_relu2 = layers.ReLU()(dec_bn2)
    dec_crop1 = layers.Cropping2D(cropping=((0,0),(0,0)))(dec_upsample1)
    dec_cat1 = layers.Concatenate(axis=-1)([dec_relu2, dec_crop1])
    dec_c3 = layers.Conv2D(256, (3,3), padding="same", name="dec_c3_")(dec_cat1)
    dec_bn3 = layers.BatchNormalization(epsilon=0.000010, name="dec_bn3_")(dec_c3)
    dec_relu3 = layers.ReLU()(dec_bn3)
    dec_c4 = layers.Conv2D(256, (3,3), padding="same", name="dec_c4_")(dec_relu3)
    dec_bn4 = layers.BatchNormalization(epsilon=0.000010, name="dec_bn4_")(dec_c4)
    dec_relu4 = layers.ReLU()(dec_bn4)
    scorer = layers.Conv2D(2, (1,1), name="scorer_")(dec_relu4)
    dec_upsample2 = layers.Conv2DTranspose(2, (8,8), strides=(4,4), name="dec_upsample2_")(scorer)
    dec_upsample2 = layers.Cropping2D(cropping=((2,2),(2,2)))(dec_upsample2)
    dec_crop2 = layers.Cropping2D(cropping=((0,0),(0,0)))(dec_upsample2)
    softmax_out = layers.Softmax()(dec_crop2)
    classification = softmax_out

    model = keras.Model(inputs=[input_1_unnormalized], outputs=[classification])
    return model

## Helper layers:

class SubtractConstantLayer(tf.keras.layers.Layer):
    def __init__(self, shape, name=None):
        super(SubtractConstantLayer, self).__init__(name=name)
        self.const = tf.Variable(initial_value=tf.zeros(shape), trainable=False)

    def call(self, input):
        return input - self.const
