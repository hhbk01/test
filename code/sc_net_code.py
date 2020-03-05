from keras.engine.saving import model_from_json
from keras.models import Model
from keras.layers import Embedding, RepeatVector, Permute, Input, Dense, \
    Dropout, Conv2D, MaxPooling2D, Flatten, Concatenate, Activation, Multiply, \
    Reshape, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.layers import Lambda
from keras import regularizers
from keras import backend as K
import keras
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def l2(x):
    x = K.l2_normalize(x, axis=3)
    return x


class Deep_Net():
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        pass

    def layer(self):
        pass

    def predict(self, X):
        pass

    def evaulate(self, X, y):
        pass

    def save(self, save_path, type='json'):
        if type == 'json':
            # save as JSON
            json_string = self.model.to_json()
            open(save_path + ".json", 'w').write(json_string)
            self.model.save_weights(save_path + ".h5")
        else:
            self.model.save(save_path + ".h5")
    def data_save_roc(self, x, y, pre_score, save_path, average="macro"):
        pass

    def load(self, model_path, type='json'):
        if type == 'json':
            # save as JSON
            self.model = model_from_json(open(model_path + ".json").read())
            # model_main.trainable = False
            self.model.load_weights(model_path + ".h5")
        else:
            self.model.load_model(model_path + ".h5")

class SC_NET(Deep_Net):
    """
    SC_NET
    """
    def __init__(self):
        self.model = None

    def fit(self, x_train, y_train, num_classes, shape, batch_size, epochs, validation_data, num_features):
        self.layer(shape, num_classes, num_features=num_features)
        self.model = Model(inputs=self.inputs, output=self.outputs)
        early_stopping = EarlyStopping(monitor='val_acc', patience=15, verbose=1)
        reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto',min_lr=1e-7,cooldown=50,
                                                   factor=0.1,verbose=1)
        optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=1,
                       validation_data=validation_data,
                       callbacks=[reduce,early_stopping ])

    def evaluate(self, X, y):
        scores = self.model.evaluate(X, y, verbose=1)
        print('loss:', scores[0])
        print('accuracy:', scores[1])
        return scores

    def predict(self, X):
        y_score = self.model.predict(X)
        return y_score

    # Layers
    def layer(self, shape, num_classes, num_features):
        # param:  the type of shape is [(,),(,)]
        img_input = Input(shape=shape[0], name='img_input')
        txt_input = Input(shape=shape[1], name='txt_input')
        self.inputs = [img_input, txt_input]

        word_embed = Embedding(num_features, 48)(txt_input)
        xt = word_embed

        # Text
        xt_att = Dense(1, activation='tanh', name='self_att_tanh')(xt)
        xt_att = Flatten()(xt_att)
        xt_att = Activation(activation='softmax', name='self_att_soft')(xt_att)
        xt_att = RepeatVector(48)(xt_att)
        xt_att = Permute((2, 1))(xt_att)
        xt_ = Multiply(name='self_att_multi')([xt, xt_att])
        xt = Lambda(lambda x: K.sum(x, axis=1), name='self_att_sum')(xt_)

        # Image
        x1 = Conv2D(64, (11, 11), strides=(4, 4), input_shape=shape, padding='valid', activation='relu',
                    kernel_initializer='glorot_uniform', name='conv1')(img_input)
        x1 = Lambda(lambda x: K.l2_normalize(x, axis=3), name='l2_normal1')(x1)
        x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='max_pooling1')(x1)
        x3 = Conv2D(128, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                    name='conv2')(x2)
        x3 = Lambda(lambda x: K.l2_normalize(x, axis=3), name='l2_normal2')(x3)
        x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='max_pooling2')(x3)
        x5 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu',kernel_initializer='glorot_uniform',name='conv3')(x4)
        x6 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform',name='conv4')(x5)
        x6 = BatchNormalization()(x6)
        x6 = Activation(activation='relu')(x6)
        x6 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='max_pooling3')(x6)
        x7 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform',
        kernel_regularizer=regularizers.l2(0.001),name='conv5')(x6)
        x8 = Activation(activation='relu',name="act_relu")(x7)
        xi = x8

        # channels
        xt_att = Dense(512, name='fc1_txt',activation='relu')(xt)
        xi_ = GlobalAveragePooling2D(name='global_pooling')(xi)
        x_mix = Concatenate(name='concat1')([xi_, xt_att])
        x_mix = Dense(512, activation='relu', name='att1')(x_mix)
        x_mix = Dense(512, activation='relu', name='att2')(x_mix)
        x_mix = Reshape((1, 1, 512))(x_mix)
        xi = Multiply(name='map')([xi, x_mix])
        x9 = Flatten()(xi)

        x10 = Dense(1024, activation='relu', name='fc1')(x9)
        x11 = Dropout(0.5)(x10)
        x12 = Dense(48, activation='relu', name='fc2')(x11)
        x13 = Concatenate(name='concat2')([x12, xt])
        x14 = Dense(48, activation='relu', name='fc3')(x13)
        self.outputs = Dense(num_classes, activation='softmax')(x14)


