from libs import *
from configs import *


def bn_rl_conv(x, filters, kernel_size):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same')(x)

    return x


def dense_block(tensor, k, reps):

    for _ in range(reps):
        x = bn_rl_conv(tensor, filters=4*k, kernel_size=1)
        x = bn_rl_conv(x, filters=k, kernel_size=3)

        tensor = Concatenate()([tensor, x])

    return tensor


def transition_layer(x, theta):
    f = int(tensorflow.keras.backend.int_shape(x)[-1] * theta)
    x = bn_rl_conv(x, filters=f, kernel_size=1)
    x = AvgPool2D(pool_size=2, strides=2, padding='same')(x)

    return x

def get_model():
    input = Input(shape=IMAGE_SHAPE)


    x = Conv2D(filters=2*K, kernel_size=7, strides=2, padding='same')(input)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    
    for reps in REPETITIONS:
        d = dense_block(x, K, reps)
        x = transition_layer(d, THETA)
    
    x = GlobalAvgPool2D()(d)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(1024, activation='softmax')(x)

    x = Dense(512, activation='softmax')(x)
    output = Dense(len(CLASS_NAME), activation='softmax')(x)

    model = Model(input, output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    model = get_model()
    model.summary()