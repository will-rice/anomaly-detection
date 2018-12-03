import os
import time

import keras
import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, accuracy_score, f1_score
from keras.callbacks import Callback
from keras.backend.tensorflow_backend import set_session
from sklearn.svm import SVC

np.random.seed(1234)
tf.set_random_seed(1234)


df = pd.read_excel('./Normal_Attacker_Data.xlsx', dropna=True)

columns = df.columns[1:7]
print(columns)

df = df[['RSU-BTx-Car', 'RSU-PSR (%)', 'Car-PDR (%)', 'RSU PDSR (%)',
         'Car Received power (dBm)', 'Designation ( 0-Attacker, 1-Normal)']]

df['RSU-BTx-Car'] = df['RSU-BTx-Car'] / df['RSU-BTx-Car'].max()
df['RSU PDSR (%)'] = df['RSU PDSR (%)'] / 100
df['Car-PDR (%)'] = df['Car-PDR (%)'] / 100
df['RSU-PSR (%)'] = df['RSU-PSR (%)'] / 100
df['Car Received power (dBm)'] = df['Car Received power (dBm)'] / \
    df['Car Received power (dBm)'].min()
df.head()

inputs = df[['RSU-BTx-Car', 'RSU-PSR (%)', 'Car-PDR (%)', 'RSU PDSR (%)',
             'Car Received power (dBm)']].dropna()

labels = df[['Designation ( 0-Attacker, 1-Normal)']].dropna().astype(np.int32)

print(inputs.shape)
print(labels.shape)

X_train, X_test, y_train, y_test = train_test_split(
    inputs, labels, test_size=0.3)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

times = []
for i in range(1):
    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()
    times.append((end - start)*1000)

print('{:.2f} ms'.format(np.array(times)[1:].mean()))
print('{:.2f} accuracy'.format(accuracy_score(y_test, np.round(y_pred))))
svm_report = classification_report(y_test, np.round(
    y_pred), target_names=['anomaly', 'normal'])

latent_dim = 100


def generator():
    inputs = keras.layers.Input((latent_dim,))
    out = keras.layers.Dense(256)(inputs)
    out = keras.layers.BatchNormalization(momentum=0.8)(out)
    out = keras.layers.LeakyReLU()(out)
    out = keras.layers.Dropout(0.05)(out)
    out = keras.layers.Dense(5)(out)
    out = keras.layers.LeakyReLU()(out)

    model = keras.Model(inputs, out)
    return model


def build_discriminator():
    inputs = keras.layers.Input((5,))
    out = keras.layers.Dense(32)(inputs)
    out = keras.layers.BatchNormalization(momentum=0.8)(out)
    out = keras.layers.LeakyReLU(0.3)(out)
    out = keras.layers.Dropout(0.05)(out)
    out = keras.layers.Dense(32)(out)
    out = keras.layers.BatchNormalization(momentum=0.8)(out)
    out = keras.layers.LeakyReLU(0.3)(out)
    out = keras.layers.Dropout(0.05)(out)
    out = keras.layers.Dense(1, activation='sigmoid')(out)
    model = keras.Model(inputs, out)
    return model


generator = generator()
print(generator.summary())
discriminator = build_discriminator()
print(discriminator.summary())

optimizer = 'adam'

discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer)

z = keras.layers.Input((latent_dim,))
output = generator(z)

discriminator.trainable = False

validity = discriminator(output)

combined = keras.Model(z, validity)
combined.compile(loss='binary_crossentropy',
                 optimizer=optimizer)

X_train = np.array(X_train)
X_test = np.array(X_test)

batch_size = 4
epochs = 30

valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

d_losses = []
g_losses = []

start_d_loss = 1.0
start_precision = 0.0
for epoch in range(epochs + 1):
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    imgs = X_train[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    gen_data = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(imgs, valid)
    d_loss_fake = discriminator.train_on_batch(gen_data, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    d_losses.append(d_loss)

    g_loss = combined.train_on_batch(noise, valid)
    g_losses.append(g_loss)

    discriminator.save_weights('disc.h5')
    print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss, g_loss))


model = build_discriminator()
model.load_weights('disc.h5')
times = []
for i in range(1):
    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()
    times.append((end - start)*1000)

print('{:.2f} ms'.format(np.array(times).mean()))
print('{:.2f} accuracy'.format(accuracy_score(y_test, np.round(y_pred))))
gan_report = classification_report(y_test, np.round(
    y_pred), target_names=['anomaly', 'normal'])


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def build_classifier():
    inputs = keras.layers.Input((5,))
    out = keras.layers.Dense(32)(inputs)
    out = keras.layers.BatchNormalization(momentum=0.8)(out)
    out = keras.layers.LeakyReLU(0.3)(out)
    out = keras.layers.Dropout(0.05)(out)
    out = keras.layers.Dense(32)(out)
    out = keras.layers.BatchNormalization(momentum=0.8)(out)
    out = keras.layers.LeakyReLU(0.3)(out)
    out = keras.layers.Dropout(0.05)(out)
    out = keras.layers.Dense(1, activation='sigmoid')(out)
    model = keras.Model(inputs, out)
    return model


ckpt = keras.callbacks.ModelCheckpoint(
    './classifier.h5', monitor='val_f1', save_best_only=True, mode='max')

model = build_classifier()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', f1])
print(model.summary())
model.fit(X_train,
          y_train,
          batch_size=4,
          epochs=10000,
          validation_data=[X_test, y_test],
          callbacks=[ckpt],
          verbose=0)

model = build_classifier()
model.load_weights('classifier.h5')
times = []
for i in range(10):
    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()
    times.append((end - start)*1000)

print('{:.2f} ms'.format(np.array(times)[1:].mean()))
print('{:.2f} accuracy'.format(accuracy_score(y_test, np.round(y_pred))))
dnn_report = classification_report(y_test, np.round(
    y_pred), target_names=['anomaly', 'normal'])

print(svm_report)
print(gan_report)
print(dnn_report)
