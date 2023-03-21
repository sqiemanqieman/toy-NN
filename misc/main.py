import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

learning_rate = 0.01
training_epochs = 2000
display_step = 200
n_idle_epochs = 1000
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=n_idle_epochs, min_delta=0.0001)


class NEPOCHLogger(tf.keras.callbacks.Callback):
    def __init__(self, per_epoch=200):
        super(NEPOCHLogger, self).__init__()
        self.seen = 0
        self.per_epoch = per_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.per_epoch == 0:
            print('Epoch {}, loss {:.5f}, val_loss {:.5f}'.format(epoch, logs['loss'], logs['val_loss']))


log_display = NEPOCHLogger(per_epoch=display_step)
train_data = pd.read_csv(r'../two_spiral_train_data.txt', header=None, sep='\s+')
test_data = pd.read_csv(r'../two_spiral_test_data.txt', header=None, sep='\s+')
train_data['class_0'] = train_data[2].apply(lambda x: 1 if x == 0 else 0)
train_data['class_1'] = train_data[2].apply(lambda x: 1 if x == 1 else 0)
test_data['class_0'] = test_data[2].apply(lambda x: 1 if x == 0 else 0)
test_data['class_1'] = test_data[2].apply(lambda x: 1 if x == 1 else 0)


train_X = tf.convert_to_tensor(np.asarray(train_data.iloc[:, 0:2]))
train_y = tf.convert_to_tensor(np.asarray(train_data.iloc[:, 2]))
test_X = tf.convert_to_tensor(np.asarray(test_data.iloc[:, 0:2]))
test_y = tf.convert_to_tensor(np.asarray(test_data.iloc[:, 2]))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=100000, input_shape=(2,), activation='tanh'),
    # tf.keras.layers.Dense(units=2000, activation='tanh'),
    # tf.keras.layers.Dense(units=1000, activation='tanh'),
    # tf.keras.layers.Dense(units=500, activation='tanh'),
    # tf.keras.layers.Dense(units=200, activation='tanh'),
    # tf.keras.layers.Dense(units=100, activation='tanh'),
    # tf.keras.layers.Dense(units=50, activation='tanh'),
    # tf.keras.layers.Dense(units=10, activation='tanh'),
    # tf.keras.layers.Dense(units=5, activation='tanh'),
    tf.keras.layers.Dense(units=1, activation='swish')
])

optimizer = tf.keras.optimizers.SGD(learning_rate)

model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

# history = model.fit(train_X, train_y, epochs=training_epochs, validation_split=0.1, verbose=0, callbacks=[early_stopping, log_display])
history = model.fit(train_X, train_y, epochs=training_epochs, validation_split=0.1, verbose=0, callbacks=[log_display])
print("Train Finished!")

print('Final train loss: %10.5f' % history.history["loss"][-1])
print('Final test loss: %10.5f' % model.evaluate(test_X, test_y, verbose=0)[0])
y_hat = model.predict(test_X, verbose=0)
abs_error = np.abs((tf.convert_to_tensor(np.reshape(y_hat, (300,))) - tf.cast(test_y, tf.float32)).numpy())
print('total number of test samples: %d' % len(test_y))
print('number of test samples with absolute error >= 0.1: %d' % len(abs_error[abs_error >= 0.10]))
print('number of test samples with absolute error >= 0.5: %d' % len(abs_error[abs_error >= 0.50]))
with open('../abs_error.txt', 'w') as f:
    for item in abs_error:
        f.write("%s\n" % item)

# area cutting
class_0_area, class_1_area = {}, {}
for x in np.arange(-6.1, 6.1, 0.4):
    for y in np.arange(-6.1, 6.1, 0.4):
        if model.predict([[x, y]], verbose=0)[0][0] > 0.5:
            class_1_area.setdefault('x', []).append(x)
            class_1_area.setdefault('y', []).append(y)
        else:
            class_0_area.setdefault('x', []).append(x)
            class_0_area.setdefault('y', []).append(y)

# plot all training and testing data
class_0, class_1 = {}, {}
for X, y in zip(np.concatenate([train_X.numpy(), test_X.numpy()], 0), np.concatenate([train_y.numpy(), test_y.numpy()], 0)):
    if y == 0:
        class_0.setdefault('x', []).append(X[0])
        class_0.setdefault('y', []).append(X[1])
    else:
        class_1.setdefault('x', []).append(X[0])
        class_1.setdefault('y', []).append(X[1])
plt.axes().set_facecolor('gray')
plt.plot(class_0_area['x'], class_0_area['y'], 'co', markersize=12, label='class 0 area')
plt.plot(class_1_area['x'], class_1_area['y'], 'yo', markersize=12, label='class 1 area')
plt.plot(class_0['x'], class_0['y'], 'wo', label='class 0')
plt.plot(class_1['x'], class_1['y'], 'o', color='#000000', label='class 1')
plt.show()
