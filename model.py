import tensorflow as tf
from DownData import  x_train,y_train
import matplotlib.pyplot as plt

import os




model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(
    units=64,
    input_dim=7,
    use_bias=True,
    kernel_initializer='uniform',
    #初始化权重uniform分布
    bias_initializer='zeros',
    activation='relu'
))
model.add(tf.keras.layers.Dropout(rate=0.3))
model.add(tf.keras.layers.Dense(units=32,activation='sigmoid'))
model.add(tf.keras.layers.Dropout(rate=0.3))
model.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
#print(model.summary())
#如果是softmax激活函数，则选用分类交叉熵的损失函数“categorical——cross”
model.compile(optimizer=tf.keras.optimizers.Adam(0.003),
              loss='binary_crossentropy',
              metrics=['accuracy']
              )


checkpoint_path='./checkpoint/Titanic.{epoch:02d}-{val_loss:.2f}.ckpt'
callbacks=[
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                       save_weights_only=True,
                                       verbose=2,
                                       period=5
                                       )
]

# 模型训练
train_history=model.fit(x=x_train,
                        y=y_train,
                        validation_split=0.2,
                        epochs=100,
                        batch_size=40,
                        callbacks=callbacks,
                        verbose=2
                        )

def visu_train_history(train_history,train_metric,validation_metric):
    plt.plot(train_history.history[train_metric])
    plt.plot(train_history.history[validation_metric])
    plt.title('train history')
    plt.ylabel(train_metric)
    plt.xlabel('epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

visu_train_history(train_history,'acc','val_acc')
visu_train_history(train_history,'loss','val_loss')




