# keras-titanic-
### 环境：python3.7 tensorflow 1.14
  代码直接克隆就可以运行了。
  #### keras介绍
Keras支持序列化模型和函数化模型，且二者具有一定数量的公有属性（attribute）和方法（method）。其中的公有属性包括layers、inputs、outputs，能够以Python列表的形式返回模型信息 [9]  ：
model.layers返回一个列表，列表中包含该模型内所有已创建的层对象（参见“层API”），例如keras.layers.Dense
model.inputs返回一个列表，列表包含该模型输入端接收的数据类型，例如以Tensorflow为后台时返回tf.Tensor
model.outputs与model.inputs相同但返回输出端信息。
Keras模型的公有方法主要与模型权重和结构有关，这里分别介绍如下：
model.summary返回该模型的结构信息、总参数量、可学习参数量等信息。
model.get_config返回一个字典，字典包含该模型所有对象的结构和编译信息。Keras可以通过该信息建立新模型。
model.get_weights返回一个列表，列表中每个成员都是NumPy数组形式的模型权重，列表的顺序为输入端到输出端。
model.set_weights(pre_trained_w)指定模型的所有权重，指定的权重必须与model.get_weights返回的权重大小一致。
model.to_yaml将Keras模型的结构输出为yaml文件，不包含模型权重。输出完成后，Keras模型可以由yaml文件导入。
model.save_weights(filepath)将Keras模型的权重保存为HDF5文件，运行时指定文件路径filepath。
model.load_weights(filepath, by_name=False)由HDF5文件导出权重到模型。model.load_weights通常只接受model.save_weights输出的文件，在接收其他来源的文件时，需要指定by_name=True并要求HDF5的变量名与模型层对象的名字相同。


### 编译（model.complie）
Keras模型的编译由model.compile实现，运行时可将Keras代码翻译为后台代码以执行学习、评估等后续操作。编译可以指定学习组件（参见“学习与优化API”），其使用格式如下 [11]  ：
1
2
model.compile(optimizer, loss=None, metrics=None, loss_weights=None, 
　           sample_weight_mode=None, weighted_metrics=None, target_tensors=None)·
格式中各参量的含义和调用方法如下 [11]  ：
optimizer为优化器、loss为损失函数、metrics为评价函数，可以按列表使用多个评价。
loss_weights为损失权重，可以在多输出的模型中对不同输出所对应的损失叠加不同的权重系数，要求提供与模型损失相对应的列表或张量。
sample_weight_mode是进行样本赋权的参量，默认为None，表示在model.fit中进行一维赋权；在编译时间序列模型时，可选择sample_weight_mode="temporal"，此时模型对时间序列样本（按时间步）进行二维赋权。
weighted_metrics和metrics的用法相同，在不指定样本赋权时等价于metrics，在指定了样本赋权时会对赋权样本的学习进行评价。
target_tensor：较少使用的参量，Tensorflow后台的Keras默认为学习目标分配张量占位符，但使用者可以调用该参量显式指定学习目标的张量。
除上述参量外，根据后台的不同，model.compile也可以将其它参量和关键字传递至keras.backend.function或tf.Session.run。
### 学习（model.fit、model.fit_generator）
模型编译完成后可以使用model.fit或model.fit_generator进行学习，其使用格式如下 [11]  ：

model.fit(x=None, y=None, verbose=1, callbacks=None, 
          epochs=1, initial_epoch=0, class_weight=None
          steps_per_epoch=None, batch_size=None, 
          validation_split=0.0, validation_data=None,
          validation_steps=None, validation_freq=1,
          shuffle=True, sample_weight=None) 
model.fit_generator(generator,...,
          max_queue_size=10, workers=1, use_multiprocessing=False)
         # ... 表示与model.fit相同的关键字
model.fit和model.fit_generator的使用格式包含一些共有的参量，其含义和调用方法如下 [11]  ：
verbose表示在学习时显示进度条和当前评估结果，默认为开启。
callback是回馈器选项（参见“回馈器”）。
epochs是学习的纪元数，即对所有学习样本迭代的次数。
initial_epoch表示开始学习时的纪元数，如果没有加载先前的学习权重则默认为从0开始。
class_weight是在分类问题中进行类别赋权的参量，即对不同分类的学习目标使用不同的权重。calss_weight的输入为一个字典，变量名为类别名，内容为权重，例如对二元分类，类别名通常为“0”和“1”。
