# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:27:17 2021

@author: Lyla
"""


import tensorflow as tf
tf.__version__

import pandas as pd
import numpy as np
from tensorflow.keras import layers
import random
import os
import time
from IPython import display

file_path='/content/hl60.tsv'
data_read=pd.read_csv(file_path,sep='\t')
guideSeq=np.array(data_read['sgRNA'])
labels=np.array(data_read['Normalized_efficacy'])

ntmap = {'A': (1, 0, 0, 0),
         'C': (0, 1, 0, 0),
         'G': (0, 0, 1, 0),
         'T': (0, 0, 0, 1)
         }
ntmap_new={}
for key,val in ntmap.items():
    ntmap_new[val]=key

def get_seqcode(seq):
    return list(map(lambda c: ntmap[c], seq))

def oneHotcoding(Seq):
    n=0
    for seq in Seq:
        if n==0:
            SeqcodeL=[]
        seqcode=get_seqcode(seq)
        n+=1
        SeqcodeL.append(seqcode)
        seqcode=[]
        SeqcodeA=np.array(SeqcodeL)
    return SeqcodeA

guidecode=oneHotcoding(guideSeq)
pre=1
if pre==1:
    train_data=guidecode
    
a=np.reshape(train_data,(2076,1,23,4))
train_data=tf.convert_to_tensor(a)
#print(train_data[0])

BUFFER_SIZE = train_data.shape[0]
BATCH_SIZE = 16

# 批量化和打乱数据
train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(5*1, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((5,1)))
    assert model.output_shape == (None, 5,1) # 注意：batch size 没有限制

    model.add(layers.Conv1DTranspose(128,5, strides=1, padding='same', use_bias=False))
    assert model.output_shape == (None, 5, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1DTranspose(64,5, strides=2, padding='same', use_bias=False))
    assert model.output_shape == (None, 10, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1DTranspose(4, 5, strides=3, padding='same', use_bias=False))
    assert model.output_shape == (None, 30, 4)

    model.add(layers.Conv1D(4,8,strides=1, padding='valid', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 23, 4)
    
    return model

model = make_generator_model()
model.summary()

def make_discriminator_model():
    model = tf.keras.Sequential()
    
    model.add(layers.Conv1D(16, 3, strides=1, padding='same',input_shape=(23, 4)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(32,3, strides=1, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

model = make_discriminator_model()
model.summary()


# 训练之前进行的尝试
generator = make_generator_model()
noise = tf.random.normal([1,100])
generated_gRNA=generator(noise,training=False)
# 对刚制造出的假序列进行判断
discriminator = make_discriminator_model()
decision = discriminator(generated_gRNA)
print(decision)
#对原本数据中的第一条真序列进行判断
discriminator = make_discriminator_model()
decision = discriminator(train_data[0])
print(decision)


# 该方法返回计算交叉熵损失的辅助函数
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 20

# 将重复使用该种子（更容易可视化进度）
seed = tf.random.normal([num_examples_to_generate, noise_dim])
#print(seed)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt") # 连接两个或更多的路径名组件
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                  discriminator_optimizer=discriminator_optimizer,
                  generator=generator,
                  discriminator=discriminator)

def train_step(sequences):
  noise = tf.random.normal([BATCH_SIZE, noise_dim])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_sequences = generator(noise, training=True)
      
    real_output = discriminator(sequences, training=True)
    fake_output = discriminator(generated_sequences, training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)

  gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
  gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
  generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
  
  
ntmap_new={}
for key,val in ntmap.items():
    ntmap_new[val]=key

def get_seq(seqcode): # 只能处理个碱基
    for code in ntmap_new:
        if (seqcode==code).all():
            seq=ntmap_new[code]
    return seq

def oneHotdecoding(codes): # 只能处理一条序列
    n=0
    for code in codes:
        if n==0:
            SeqL=[]
        seq=get_seq(code)
        n+=1
        SeqL.append(seq)
        seq=[]
        SeqA=np.array(SeqL)
    return SeqA

def savedata1D(list1,filepath):
    output = open(filepath,'a+',encoding='utf-8')
    #output.write(string)
    #output.write('\n')
    for i in range(len(list1)):
        output.write(str(list1[i]))    
    output.write('\n')           
    output.close()

def generate_and_save_sequence(model, epoch, test_input,save=False):
  # 注意 training` 设定为 False
  # 因此，所有层都在推理模式下运行（batchnorm）。
  predictions = model(test_input, training=False)
  for i in range(predictions.shape[0]):
    #print(predictions[i,:,:])
    list1=[]
    temp=predictions[i,:,:].numpy() # 急切执行默认情况下.numpy()处于启用状态，因此只需调用Tensor对象即可
    temp2=np.zeros(temp.shape)
    for i in range(temp.shape[0]):
      for j in range(temp.shape[1]):
        if temp[i][j]==np.amax(temp[i]):
          temp2[i][j]=1
        else:
          temp2[i][j]=0
    print(temp2)
    list1.append(temp2)
    if save:
      new_codes=np.stack(list1, axis=0)
      guideseqs=[]
      for codes in new_codes:
        guideseqs.append(oneHotdecoding(codes))
      for seq in guideseqs:
        fp='/content/generated_seq.txt'
        savedata1D(seq,fp)  

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for sequence_batch in dataset:
      train_step(sequence_batch) # 计算训练集中每一个batch对应的step
      
    # 生成并保存序列
    generate_and_save_sequence(generator,
                epoch + 1,
                seed)
    # 每 15 个 epoch 保存一次模型（模型参数）
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)


    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start)) # 每训练一次打印一次训练时长

  # 最后一个 epoch 结束后清除输出并生成保存序列
  display.clear_output(wait=True) # 清除输出
  generate_and_save_sequence(generator, # 生成保存序列
              epochs,
              seed,save=True)
  
  
%%time
train(train_dataset, EPOCHS)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))











