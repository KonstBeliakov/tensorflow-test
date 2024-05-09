import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant(1, shape=(1, 1))
#print(a) # tf.Tensor([[1]], shape=(1, 1), dtype=int32)

b = tf.constant([1, 2, 3, 4])
#print(b) # tf.Tensor([1 2 3 4], shape=(4,), dtype=int32)

c = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float16)
#print(c)
'''tf.Tensor(
[[1. 2.]
 [3. 4.]
 [5. 6.]], shape=(3, 2), dtype=float16)'''

# тензоры имеют один фиксированный тип данных (использовать разные типы в одном тензоре нельзя)
# + матрица должна быть прямоугольной

# tf.cast преобразование из одного типа данных в другой
a2 = tf.cast(a, dtype=tf.float32) # создаем тензор на используя другой тензор
#print(a2)
#tf.Tensor([[1.]], shape=(1, 1), dtype=float32)

# тензоры работают также как и массивы numpy

import numpy as np

b1 = np.array(b)
#print(b1)
#[1 2 3 4]

b2 = b.numpy() # другой способ преобразования тензора в массив numpy
#print(b2)
#[1 2 3 4]

# можно создавать изменяемые тензоры
v1 = tf.Variable(-1.2)
v2 = tf.Variable([4, 5, 6, 7], dtype=tf.float32)
v3 = tf.Variable(b)  # создание изменяемого тензора на основе другого тензора

#print(v1, v2, v3, sep='\n\n')
'''
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=-1.2>

<tf.Variable 'Variable:0' shape=(4,) dtype=float32, numpy=array([4., 5., 6., 7.], dtype=float32)>

<tf.Variable 'Variable:0' shape=(4,) dtype=int32, numpy=array([1, 2, 3, 4])>
'''

v1.assign(0) # изменение значения тензора
#print(v1)
'''
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.0>
'''

v2.assign([0, 1, 6, 7])  # должно быть столько же значений сколько и в исходном тензоре (размерности должны совпадать)
# можно пересоздавать тензор, но это работает не всегда :(
# tf.Variable() создает не ссылку на старый тензор, а новый тензор

#print(v2)
# <tf.Variable 'Variable:0' shape=(4,) dtype=float32, numpy=array([0., 1., 6., 7.], dtype=float32)>

v3.assign_add([1, 1, 1, 1])  # прибавление значений
v1.assign_sub(5)  # вычитание значений

#print(v1, v3, sep='\n\n')
'''
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=-5.0>

<tf.Variable 'Variable:0' shape=(4,) dtype=int32, numpy=array([2, 3, 4, 5])>
'''

# индексация и срезы в tensorflow работают также как и в numpy

val_0 = v3[0]  # первый элемент (тензор размерности 0)
val_12 = v3[1:3]  # элементы с 2 по 3 (тензор размерности 1)

#print(v3, val_0, val_12, sep='\n')
'''
<tf.Variable 'Variable:0' shape=(4,) dtype=int32, numpy=array([2, 3, 4, 5])>
tf.Tensor(2, shape=(), dtype=int32)
tf.Tensor([3 4], shape=(2,), dtype=int32)
'''