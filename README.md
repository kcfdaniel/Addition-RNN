# Assignment 1

### Summary

| attempt  | training time | accuracy (%) | hidden layer size | hidden layer cell | hidden layer activation | weight init          | early stopping  | learning rate                                                                            | mini batch size   | training size | valid size | remarks                     |
|----------|---------------|--------------|---------------------|---------------------|---------------------------|----------------------|-----------------|------------------------------------------------------------------------------------------|-------------------|---------------|------------|-----------------------------|
| original | 73.92020679   | 62           | 4                   | basic               | sigmoid                   | random uniform       | N/A             | 0.01                                                                                     | 1 (no mini batch) | 4000          | 0          |                             |
| 1        |               | 85           | 4                   | basic               | sigmoid                   | **xavier initlaization** | **valid error < 1** | 0.01                                                                                     | **256**               | **3000**          | **1000**       | **fixed bug: input to network** |
| 2        |               | 85-95        | 4                   | basic               | **tanh**                      | xavier initlaization | valid error < 1 | 0.01                                                                                     | 256               | 3000          | 1000       | fixed bug: input to network |
| 3        |               | 72-100       | 4                   | basic               | **relu**                      | **he initialization**    | valid error < 1 | 0.01                                                                                     | 256               | 3000          | 1000       | fixed bug: input to network |
| 4        | 56.44956613   | 100          | 4                   | **lstm**                | **tanh**                      | **xavier initlaization** | valid error < 1 | 0.01                                                                                     | **16**                | 3000          | 1000       | fixed bug: input to network |
| 5        | 43.63636518   | 100          | 4                   | lstm                | tanh                      | xavier initlaization | valid error < 1 | 0.01                                                                                     | **32**                | 3000          | 1000       | fixed bug: input to network |
| 6        | 40.54460502   | 100          | 4                   | lstm                | tanh                      | xavier initlaization | valid error < 1 | 0.01                                                                                     | **64**                | 3000          | 1000       | fixed bug: input to network |
| 7        | 39.12361383   | 100          | 4                   | lstm                | tanh                      | xavier initlaization | valid error < 1 | 0.01                                                                                     | **128**               | 3000          | 1000       | fixed bug: input to network |
| 8        | 62.35889983   | 100          | 4                   | lstm                | tanh                      | xavier initlaization | valid error < 1 | 0.01                                                                                     | **256**               | 3000          | 1000       | fixed bug: input to network |
| 9        | 36.1095109    | 100          | 4                   | lstm                | tanh                      | xavier initlaization | valid error < 1 | **0.02**                                                                                     | 256               | 3000          | 1000       | fixed bug: input to network |
| 10       | 37.00159264   | 100          | 4                   | lstm                | tanh                      | xavier initlaization | valid error < 1 | 0.02                                                                                     | **500**               | 3000          | 1000       | fixed bug: input to network |
| 11       | 72.67997599   | 100          | 4                   | lstm                | tanh                      | xavier initlaization | valid error < 1 | **0.04**                                                                                     | 500               | 3000          | 1000       | fixed bug: input to network |
| 12       | 22.81028199   | 100          | 4                   | lstm                | tanh                      | xavier initlaization | valid error < 1 | **decayed_lr = tf.train.exponential_decay(0.04, global_step, 100000, 0.99, staircase=True)** | 500               | 3000          | 1000       | fixed bug: input to network |
| 13       | 20.48334098   | 100          | 4                   | lstm                | tanh                      | xavier initlaization | valid error < 1 | **decayed_lr = tf.train.exponential_decay(0.06, global_step, 100000, 0.99, staircase=True)** | 500               | 3000          | 1000       | fixed bug: input to network |
| 14       | 12.63456893   | 100          | **16**                  | lstm                | tanh                      | xavier initlaization | valid error < 1 | **decayed_lr = tf.train.exponential_decay(0.05, global_step, 100000, 0.99, staircase=True)** | 500               | 3000          | 1000       | fixed bug: input to network |
| 15       | 12.1438601    | 100          | 16                  | lstm                | tanh                      | xavier initlaization | valid error < 1 | **decayed_lr = tf.train.exponential_decay(0.06, global_step, 100000, 0.99, staircase=True)** | 500               | 3000          | 1000       | fixed bug: input to network |
| 16       | 17.57092595   | 100          | **32**                  | lstm                | tanh                      | xavier initlaization | valid error < 1 | decayed_lr = tf.train.exponential_decay(0.06, global_step, 100000, 0.99, staircase=True) | 500               | 3000          | 1000       | fixed bug: input to network |
| 17       | 11.78049707   | 100          | **20**                  | lstm                | tanh                      | xavier initlaization | valid error < 1 | decayed_lr = tf.train.exponential_decay(0.06, global_step, 100000, 0.99, staircase=True) | 500               | 3000          | 1000       | fixed bug: input to network |
| 18       | 9.659850121   | 100          | **24**                  | lstm                | tanh                      | xavier initlaization | valid error < 1 | decayed_lr = tf.train.exponential_decay(0.06, global_step, 100000, 0.99, staircase=True) | 500               | 3000          | 1000       | fixed bug: input to network |
| 19       | 10.8763442    | 100          | **28**                  | lstm                | tanh                      | xavier initlaization | valid error < 1 | decayed_lr = tf.train.exponential_decay(0.06, global_step, 100000, 0.99, staircase=True) | 500               | 3000          | 1000       | fixed bug: input to network |

---

## Checklist of Possibilities

* input feature
* loss function
* optimizer
* mini batch size
* hidden layer size
* hidden layer cell type & hidden layer activation function & weight initialization
* early stopping (with validation)
* learning rate (decay)
* regularization

---

## Reporting

### input feature

In the original code, the input feature is wrong.

_original way of obtaining iuput features_:

```python
a = np.array(a_list[j], dtype=np.uint8) #changed
b = np.array(b_list[j], dtype=np.uint8) #changed
ab = np.c_[a,b]
x = np.array(ab).reshape([1, binary_dim, 2])
```

E.g.  
a = [a0 a1 a2 a3 a4 a5 a6 a7]  
b = [b0 b1 b2 b3 b4 b5 b6 b7]  
ab = [a0 a1 a2 a3 a4 a5 a6 a7 b0 b1 b2 b3 b4 b5 b6 b7]  
x = [[[a0 a1] [a2 a3] [a4 a5] [a6 a7] [b0 b1] [b2 b3] [b4 b5] [b6 b7]]]  

_new way of obtaining input features_ :

```python
a = np.transpose([[a_list[j]]])
b = np.transpose([[b_list[j]]])
x = np.concatenate((a,b), axis=2)
```

E.g.  
a = [[[a0]] [[a1]] [[a2]] [[a3]] [[a4]] [[a5]] [[a6]] [[a7]]]  
b = [[[b0]] [[b1]] [[b2]] [[b3]] [[b4]] [[b5]] [[b6]] [[b7]]]  
x = [[[a0 b0] [a1 b1] [a2 b2] [a3 b3] [a4 b4] [a5 b5] [a6 b6] [a7 b7]]]

### loss function

Changed loss function to avoid Nan error by log(0), in case of predicted_outputs being 0:

_original loss function_:

```python
error = tf.reduce_sum(-(outputs * tf.log(predicted_outputs) + (1.0 - outputs) * tf.log(1.0 - predicted_outputs)))
```

_new loss function_:

```python
TINY          = 1e-6    # to avoid NaNs in logs 
error = tf.reduce_sum(-(outputs * tf.log(predicted_outputs + TINY) + (1.0 - outputs) * tf.log(1.0 - predicted_outputs + TINY)))
```

### optimizer

Adam optimizer is used, same as the original code.

### mini batch size

Different batch sizes were tested, the fastest training was obtained with a mini batch of 500 data points.  With a high batch size, the every training step takes a longer time, because of more computation needed, although the gradient direction would tend to point towards the global minimum.  With a small batch size, we lose the speedup by vectorizing the data, and the gradient of each training step would not be too accurately pointing towards the global minimum, but rather points at a local minimum of that small batch of data.

### hidden layer size

At attempt 18, it was found that the convergence speed is the fastest with 24 lstm cell.

### hidden layer cell type & hidden layer activation function & weight initialization

comparing

1. basic rnn cells with sigmoid activation, random uniform weight initialization
2. basic rnn cells with tanh activation, xavier weight initlaization
3. basic rnn cells with relu activation, he weight initlaization
4. lstm cells with tanh activation, xavier weight initlaization

it was found that **4. lstm cells with tanh activation, xavier weight initlaization** gives highest accuracy.

### early stopping (with validation)

3000 data points are used for training, 1000 data points are used for validation.

Although with lstm in our case, the bias, variance are both low and no overfitting is observed, in order to determine when the training should stop, and compare the training time of different models with different settings, I used early stopping when training the network, and stop the training when the validation error drops below 1.  

### learning rate (decay)

In order to speed up the training, I tried to bump up the learning rate, but around 0.06, the training error sometimes does not decrease smoothly.  So I added learning rate decay to slow down the learning gradually in the training process, while keeping a high learning rate at the beginning, and could speed up the training.

### regularization

Because little to none overfitting was observed, no regularization was needed.