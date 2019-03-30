# Assignment 1

**Summary**

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
**Checklist of Possibilities**

* loss function
* optimizer
* hidden layer size
* hidden layer cell type
* hidden layer activation function
* weight initialization
* early stopping (with validation)
* learning rate (decay)
* mini batch size
* regularization

---
**Reporting**
original loss function:
```error = -(outputs * tf.log(predicted_outputs + TINY) + (1.0 - outputs) * tf.log(1.0 - predicted_outputs + TINY))
```

```TINY          = 1e-6    # to avoid NaNs in logs
new loss function:
```
