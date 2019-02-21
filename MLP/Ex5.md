# Exercise 5

## Keras Implementation



## Results

Train | Test | Parameters | P
-----|------|------|-------
All|All|5 epochs, 128 batch_size, dropout 0.2, 1 middle layer => size 512, act. = relu| 0.5164
All|All|5 epochs, 128 batch_size, dropout 0.15, 1 middle layer => size 512, act. = sigmoid| 0.5334
100000|10000|5 epochs, 256 batch_size, dropout 0.2, 4 hidden layer => size 512, act. = relu, optimizer = sgd| 0.55
100000|10000|5 epochs, 256 batch_size, dropout 0.2, 4 hidden layer => size 1024, act. = relu, optimizer = sgd| 0.55
100000|10000|5 epochs, 256 batch_size, dropout 0.2, 8 hidden layer => size 2048, act. = relu, optimizer = sgd| 0.56
All|All|20 epochs, 256 batch_size, dropout 0.2, 2 hidden layer => size 1024, act. = sigmoid, optimizer = rmsprop| 0.56
All|All|20 epochs, 256 batch_size, dropout 0.2, 2 hidden layer => size 1024, act. = relu, optimizer = sgd| 0.55
All|All|20 epochs, 256 batch_size, dropout 0.2, 8 hidden layer => size 2048, act. = relu, optimizer = sgd| 0.52  
All|All|20 epochs, 512 batch_size, dropout 0.2, 8 hidden layer => size 2048, act. = relu, optimizer = sgd| 0.53  
All|All|20 epochs, 256 batch_size, dropout 0.2, 8 hidden layer => size 1024, act. = relu, optimizer = sgd lr 0.001| 0.53 
