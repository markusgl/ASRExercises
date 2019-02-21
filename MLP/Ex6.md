# Exercise 6

## Results

Train | Test | Parameters | P	
-----|------|----|-------
100000|1000|5 epochs, 128 batch_size, dropout 0.2, 1 middle layer => size 512, act. = relu| 0.678  
100000|1000|5 epochs, 128 batch_size, dropout 0.2, 1 middle layer => size 512, act. = relu| 0.68  
50000|10000|5 epochs, 128 batch_size, dropout 0.2, 1 middle layer => size 512, act. = relu| 0.53  
10000|1000|5 epochs, 128 batch_size, dropout 0.2, 1 middle layer => size 512, act. = relu|0.52  
5000|1000|5 epochs, 128 batch_size, dropout 0.2, 1 middle layer => size 512, act. = relu|0.367  
100000|1000|20 epochs, 128 batch_size, dropout 0.2, 1 middle layer => size 512, act. = relu, optimizer=RMSprop lr=0.0001 | 0.706  
100000|1000|20 epochs, 128 batch_size, batch-normalization + dropout 0.2, 1 middle layer => size 512, act. = relu, optimizer=RMSprop lr=0.0001 | 0.729  
100000|1000|20 epochs, 128 batch_size, batch-normalization, 1 middle layer => size 512, act. = relu, optimizer=RMSprop lr=0.0001 | 0.687  
100000|1000|20 epochs, 128 batch_size, batch-normalization + dropout 0.2, 2 middle layer => size 512, act. = relu, optimizer=RMSprop lr=0.0001 | 0.730  
100000|1000|20 epochs, 128 batch_size, batch-normalization + dropout 0.2, 3 middle layer => size 512, act. = relu, optimizer=RMSprop lr=0.0001 | 0.725  
100000|1000|20 epochs, 256 batch_size, batch-normalization + dropout 0.2, 2 middle layer => size 512, act. = relu, optimizer=RMSprop lr=0.0001 | 0.731  
100000|1000|20 epochs, 512 batch_size, batch-normalization + dropout 0.2, 2 middle layer => size 512, act. = relu, optimizer=RMSprop lr=0.0001 | 0.738  
100000|1000|20 epochs, 512 batch_size, batch-normalization + dropout 0.2, 2 middle layer => size 1024, act. = relu, optimizer=RMSprop lr=0.0001 | 0.698    
100000|1000|20 epochs, 1024 batch_size, batch-normalization + dropout 0.2, 2 middle layer => size 512, act. = relu, optimizer=RMSprop lr=0.0001 | 0.718
100000|1000|20 epochs, 128 batch_size, batch-normalization + dropout 0.2, 2 middle layer => size 512, act. = relu, optimizer=adam | 0.693  
100000|1000|20 epochs, 128 batch_size, dropout 0.2, 3 middle layer => size 512, act. = relu, optimizer=adam | 0.691  
