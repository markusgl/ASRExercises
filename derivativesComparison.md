# KNN simple

Train | Test | k | Pmax | PmaxDer
------|------|---|------|-------
2000 | 100 | 1..14 | 0.49 | 0.65
5000 | 1000|1..24| 0.496 | 0.558

# KNN normalized

Train | Test | k | Pmax | PmaxDer
------|------|---|------|-------
2000 | 100 | 1..14 | 0.52 | 0.48
5000 | 1000|1..24| 0.515 | 0.558

# Gaussian algorithm

Train | Test |P	| Pderiv | 
-----|------|----|----|
50000 | 1000| 0.543| 0.482
10000|1000|0.495| 0.467
50000|2000|0.5145| 0.453

# MLP simple

Train | Test | Parameters | P | Pder	
-----|------|------|-------|-------
100000|10000|5 epochs, 128 batch_size, dropout 0.2, 1 hidden layer => size 512, act. = relu, optimizer = sgd| 0.523 | 0.592
100000|1000|5 epochs, 128 batch_size, dropout 0.2, 1 middle layer => size 512, act. = relu| 0.61| 0.65
50000|10000|5 epochs, 128 batch_size, dropout 0.2, 1 middle layer => size 512, act. = relu| 0.52| 0.58


# MLP neighbor

Train | Test | Parameters | P | Pder	
-----|------|------|-------|-------
100000|1000|5 epochs, 128 batch_size, dropout 0.2, 1 middle layer => size 512, act. = relu| 0.68 | 0.65
50000|10000|5 epochs, 128 batch_size, dropout 0.2, 1 middle layer => size 512, act. = relu| 0.53 | 0.58
10000|1000|5 epochs, 128 batch_size, dropout 0.2, 1 middle layer => size 512, act. = relu| 0.52| 0.549
5000|1000|5 epochs, 128 batch_size, dropout 0.2, 1 middle layer => size 512, act. = relu| 0.367| 0.482
