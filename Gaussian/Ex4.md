# Exercise 4 

## Implementation

```python

for x in testdata:
	for k in Classes:
		u[k] = - 2 * log (priori[k]);
		sum1 = 0;
		sum2 = 0;

		for f in std_dev[k], mean[k]:
			sum1 += log(2 * pi * std_dev[k][f]);
			sum2 += (x[f] - mean[k][f])^2 / std_dev[k][f];


		u[k] += sum1 + sum2;

	testdataSet.Class = min(u[k]).Key

```

## Recognition Rate

Train | Test |P	|
-----|------|----|
50000 | 1000| 0.543
10000|1000|0.495
50000|2000|0.5145
100000|Alle| 0.447
Alle|Alle|0.4574