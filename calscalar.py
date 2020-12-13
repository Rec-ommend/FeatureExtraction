#!/usr/bin/env python
import pyspark
import sys
import numpy

if len(sys.argv) != 3:
  raise Exception("Exactly 2 arguments are required: <inputUri> <outputUri>")

def callnumpy(dat1,dat2):
    return dat1,dat2

inputUri=sys.argv[1]
outputUri=sys.argv[2]

sc = pyspark.SparkContext()
f = lambda p: dat1+numpy.conj(dat2)
lines = sc.parallelize(callnumpy()).map(f).take(1)
wordCounts = words.map(lambda word: (word, 1)).reduceByKey(lambda count1, count2: count1 + count2)
wordCounts.saveAsTextFile(sys.argv[2])