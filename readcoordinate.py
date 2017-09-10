import sys
import numpy as n

path = 'E:\groundtruth.txt'

f = n.loadtxt(path, delimiter=',')

print(f[1])
print(f[2])
print(f[3])