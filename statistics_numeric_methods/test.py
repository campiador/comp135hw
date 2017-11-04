from __future__ import division
import numpy


a = numpy.array([0.91, 0.74, 0.79, 0.82, 0.92, 0.83, 0.78, 0.85, 0.81, 0.87])
sum = 0
for elem in a:
    sum += elem

errors = [0.09, 0.26, 0.21, 0.18, 0.08, 0.17, 0.22, 0.15, 0.19, 0.13]

avg = sum / len(a)

sum_errors = 0
for elem in errors:
    sum_errors += elem

avg_error = sum_errors/ len(errors)

print "avg:", avg
print "avg err:", (1 - avg), avg_error

sum_diff_squared = 0
for elem in errors:
    sum_diff_squared += (elem - avg_error) ** 2

print sum_diff_squared

# b = sqrt (sum_diff_squared/k-1) ]

