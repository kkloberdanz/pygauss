import gauss
a = gauss.Vec([1,2,3])
ans = a.dot([4,5,6])

y = gauss.Vec(2 + i for i in range(10))
x = gauss.Vec(range(10))
print(gauss.estimator.ordinary_least_squares(x, y))

a = gauss.Vec(range(10))

b = gauss.Vec(range(10))

print(a.dot_cl(b))
assert(gauss.estimator.ordinary_least_squares(x, y) == (1.0, 2.0))

a = gauss.Vec([9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
b = gauss.Vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

assert(gauss.error.mean_squared_error(a, b) == 7.3)
assert(7.3 == (a - b).square().sum() / len(a))

print('OK')
