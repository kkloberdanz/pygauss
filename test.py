import gauss
a = gauss.Vec([1,2,3])
ans = a.dot([4,5,6])

y = gauss.Vec(2 + i for i in range(10))
x = gauss.Vec(range(10))
print(gauss.estimator.ordinary_least_squares(x, y))

a = gauss.Vec(range(10))

b = gauss.Vec(range(10))

print(a.dot32(b))
