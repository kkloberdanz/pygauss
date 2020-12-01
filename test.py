import gauss

a = gauss.Vec(range(100))
s = a.sum()
print(s)
assert(s == 4950.0)

a = gauss.Vec([1,2,3])
ans = a.dot([4,5,6])
print(ans)
assert(ans == 32.0)

nrm = a.l1norm()
print(nrm)
assert(nrm == 6.0)

y = gauss.Vec(2 + i for i in range(10))
x = gauss.Vec(range(10))

a = gauss.Vec(range(10))

b = gauss.Vec(range(10))

#assert(gauss.estimator.ordinary_least_squares(x, y) == (1.0, 2.0))

a = gauss.Vec([9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
b = gauss.Vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

#assert(gauss.error.mean_squared_error(a, b) == 7.3)
#assert(7.3 == (a - b).square().sum() / len(a))

nrm = str(a.l2norm())[:6]
print(nrm)
assert(nrm == "19.131")

m = b.argmax()
print(m)
assert(m == 9)

m = a.argmin()
print(m)
assert(m == 1)

v = str(a.variance())[:4]
print(v)
assert(v == '7.44')

print('OK')
