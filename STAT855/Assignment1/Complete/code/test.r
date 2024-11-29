A = matrix(c(4, 2, 3, 7, 5, 6, 7, 8, 1), 3, 3)
Y = c(1, 2, 3)

X = solve(A) %*% Y
X = solve(A, Y)
X

is.vector(X)
seq(1,3,2)
