# Finding invariant distributions of symmetric walk on the hexagon
library(pracma) # used to find system RREF in 4c

# Generate one step transition matrix P
ind_f = c(2, 3, 4, 5, 6, 1)
ind_b = c(6, 1, 2, 3, 4, 5)
P_vec = c()
for (i in 1:6) {
    x = rep(0, 6)
    x[c(ind_b[i], ind_f[i])] <- 1/2
    P_vec = append(P_vec, x)
}
P = matrix(P_vec, nrow=6, ncol=6)

# Write out the system to solve by hand
pi_system = matrix(c(
    -1, 1/2, 0, 0, 0, 1/2,
    1/2, -1, 1/2, 0, 0, 0,
    0, 1/2, -1, 1/2, 0, 0,
    0, 0, 1/2, -1, 1/2, 0,
    0, 0, 0, 1/2, -1, 1/2,
    1, 1, 1, 1, 1, 1
), nrow=6, ncol=6)
pi_system = t(pi_system)

# Write out the RHS of the system
constants = matrix(c(0, 0, 0, 0, 0, 1), nrow=6)

# Solve the system and print the solution
sol = solve(pi_system, constants)
sol


# Get two step transition matrix from P in 4a
P_y = P %*% P
y_pi_system = P_y # initialize the system of equations
y_pi_system[y_pi_system == 0.5] <- -0.5 # change 0.5 entries to -0.5
y_pi_system[6, seq(6)] <- 1 # replace bottom equation distribution constraint

# Create the augmented matrix
aug_y = cbind(y_pi_system, constants)

# Obtain reduced row echelon form and read off solution
rref(aug_y)

# Test the invariance property using over many elements of the solution family 
generate_y_invariant = function(t){
    return(c(1/3 - t, t, 1/3 - t, t, 1/3 - t, t))
}
fail = FALSE
for (t in seq(0, 0.45, length.out=1000)){
    pi_inv = generate_y_invariant(t) 
    res = sum(abs(pi_inv - (pi_inv %*% P_y)))
    if (res != 0){
        fail = TRUE
    }
}
fail

