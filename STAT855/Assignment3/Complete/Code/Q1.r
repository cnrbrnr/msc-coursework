# Compute gamma-discounted cost function for given system and parameters
S = c(1, 2, 3) # state space
U = c(1, 2) # action space
gamma = 0.8 # discounting parameter

# Action-dependent state probability transition matrices
P_1 = matrix(c(1/2, 1/4, 1/4, 1/4, 1/2, 1/4, 1/4, 1/4, 1/2), nrow=3)
P_2 = matrix(rep(1/3, 9), nrow=3)

# Function for computing P_mu elementwise using formula in (a)
P_ij = function(P_1, P_2, u, i, j){
    if (u == 1){
        e = P_1[i, j]
    }
    else {
        e = P_2[i, j]
    }
    return(e)
}

# Policy for selecting actions under the given policy
policy = function(s) {
    if (s == 1){
        e = 1
    }
    else if (s == 2) {
        e = 2
    }
    else {
        e = 1
    }
    return(e)
}

# Initialize solution constructs
P_mu = matrix(rep(0, length(S)**2), nrow=3)
C = rep(0, length(S))

# Find P_mu
for (i in S){
    for (j in S){
        u = policy(i)
        P_mu[i, j] = P_ij(P_1, P_2, u, i, j)
    }
}

# Find C
for (i in seq_along(S)){
    C[i] = S[i] * policy(S[i])
}

# Compute (I - gamma*P_mu)^-1
I = diag(length(S))
ic_trans = (I - (gamma * P_mu))
inv_ic_trans = solve(sol_transform)

# Solve for the gamma-discounted cost function
J_mu = inv_ic_trans %*% C

# Report solution and constructs
ic_trans
P_mu
C
J_mu
