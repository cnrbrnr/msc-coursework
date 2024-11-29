# Numerically compute average cost function for specific instance of MDP
S = c(1,2,3) # State space
U = c(1,2) # Action space

# Transition kernels
P_1 = matrix(c(1/2, 1/4, 1/4, 1/4, 1/2, 1/4, 1/4, 1/4, 1/2), nrow=3)
P_2 = matrix(rep(1/3, 9), nrow=3)
t_kernel = list(P_1, P_2)

# MDP simulation
sim_MDP = function(x_0, T){

    mu = c(1, 2, 1) # indexed state-action mappings under policy mu 
    
    # Initialize
    u = mu[x_0] 
    x = x_0
    res = list(c(x_0), c(u))

    # Iterate
    for (i in 1:T) {

        # Update state variables
        x = sample(S, size=1, prob=t_kernel[[u]][x, ]) # Generate random state
        u = mu[x] # Update action

        # Store current iterate
        res[[1]][i] = x
        res[[2]][i] = u
    }

    return(res)
}

# Run simulation
T = 1e3
x_0 = 1
process = sim_MDP(x_0, T) 

# Functions for computing cost
cost = function(x, u){
    return(x * u)
}
estimate_avg_cost = function(data){
    N = length(data[[1]]) # sample size
    est = (1 / cumsum(rep(1, N))) * cumsum(mapply(cost, data[[1]], data[[2]])) # estimate
    return(est)
}

# Compute and print result
avg_cost = estimate_avg_cost(process)

# Plot result
t = seq(1, T, 1)
ylab = expression(V^mu)
plot(t, avg_cost, type='l', ylab=ylab, xlab='Iterate', pch=16, lwd=3, cex.axis=1.5, cex.lab=2, bty='l')
