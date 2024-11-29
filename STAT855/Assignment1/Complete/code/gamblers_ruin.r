gr_run <- function(p, init){
    # Simulate gambler's run once, returning stopping time and maximal fortune
    T = 0 # initialize number of bets
    fortune = init # initialize fortune
    M = fortune # initialize maximal fortune
    while(fortune > 0){
        fortune = fortune + (2 * rbinom(1, 1, p) - 1) # +1 w.p. p, -1 w.p. (1-p)
        if(fortune > M){
            M = fortune # update maximal fortune 
        }
        T = T + 1 # increment number of bets
    }
    return(c(T, M))
}

set.seed(196883)
p_vals = c(0.4, 0.45, 0.48) # win probabilities to check
R = 10^5 # number of Monte Carlo repeats
init = 5 # initial fortune
for(p in p_vals){
    res = matrix(0, R, 2) # matrix to store results
    for(r in 1:R){
        res[r, ] = gr_run(p, init) # simulate, sample T_i and M_i
    }
    # Report E[T], E[M] along with SDs
    cat(
        '\np=', p,
        '\nE[T]=', mean(res[,1]), ', SD(T)=', sd(res[,1])/sqrt(R),
        '\nE[M]=', mean(res[,2]), ', SD(M)=', sd(res[,2])/sqrt(R)
    )

    # Report P(M \geq 10) estimated by E[1_{M \geq 10}] along with SD
    cat(
        '\nE[1_{M >= 10}]=', mean(res[,2] >= 10), ', SD(1_{M >= 10})=', sd(res[,2] >= 10)/sqrt(R), '\n'
    )
}
