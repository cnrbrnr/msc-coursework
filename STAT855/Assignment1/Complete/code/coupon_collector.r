collect_coupons <- function(n){

    collection_size = 0
    coupon_counter = matrix(0, n, 1)
    coupons = 1:n
    T = 0

    while(collection_size < n){
        c = sample(coupons, 1)
        if(coupon_counter[c,] == 0){
            collection_size = collection_size + 1
        }
        coupon_counter[c,] = coupon_counter[c,] + 1
        T = T + 1
    }
    return(c(T, max(coupon_counter)))
}

set.seed(196883)
n_vals = c(10, 20) # number of coupons to try
R = 10^5 # number of Monte Carlo repeats
for(n in n_vals){
    res = matrix(0, R, 2) # matrix to store results
    for(r in 1:R){
        res[r, ] = collect_coupons(n) # simulate, sample T_i and M_i
    }
    # Report E[T], E[M] along with SDs
    cat(
        '\nn=', n,
        '\nE[T]=', mean(res[,1]), ', SD(T)=', sd(res[,1])/sqrt(R),
        '\nE[X]=', mean(res[,2]), ', SD(X)=', sd(res[,2])/sqrt(R)
    )
}
