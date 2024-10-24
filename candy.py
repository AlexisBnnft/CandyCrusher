from numpy import *

def candy(n, m):
    # Create a 2D array to store the maximum number of candies that can be collected
    dp = zeros((n + 1, m + 1), dtype=int)

    # Fill the dp array
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]) + 1

    # Return the maximum number of candies that can be collected
    return dp[n][m]
