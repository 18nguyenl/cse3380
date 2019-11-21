# Long Nguyen
# 1001247753
import numpy as np

def lsSVD(data, tol) :
    """
        student code goes here
    """
    b = data[:,-1]

    u, s, vh = np.linalg.svd(data[:,:-1], full_matrices=False)

    r = len([i for i in s if i >= tol])

    x = np.array([0.0, 0.0, 0.0, 0.0])

    for i in range(1, r + 1):
        x += (u.transpose()[i-1].dot(b) * (vh[i-1]))/s[i-1]

    S = s

    norm = np.linalg.norm(b - data[:,:-1].dot(x))

    return (x, S, r, norm)


#############################  main  #############################
#
#    DO NOT CHANGE ANYTHING BELOW THIS
#
if __name__ == "__main__" :
    data = np.genfromtxt("waterUsage.csv", dtype=None, delimiter=',', skip_header=5)
    
    # function leastSquaresSvdStudent
    #      parameters:  2D numpy array, int
    #   return values:  1D numpy array, 1D numpy array, float
    #
    
    tolerance = 2.5
    x, S, r, norm = lsSVD(data, tolerance)
    
    print("singular values")
    print(S)
    
    print("\neffective rank = %d when tolerance is %.1f" % (r, tolerance))
    
    print("\nEstimates")
    print("    laundry = %5.1f gallons" % x[0])
    print(" dishwasher = %5.1f gallons" % x[1])
    print("     shower = %5.1f gallons" % x[2])
    print(" sprinklers = %5.1f gallons" % x[3])
    
    print("\nthe norm of the residual is %.1f" % norm)

