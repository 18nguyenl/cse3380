# Long Nguyen 1001247753
import numpy as np
import matplotlib.pyplot as plt


def findPCA(points) :
    """  student fills this in
    """
    
    mean = np.mean(points, axis=0)
    
    shiftedPoints = points - mean
    
    covMatrix = shiftedPoints.transpose().dot(shiftedPoints)
    
    eigenvalues, eigenvectors = np.linalg.eig(covMatrix)
    
    indices = abs(eigenvalues).argsort()[::-1]
    
    eigenvalues = abs(eigenvalues)[indices]
    eigenvectors = abs(eigenvectors)[:, indices]
    
    fpc = eigenvectors[0]
    
    return [shiftedPoints, covMatrix, fpc]

def plotPCA(points, fpc) :
    x = points[:, 0];   # make plotting easier to write
    y = points[:, 1];

    plt.plot(x, y, '.')
    mult = 10
    fpcX = fpc[0]
    fpcY = fpc[1]

    plt.plot([-mult*fpcX, mult*fpcX], [-mult*fpcY, mult*fpcY], 'm-')

    xmin = min(x) - 1
    xmax = max(x) + 1
    ymin = min(y) - 1
    ymax = max(y) + 1
    plt.axis([xmin, xmax, ymin, ymax])
    plt.title("points and first principal component")

    # from https://stackoverflow.com/questions/16183462/saving-images-in-python-at-a-very-high-quality
    #plt.savefig('pca.eps', format='eps', dpi=1000)

    plt.show()     # this causes the plot to display


######   main   ######
points = np.array([[2, 4, 3, 5, 6, 4, 5, 7, 5, 6, 7],
                   [2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6]]).T # the .T makes nx2

shiftedPoints, covMatrix, fpc = findPCA(points)
#        points : nx2 matrix (i.e., 2D Numpy array)
# shiftedPoints : nx2 matrix (i.e., 2D Numpy array)
#     covMatrix : 2x2 matrix (i.e., 2D Numpy array)
#           fpc : vector (i.e., 1D Numpy array)

print("part a: shifted points")
print(shiftedPoints)

print("\npart b: covariance matrix")
print(covMatrix)

print("\npart c: first principal component")
print(fpc)

plotPCA(points, fpc)