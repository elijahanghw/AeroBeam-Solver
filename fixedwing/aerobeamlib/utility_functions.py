import numpy as np

class LargeMat_zeros():
    """
    Create a 2D matrix container where each elements holds a numpy array.
    Container is initialized with zeros.
    
    :param index: Shape of the matrix container
    :type index: list or tuples
    """
    
    def __init__(self, index):
        x, y = index
        self.largemat = []
        for i in range(x):
            largmat_inner = []
            for j in range(y):
                largmat_inner.append(0)
            self.largemat.append(largmat_inner)
            
    def __getitem__(self, index):
        x, y = index
        return(self.largemat[x][y])
    
    def __setitem__(self, index, entry):
        x, y = index
        self.largemat[x][y] = entry
        
    def __repr__(self):
        return str(self.largemat)
    
    def concat(self):
        """
        Concatenate matrix container into a single numpy array.
        """
        
        col = []
        for i, row in enumerate(self.largemat):
            row_concat = np.concatenate(row, axis=1)                # Concatenate arrays along the same row
            col.append(row_concat)
        return np.concatenate(col, axis=0)                          # Concatenate all rows into 1 matrix
    
class LargeMat_array():
    """
    Create a 2D matrix container where each elements holds a numpy array.
    
    :param largemat: 2 dimensional list where each element is a numpy array.
    :type index: list
    """
    
    def __init__(self, largemat):
        self.largemat = largemat
        
    def __getitem__(self, index):
        x, y = index
        return(self.largemat[x][y])
    
    def __setitem__(self, index, entry):
        x, y = index
        self.largemat[x][y] = entry
        
    def __repr__(self):
        return str(self.largemat)
    
    def concat(self):
        """
        Concatenate matrix container into a single numpy array.
        """
        
        col = []
        for i, row in enumerate(self.largemat):
            row_concat = np.concatenate(row, axis=1)                # Concatenate arrays along the same row
            col.append(row_concat)
        return np.concatenate(col, axis=0)