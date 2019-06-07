import pandas as pd

# expand function for all type of file


def csvRead(file, Y=None):
    data = pd.read_csv(file)
    if(Y == None):
        return data
    else:
        y = data[Y]
        x = data.drop([Y], axis=1)
        return x, y
