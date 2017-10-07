import numpy

def modifyImage(image):

    rows = image.shape[0]
    coloumns = image.shape[1]

    zoomedImage = []
    for i in rows:
        zoomedImage.append(tmp)
        tmp = []
        for j in coloumns:
            if(image[i][j] != 0):
                tmp.append(image[i][j])

    return zoomedImage