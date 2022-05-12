import numpy as np

Q = [[16,11,10,16,24,40,51,61],
     [12,12,14,19,26,58,60,55],
     [14,13,16,24,40,57,69,56],
     [14,17,22,29,51,87,80,62],
     [18,22,37,56,68,109,103,77],
     [24,35,55,64,81,104,113,92],
     [49,64,78,87,103,121,120,101],
     [72,92,95,98,112,100,103,99]]


def zigzag(input):
    h, v, i = 0,0,0
    output = np.zeros((64))
    while ((v < 8) and (h < 8)): # going up
        if ((h + v) % 2) == 0:
            if (v == 0):
                output[i] = input[v, h]
                if (h == 8):
                    v = v + 1
                else:
                    h = h + 1
                i = i + 1
            elif ((h == 7) and (v < 7)):
                output[i] = input[v, h]
                v = v + 1
                i = i + 1

            elif ((v > 0) and (h < 7)):
                output[i] = input[v, h]
                v = v - 1
                h = h + 1
                i = i + 1
        else:  # going down
            if ((v == 7) and (h <= 7)):
                output[i] = input[v, h]
                h = h + 1
                i = i + 1
            elif (h == 0):
                output[i] = input[v, h]
                if (v == 7):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
            elif ((v < 7) and (h > 0)):
                output[i] = input[v, h]
                v = v + 1
                h = h - 1
                i = i + 1
        if ((v == 7) and (h == 7)):
            output[i] = input[v, h]
            break
    return output

def inverse_zigzag(input):
    h, v, i = 0,0,0
    output = np.zeros((8,8))
    while ((v < 8) and (h < 8)):
        if ((h + v) % 2) == 0: # going up
            if (v == 0):
                output[v, h] = input[i]
                if (h == 8):
                    v = v + 1
                else:
                    h = h + 1
                i = i + 1
            elif ((h == 7) and (v < 7)):
                output[v, h] = input[i]
                v = v + 1
                i = i + 1

            elif ((v > 0) and (h < 7)):
                output[v, h] = input[i]
                v = v - 1
                h = h + 1
                i = i + 1

        else:  # going down
            if ((v == 7) and (h <= 7)):
                output[v, h] = input[i]
                h = h + 1
                i = i + 1
            elif (h == 0):
                output[v, h] = input[i]
                if (v == 7):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
            elif ((v < 7) and (h > 0)):  # all other cases
                output[v, h] = input[i]
                v = v + 1
                h = h - 1
                i = i + 1
        if ((v == 7) and (h == 7)):  # bottom right element
            output[v, h] = input[i]
            break
    return output