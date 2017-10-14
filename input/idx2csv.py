import struct

with open('./idx_input/train-images-idx3-ubyte', 'rb') as f:
    ''' Converts the training images file into a csv file. '''

    outfile = open('./original_mnist/train-images.csv', 'w')
    bytes = f.read(8)
    magic, num_imgs = struct.unpack('>II', bytes)
    bytes = f.read(8)
    rows, cols = struct.unpack('>II', bytes)
    for i in range(num_imgs + 1):
        for j in range(rows * cols):
            if i == 0:
                outfile.write('pixel' + str(j) + (',' if j <
                                                  (rows * cols - 1) else '\n'))
            else:
                byte = f.read(1)
                byte = struct.unpack('>B', byte)
                if j != (rows * cols) - 1:
                    outfile.write(str(byte[0]) + ',')
                else:
                    outfile.write(str(byte[0]) + '\n')
    outfile.close()

with open('./idx_input/train-labels-idx1-ubyte', 'rb') as f:
    ''' Converts the training labels file into an equivalent csv file. '''

    outfile = open('./original_mnist/train-labels.csv', 'w')
    bytes = f.read(8)
    magic, size = struct.unpack('>II', bytes)
    for i in range(size + 1):
        if i == 0:
            outfile.write('label\n')
        else:
            byte = f.read(1)
            byte = struct.unpack('>B', byte)
            outfile.write(str(byte[0]) + '\n')
    outfile.close()


with open('./idx_input/t10k-images-idx3-ubyte', 'rb') as f:
    ''' Converts the test images file into an equivalent csv file. '''

    outfile = open('./original_mnist/test-images.csv', 'w')
    bytes = f.read(8)
    magic, num_imgs = struct.unpack('>II', bytes)
    bytes = f.read(8)
    rows, cols = struct.unpack('>II', bytes)

    for i in range(num_imgs + 1):
        for j in range(rows * cols):
            if i == 0:
                outfile.write('pixel' + str(j) + (',' if j <
                                                  (rows * cols - 1) else '\n'))
            else:
                byte = f.read(1)
                byte = struct.unpack('>B', byte)
                if j != (rows * cols) - 1:
                    outfile.write(str(byte[0]) + ',')
                else:
                    outfile.write(str(byte[0]) + '\n')
    outfile.close()

with open('./idx_input/t10k-labels-idx1-ubyte', 'rb') as f:
    ''' Converts the test labels file into an equivalent csv file. '''

    outfile = open('./original_mnist/test-labels.csv', 'w')
    bytes = f.read(8)
    magic, size = struct.unpack('>II', bytes)
    for i in range(size + 1):
        if i == 0:
            outfile.write('label\n')
        else:
            byte = f.read(1)
            byte = struct.unpack('>B', byte)
            outfile.write(str(byte[0]) + '\n')
    outfile.close()
