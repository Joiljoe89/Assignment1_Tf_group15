from PIL import Image
import numpy as np

#import 'pickled' file
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#unpacking training and test data
#test = unpickle("E:\joe\Assignment_02_Data\Assignment_Data\Test_Data\Test_Data\images1.pickle")
test = unpickle("/home/dlgroup2/dl_assignment/assignment2/test_data/images1.pickle")

for i in range(0,4033):
    img = Image.fromarray(test[i], 'RGB')
    #img.save("E:\joe\Assignment_02_Data\Assignment_Data\Test_Data\Test_Data\new_data\img%s.jpg" %i)
    img = img.resize((312,372), Image.NEAREST)
    img = img.convert('L')
    img.save("/home/dlgroup2/dl_assignment/assignment2/test_data/new_test_data/img%s.jpg" %i)

'''
###########################################################
path_save = "E:/joe/"

w, h = 512, 512
data = np.zeros((h, w, 3), dtype=np.uint8)

img = Image.fromarray(data, 'RGB')
img.save(path_save + 'TSFS15img.jpg')
data = img.resize((312,372), Image.NEAREST)
data = data.convert('L')
#img.save('my.png')
img.show()
data.show()
'''
