import cv2
import os
import numpy as np
import time
import sys
import codecs
import errno
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.datasets.mnist
from torchvision import transforms
from tqdm import tqdm


do_learn = True   #true:training or false:test
save_frequency = 2
batch_size = 12
lr = 0.001
num_epochs = 15
weight_decay = 0.0001
pred=False
feature=[]
images=[]
dim=(150,150)

class BalancedMNISTPair(torch.utils.data.Dataset):


   def __init__(self, root, train=True, transform=None, target_transform=None, download=False,pred=False):
      self.root = os.path.expanduser(root)
      self.transform = transform
      self.target_transform = target_transform
      self.train = train # training set or test set

      if self.train:
         train_labels_class = []
         train_data_class = []
         self.image_names=[]

         train_path=  r'D:/Train/'
         train_data_class=[]
         i=1
         classes=[]
         imgList=[]
         files=os.listdir(train_path)
         files.sort()
         for file in files:
             className= int(file[:3])
             classes.append(className)
             path=os.path.join(train_path,file)

             if className==i:
                  img=cv2.imread(path,0)
                  img = cv2.resize(img,dim, interpolation = cv2.INTER_AREA)
                  img=torch.from_numpy(img).float()
                  img=img.view(1,1,dim[0],dim[1])
                  imgList.append(img)

             else:
                i+=1
                train_data_class.append(torch.stack(imgList))
                imgList=[]
         train_data_class.append(torch.stack(imgList))

         classes=set(classes)

         # generate balanced pairs
         self.train_data = []
         self.train_labels = []
         count=0
         class_len=len(classes)
         for i in range(0,class_len):
            l=len(train_data_class[i])
            for j in range(l-1): # counter to keep track of image pairs to be created
               rnd_cls = random.randint(0,class_len-2) # choose random class that is not the same class .
               if rnd_cls >= i:
                  rnd_cls = rnd_cls + 1

               kval=len(train_data_class[rnd_cls])
               k= random.randint(1, kval-1)
               self.train_data.append(torch.stack([train_data_class[i][j], train_data_class[i][j+1], train_data_class[rnd_cls][k]]))
               self.train_labels.append([1,0])

         self.train_data = torch.stack(self.train_data)
         self.train_labels = torch.tensor(self.train_labels)

      else:
         test_path=  r'D:/Test/'
         test_data_class=[]
         i=1

         classes=[]
         imgList=[]
         files=os.listdir(test_path)
         files.sort()
         for file in files:
             className= int(file[:3])
             classes.append(className)
             path=os.path.join(test_path,file)

             if className==i:
                  img=cv2.imread(path,0)
                  img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                  img=torch.from_numpy(img).float()
                  img=img.view(1,1,dim[0],dim[1])
                  imgList.append(img)

             else:
                i+=1
                if(imgList):
                    test_data_class.append(torch.stack(imgList))
                imgList=[]

         if imgList:
            test_data_class.append(torch.stack(imgList))

         classes=set(classes)
         # generate balanced pairs
         self.test_data = []
         self.test_labels = []
         count=0
         class_len=len(classes)
         for i in range(0,class_len):
            l=len(test_data_class[i])
            for j in range(l-1): # counter to keep track of image pairs to be created
               rnd_cls = random.randint(0,class_len-2) # choose random class that is not the same class
               if rnd_cls >= i:
                  rnd_cls = rnd_cls + 1

               kval=len(test_data_class[rnd_cls])
               k= random.randint(1, kval-1)

               self.image_names=[[i+1,j+1],[rnd_cls+1,k+1]]
               self.image_names=torch.from_numpy(np.asarray(self.image_names))
               #creating triplets of the form(image1, image_similar_to_image1, image_different_from_image1)
               self.test_data.append(torch.stack([test_data_class[i][j], test_data_class[i][j+1], test_data_class[rnd_cls][k]]))
               self.test_labels.append([1,0])

         self.test_data = torch.stack(self.test_data)
         self.test_labels = torch.tensor(self.test_labels)


   def __getitem__(self, index):
      if self.train:
         imgs, target = self.train_data[index], self.train_labels[index]
      else:
         imgs, target = self.test_data[index], self.test_labels[index]

      img_ar = []
      for i in range(len(imgs)):
         temp=imgs[i].view(dim[0],dim[1])
         img = Image.fromarray(temp.numpy(), mode='L') #from array returns PIL object

         if self.transform is not None:
            img = self.transform(img)
         img_ar.append(img)

      if self.target_transform is not None:
         target = self.target_transform(target)

      if pred:
          return img_ar, target, self.image_names, imgs
      else:
         return img_ar, target

   def __len__(self):
      if self.train:
         return len(self.train_data)
      else:
         return len(self.test_data)

   def _check_exists(self):
      return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
         os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))


   def __repr__(self):

      fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
      fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
      tmp = 'train' if self.train is True else 'test'
      fmt_str += '    Split: {}\n'.format(tmp)
      fmt_str += '    Root Location: {}\n'.format(self.root)
      tmp = '    Transforms (if any): '
      fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
      tmp = '    Target Transforms (if any): '
      fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
      return fmt_str

# defning the Model structure and forward pass
class Net(nn.Module):
   def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(1, 128, 7) #default padding=0, stride=1
      self.pool1 = nn.MaxPool2d(2) #does same as maxpool(2,stride=2) and maxpool(1,stride=2). By default stride value=kernel size
      self.conv2 = nn.Conv2d(128, 64, 5)
      self.conv3 = nn.Conv2d(64, 64, 5)
      self.conv4 = nn.Conv2d(64, 32, 5)
      self.conv5 = nn.Conv2d(32, 16, 5)
      self.linear1 = nn.Linear(43264, 512) #nn.Linear(121104,512) # for 190*190 -->82944 #for 150*150--> 43264
      self.linear2 = nn.Linear(512, 2)

   def forward(self, data):
      res = []
      global feature
      feature=[]
      for i in range(2): # Siamese nets; sharing weights
         x = data[i]
         x = self.conv1(x)
         if(pred):
             feature.append(x)
         x = F.relu(x)
         x = self.pool1(x)
         x = self.conv2(x)
         x = F.relu(x)
         x = self.conv3(x)
         x = F.relu(x)
         x = self.conv3(x)
         x = F.relu(x)
         x = self.conv4(x)
         x = F.relu(x)
         x = self.conv5(x)
         x = F.relu(x)
         x = x.view(x.shape[0], -1)
         x = self.linear1(x)
         res.append(F.relu(x))

      res = torch.abs(res[1] - res[0])
      res = self.linear2(res)
      return res

def oneshot(model, device, data):
   model.eval()
   with torch.no_grad():
      for i in range(len(data)):
            data[i] = data[i].to(device)

      output = model(data)
      return torch.squeeze(torch.argmax(output, dim=1)).cpu().item()

def displayChangeMap(img1,img2):

    dim=img1.shape[:2]
    image1 = cv2.resize(img1, (dim[1],dim[0]))
    image2 = cv2.resize(img2, (dim[1],dim[0]))

    # compute difference
    difference = cv2.subtract(image2, image1)

    # color the mask red
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
    difference[mask != 255] = [0, 0, 255]

    # add the red mask to the images to make the differences obvious
    image1[mask != 255] = [0, 0, 255]
    image2[mask != 255] = [0, 0, 255]

    # store images
    cv2.imwrite("./static/images/output.jpg",image2)
    # plt.imshow(image2)
    # plt.show()

def alignment(img1_color,img2_color):
    # Open the image files.
    #img1_color = cv2.imread(r'D:\\Final year project\\Siamese CNN\\Images\\pano14.jpg')  # Image to be aligned.
    #img2_color = cv2.imread(r'D:\\Final year project\\Siamese CNN\\Images\\pano13.jpg')    # Reference image.

    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not reqiured in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key = lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches)*90)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
      p1[i, :] = kp1[matches[i].queryIdx].pt
      p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1_color,
                        homography, (width, height))

    # Save the output.
    plt.title("Image before alignment")
    plt.imshow(img1_color)
    plt.show()

    plt.title("Image after alignment")
    plt.imshow(transformed_img)
    plt.show()


    return transformed_img


def getChangeOutput(imag1,imag2):
      device = torch.device('cuda')
      trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

      model = Net().to(device)
      global pred
      pred=True
      images=[]
      model=torch.load('./changedetection/siamese5_009.pt')
      data = []
      #resizing and converting each image into a torch tensor
      img1 = cv2.resize(imag1, dim, interpolation = cv2.INTER_AREA)
      data0 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
      data0=torch.from_numpy(data0).float()
      data0=data0.view(1,1,dim[0],dim[1])
      data.append(data0)
      img2 = cv2.resize(imag2, dim, interpolation = cv2.INTER_AREA)
      data1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
      data1=torch.from_numpy(data1).float()
      data1=data1.view(1,1,dim[0],dim[1])
      data.append(data1)

      #displying images accepted as input
      plt.title("Panorama 1")
      plt.imshow(imag1)
      plt.show()
      plt.title("Panorama2")
      plt.imshow(imag2)
      plt.show()

      #calling oneshot to obtain results
      same = oneshot(model, device, data)
      if same > 0:
         print('These two images are similar')
      else:
         trans=alignment(imag1,imag2)
         displayChangeMap(imag2,trans)
         print('These two images are different')


#function to convert video to panorama

def convert_video_to_panorama(videofile,a):
    if(a==0):
        frames = os.path.join(file_dir, 'frames\\video1')
    else:
        frames = os.path.join(file_dir, 'frames\\video2')
    videofile_name = os.path.join(videoFolder,videofile)
    vidcap = cv2.VideoCapture(videofile_name)
    vlc = 'vlcsnap-'
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            cv2.imwrite(os.path.join(frames,vlc + str(count).zfill(5) +".jpg"), image)     # save frame as JPG file
        return hasFrames
    sec = 0
    frameRate = 0.5 #it will capture image in each 1 second
    count=1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)


    #beginning of key frames extraction

    def extract_key_frames():
        vlc = 'vlcsnap-'
        if(a==0):
            img_for_size_path = os.path.join(file_dir ,'frames\\video1')
            img_path = os.path.join(file_dir ,'frames\\video1')
        else:
            img_for_size_path = os.path.join(file_dir ,'frames\\video2')
            img_path = os.path.join(file_dir ,'frames\\video2')

        img_name_for_size = os.path.join(img_for_size_path ,vlc + str(1).zfill(5) + '.jpg')

        img_for_size = cv2.imread(img_name_for_size,0)
        width, height = img_for_size.shape

        index=0
        index_diff=0
        diff = np.zeros((3500,1))
        key_frames = []
        batch_size = 2

        for i in range(batch_size,count-1,batch_size):
            gray = np.zeros((width,height,batch_size), np.uint8)
            index=0
            mean_img = np.zeros((width,height), np.float64)
            for j in range(i-1,i+1,1):
                img_name = os.path.join(img_path , vlc + str(j).zfill(5) + '.jpg')
                #print(img_name)
                gray[:,:,index] = cv2.imread(img_name, 0)
                mean_img += gray[:,:,index]
                index+=1

            mean_img = mean_img/float(batch_size)
            diff2 = np.zeros((batch_size,1))
            for k in range(batch_size):
                diff2[k,0] = np.sum((gray[:,:,k] - mean_img)**2)
            diff2 = diff2/(height*width)
            diff[index_diff:index_diff + batch_size] = diff2
            index_diff+=batch_size
            if np.all(diff2) == True:
                min_ = np.where(diff2 == np.min(diff2))[0][0]
                key_frames.append(cv2.imread( os.path.join(img_path, vlc+ str(i-batch_size + min_+1).zfill(5) + '.jpg')))

        return key_frames




    key_frames = extract_key_frames()
    if(a==0):
        key_frames_path = os.path.join(file_dir, 'key-frames\\video1')
    else:
        key_frames_path = os.path.join(file_dir, 'key-frames\\video2')
    for i in range(0,len(key_frames)-1,1):

        result = key_frames[i]
        cv2.imwrite(os.path.join(key_frames_path, str(i) + '.jpg'), result)
    # print("Key frames extracted")
    #end of key frames extraction

    #collecting key frames

    images=[]
    for filename in os.listdir(key_frames_path):
        img = cv2.imread(os.path.join(key_frames_path,filename),cv2.IMREAD_COLOR)
        images.append(img)

    #Beginnig of panorama stitching

    stitcher = cv2.Stitcher_create()
    ret,pano = stitcher.stitch(images)
    panorama_path = os.path.join(file_dir, 'result')
    if ret==cv2.STITCHER_OK:
        cv2.imwrite(os.path.join(panorama_path, 'pano' + str(a) +'.jpg'), pano)
        cv2.destroyAllWindows()
    else:
         print("Error during Stitching")

    return pano

#Beginning of video to frames

file_dir = os.path.dirname(os.path.realpath('.\\media\\media\\video'))
videoFolder = os.path.join(file_dir, 'video')
a = 0
images=[]
for videofile in os.listdir(videoFolder):
    pano=convert_video_to_panorama(videofile,a)          #calling convert_video_to_panorama function
    a = a+1
    #appending image vectors to an array to pass it as functon parameters later to other functions
    images.append(pano)


getChangeOutput(images[1],images[0]) #with_obj, original
