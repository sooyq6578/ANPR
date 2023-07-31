#%%
# Group06 testing
# ID: 31989632, Tuan Muhammad Zafri
# ID: 31861393, Ong Yi See
# ID: 32457375, Soo Yong Qi

import os
import random
import math
import numpy as np
import glob
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import cv2


class Vertex: # class for neurons
    def __init__(self, id, input=0):
        """attributes for every single neuron"""
        self.id = id
        self.input = input
        self.output = 0
        self.bias = 0
        self.target=0
        self.error = 0

    def __str__(self):
        return_str = str(self.id)
        return return_str


class Ann:

    def __init__(self, img_arr, no_of_hidden, no_of_output, target_arr, set_error, set_itteration):
        self.total_error = 0
        self.no_of_input = len(img_arr)*len(img_arr[0])
        self.no_of_hidden = no_of_hidden
        self.no_of_output = no_of_output
        self.target_arr=target_arr
        self.set_error=set_error
        self.iteration=0 #reset iteration to 0
        self.set_itteration=set_itteration
        


        #====initialze input layer to be every single pixel of the image===
        self.input=[]
        for i in range(len(img_arr)):
            for j in range(len(img_arr[0])):
                # print(img_arr[i][j],'jkkk')
                self.input.append(Vertex(j, input=img_arr[i][j])) # input is the pixel value

        #====initialze hidden neurons===
        self.hidden=[None] * self.no_of_hidden
        for i in range(self.no_of_hidden): 
            self.hidden[i]=Vertex(i)
           
         #====initialze output neurons===
        self.output=[None] * self.no_of_output
        for i in range(self.no_of_output): 
            self.output[i]=Vertex(i)
       
        
  
        
    def Weight_Initialization(self, saved):
        """Initialize weights and biases, if saved==True, used np.load() to load previously saved weights and biases"""

        np.random.seed(1)
        if saved:
            self.wji = np.load('wji.npy')
            self.wjk = np.load('wjk.npy')
            self.bias_j = np.load('bias_j.npy')
            self.bias_k = np.load('bias_k.npy')
        else:
            self.wji = np.random.uniform(-0.5, 0.5, size=(self.no_of_hidden, self.no_of_input))
            self.wjk = np.random.uniform(-0.5, 0.5, size=(self.no_of_output, self.no_of_hidden))
            self.bias_j = np.random.uniform(0, 1, size=(self.no_of_hidden, 1))
            self.bias_k = np.random.uniform(0, 1, size=(self.no_of_output, 1))
    
        self.delta_wji = np.zeros((self.no_of_hidden, self.no_of_input))
        self.delta_wjk = np.zeros((self.no_of_output, self.no_of_hidden))
        self.delta_bias_j = np.zeros((self.no_of_hidden, 1))
        self.delta_bias_k = np.zeros((self.no_of_output, 1))
    

        #======assign target array to respective outputs========
        for k in range(self.no_of_output):
            if self.target_arr is not None:
                self.output[k].target=self.target_arr[k]
            

    def Saving_Weights_Bias(self):
        np.save('wji.npy', self.wji)
        np.save('wjk.npy', self.wjk)
        np.save('bias_j.npy', self.bias_j)
        np.save('bias_k.npy', self.bias_k)
    

    def Forward_Input_Hidden(self):
        for j in range(self.no_of_hidden):
            for i in range(self.no_of_input):
                self.hidden[j].input += ( self.wji[j][i] * self.input[i].input )
            self.hidden[j].output = 1 / (1 + math.exp(-self.hidden[j].input - self.bias_j[j]))


    def Forward_Hidden_Output(self):
        for k in range(self.no_of_output):
            for j in range(self.no_of_hidden):
                self.output[k].input += ( self.wjk[k][j] * self.hidden[j].output )
            self.output[k].output = 1 / (1 + math.exp(-self.output[k].input - self.bias_k[k]))


    def forward_error(self):
        self.total_error=0
        for k in range(self.no_of_output):
            self.output[k].error = 0.5 * ( (self.output[k].target - self.output[k].output)**2 )
            # print(self.output[k].output,'forward_error')
            self.total_error += self.output[k].error
        # print(self.total_error,'total_error')


    def Weight_Bias_Correction_Output(self):
        for k in range(self.no_of_output):
            for j in range(self.no_of_hidden):

                out_k = self.output[k].output
                delta_wjk = (out_k -self.output[k].target) * (out_k-out_k**2) * self.hidden[j].output
                self.delta_wjk[k][j] = delta_wjk

            delta_bias_k = (out_k -self.output[k].target) * (out_k-out_k**2)
            self.delta_bias_k[k] = delta_bias_k
                

    def Weight_Bias_Correction_Hidden(self):
        for j in range(self.no_of_hidden):
            for i in range(self.no_of_input):
                firsthalf_formula=0
                for k in range(self.no_of_output):
                    out_k = self.output[k].output
                    
                    firsthalf_formula += (out_k-self.output[k].target)*(out_k-out_k**2)*self.wjk[k][j]
                
                delta_wji = firsthalf_formula * self.hidden[j].output*(1-self.hidden[j].output) * self.input[i].input
                self.delta_wji[j][i] = delta_wji

            delta_bias_j = firsthalf_formula * self.hidden[j].output * (1-self.hidden[j].output)
            self.delta_bias_j[j] = delta_bias_j

            

    def Weight_Bias_Update(self):
        for j in range(self.no_of_hidden):
            for i in range(self.no_of_input):
                self.wji[j][i] -= 0.5*self.delta_wji[j][i]
            self.bias_j[j] -= 0.5*self.delta_bias_j[j]

        for k in range(self.no_of_output):
            for j in range(self.no_of_hidden):
                self.wjk[k][j] -= 0.5*self.delta_wjk[k][j]
            self.bias_k[k] -= 0.5*self.delta_bias_k[k]

    

    def reset(self): #reset input and output of hidden and output neurons to 0
        for j in range(self.no_of_hidden):
            self.hidden[j].input=0
            self.hidden[j].output=0
        for k in range(self.no_of_output):
            self.output[k].input=0
            self.output[k].output=0


    def add_iteration(self):
        self.iteration+=1


    def Check_for_End(self):
        """stop when error is less then set_error, number of iterations will be checked at the main_train() while loop"""
        self.forward_error()
        if self.total_error < self.set_error or self.iteration > self.set_itteration:
            self.check_training_output()
            return True
        else:
            return False
    
    def check_training_output(self):
        # final_output=[]
        target_output=[]
        for k in range(self.no_of_output):
            # print(self.output[k].output, 'self.output[k].output')
            # final_output.append(self.output[k].output)
            if self.output[k].output >= 0.9 and self.output[k].target==1:
                target_output.append(True)
            elif self.output[k].output < 0.1 and self.output[k].target==0:
                target_output.append(True)
        if len(target_output)==20:
            print(len(target_output), 'len(target_output)==20, pass')
        else:
            for k in range(self.no_of_output):
                print(self.output[k].output, 'self.output[k].output')
            print(target_output, 'target_output')
            print('FAIL')

             


def Read_Files(file):
    """Reads the image file and preprocess them to be fed to the ann"""
    img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    img = Image.fromarray(img)
    img = ImageEnhance.Brightness(img).enhance(1.2)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    img = np.array(img)
    #white_masked = w_mask(img, 100, 255, 0)
    _, white_masked = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(white_masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    char_cnt = np.zeros((img.shape), dtype=np.uint8)

    cv2.drawContours(char_cnt, contours, -1, 1, thickness=cv2.FILLED)

    #Image.fromarray(char_cnt).show()
    char_cnt[char_cnt > 0] = 1
    return char_cnt

#dictionary to map characters to their index in target array
target_dict = {'B':10,
               'F':11,
               'L':12,
               'M':13,
               'P':14,
               'Q':15,
               'T':16,
               'U':17,
               'V':18,
               'W':19,
               '0':0,
               '1':1,
               '2':2,
               '3':3,
               '4':4,
               '5':5,
               '6':6,
               '7':7,
               '8':8,
               '9':9,
               '10':'B',
               '11':'F',
               '12':'L',
               '13':'M',
               '14':'P',
               '15':'Q',
               '16':'T',
               '17':'U',
               '18':'V',
               '19':'W'}

alphabets=['B','F','L','M','P','Q','T','U','V','W'] #alphabets to be classified


def main_train(no_of_hidden, saved, set_error, set_itteration):
    random.seed(1)
    image_paths = []
    for folder in range(10): #append all numbers in number dataset to image_paths
        folder_path = f"number dataset/{folder}/"  
        image_paths.extend(glob.glob(folder_path + "*.jpg"))  

    for alphab in alphabets: #append all alphabets in alphabets dataset to image_paths
        folder_path = f"Alphab dataset/{alphab}/"  
        image_paths.extend(glob.glob(folder_path + "*.png"))  
  
        
    random.shuffle(image_paths) #randomize the order of inputs to be presented to the neural network

    first_time=True #to initialize weights and biases for the very first time, others all used the saved weights and biases

    for image in image_paths:
            print(image)
            img_arr = Read_Files(image)
            
            #====to skip 9th and 10th images for training, as we only want 80% data========
            imageNumber = image.split("_")
            if len(imageNumber) < 2:
                imageNumber = image.split("-")

            imageNumber = int(imageNumber[1].split(".")[0])
            if imageNumber==9 or imageNumber==10:
                continue

            #====to assign targets based on input========
            # the order of target array['0','1','2','3','4','5','6','7','8','9','B','F','L','M','P','Q','T','U','V','W']
            # eg. if input is 'B' target[10]=1, if input is '3', target[3]=1 etc. and the rest are 0
            target = [0]*20
            if image[15].isalpha():
                target_index = target_dict[image[15]]
            else:
                target_index = int(image[15])
            target[target_index] = 1
           
            #====to feed the current input with correspinding targets to the ann========
            ann = Ann(img_arr, no_of_hidden, len(target), target, set_error, set_itteration)
      
            #====only initialze weights to be random for the very first iteration====
            if first_time and saved==False:
                ann.Weight_Initialization(saved=False)
                first_time=False
            else:
                ann.Weight_Initialization(saved=True)

           
     
            # iteration=0 #to count the number of iterations
            flag=False
            while not flag:
                # iteration+=1
                ann.add_iteration()
                ann.Forward_Input_Hidden()
                ann.Forward_Hidden_Output()
                flag = ann.Check_for_End()
                if flag==True:
                    break
                ann.Weight_Bias_Correction_Output()
                ann.Weight_Bias_Correction_Hidden()
                ann.Weight_Bias_Update()
                ann.reset()

                # if iteration == 200: #break when error is lower than set_error or after 100 iterations
                #     break

            ann.Saving_Weights_Bias()

 


def main_test_number(no_of_hidden): #test on number dataset
    accuracy=0
  
    for i in range(10):
        folder = f'number dataset/{i}'

        target = [0]*20
        target[i] = 1
        for j in range(9,11):
            print(i,j)
            img_arr = Read_Files(folder+f'/{i}-{j}.jpg')
    
            ann = Ann(img_arr, no_of_hidden, len(target), target, None, None)
      
            # only forward propagation once
            ann.Weight_Initialization(saved=True)
            ann.Forward_Input_Hidden()
            ann.Forward_Hidden_Output()

            # to get the highest output between the 20 output neurons and check if the assigned target==1, if true, means correctly classified
            max_output = -math.inf
            output_neuron_id = None

            for k in range(len(ann.output)):
                # print(ann.output[k].output,ann.output[k].target,'output')
                if ann.output[k].output > max_output:
                    max_output = ann.output[k].output
                    output_neuron_id = k

            # print(ann.output[output_neuron_id].output , ann.output[output_neuron_id].target,'accuracy')
            if ann.output[output_neuron_id].target == 1:
                accuracy+=1
            else:
                print(ann.output[output_neuron_id].output , ann.output[output_neuron_id].target,'accuracy', output_neuron_id)
                print("FAIL")
            
            ann.reset()
            
    print(accuracy)
    print(accuracy/20*100,'%')
    return accuracy

def main_test_alphab(no_of_hidden): #test on alphabets dataset
    accuracy=0
    
    for alphab in alphabets:
        folder = f'Alphab dataset/{alphab}'

        target = [0]*20
        target_index = target_dict[alphab]
        target[target_index] = 1
        for j in range(9,11):
            print(alphab,j)
            img_arr = Read_Files(folder+f'/{alphab}_{j}.png')
            
            ann = Ann(img_arr, no_of_hidden, len(target), target, None, None)
       
            # only forward propagation once
            ann.Weight_Initialization(saved=True)
            ann.Forward_Input_Hidden()
            ann.Forward_Hidden_Output()

            # to get the highest output between the 20 output neurons and check if the assigned target==1, if true, means correctly classified
            max_output = -math.inf
            output_neuron_id = None

            for k in range(len(ann.output)):
                # print(ann.output[k].output,ann.output[k].target,'output')
                if ann.output[k].output > max_output:
                    max_output = ann.output[k].output
                    output_neuron_id = k

            # print(ann.output[output_neuron_id].output , ann.output[output_neuron_id].target,'accuracy')
            if ann.output[output_neuron_id].target == 1:
                accuracy+=1
            else:
                print(ann.output[output_neuron_id].output , ann.output[output_neuron_id].target,'accuracy',output_neuron_id)
                print("FAIL")
            
            ann.reset()
            
    print(accuracy)
    print(accuracy/20*100,'%')
    return accuracy

def main_test_carnum(no_of_hidden):
    accuracy=0
    image_paths = []
    for i in range(1,11):
        folder = f'carnum dataset/{i}/'
        image_paths.extend(glob.glob(folder + "*.png"))  
    
    for image in image_paths:
        img_arr = Read_Files(image)
        
        ann = Ann(img_arr, no_of_hidden, 20, None, None, None)

        # only forward propagation once
        ann.Weight_Initialization(saved=True)
        ann.Forward_Input_Hidden()
        ann.Forward_Hidden_Output()
        
       
        max_output = -math.inf
        output_neuron_id = None

        # print(image.split('\\')[0], image.split('\\')[1][4]) #windows
        print(image.split('/')[1][0], image.split('/')[2][4]) #mac 
        for k in range(len(ann.output)):
            # print(ann.output[k].output,target_dict[image.split('/')[2][4]],'output')
            
            if ann.output[k].output > max_output:
                max_output = ann.output[k].output
                output_neuron_id = k
        
        # print(ann.output[output_neuron_id].output,output_neuron_id,target_dict[image.split('/')[2][4]],'accuracy')
       
        # if output_neuron_id == target_dict[image.split('\\')[1][4]]: #windows 
        if output_neuron_id == target_dict[image.split('/')[2][4]]: #the dictionary will correspond to the character represented by the index in the target array, so [0] will be '0', [10] will be 'B' etc.
            accuracy+=1
        else:
            print(ann.output[output_neuron_id].output,target_dict[image.split('/')[2][4]],'accuracy', output_neuron_id)
            print("FAIL")
        
        ann.reset()
            
    print(accuracy)
    print(accuracy/69*100,'%')
    return accuracy

# parameters to tune
no_of_hidden = 160
epoch = 10
set_error = 0.001
set_itteration=100

#for training
# main_train(no_of_hidden, False, set_error, set_itteration) #trainining for the first epoch
# for i in range(epoch):
#     print('haha',i)
#     main_train(no_of_hidden, True, set_error, set_itteration) #training from the second epoch onwards
main_test_number(no_of_hidden) #testing on number dataset
main_test_alphab(no_of_hidden) #testing on alphabets dataset
# main_test_carnum(no_of_hidden) #testing on car number plate dataset

#%%

# no_of_hidden = 160
# epoch = 10
# set_error = 0.015
# accuracy = 64/68 

# no_of_hidden = 150
# epoch = 10
# set_error = 0.01
# accuracy = ?/68 

# no_of_hidden = 165
# epoch = 10
# set_error = 0.01
# accuracy = ?/68 

# no_of_hidden = 170
# epoch = 10
# set_error = 0.01
# accuracy = ?/68 

