# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 21:03:37 2019

@author: Sree Nori
@date: 11/3/2019

1. Introduction of this script
This script takes the sense list identified by "decision-list.py" (stored in "my-line-answers.txt") as input, and compares it with 
the gold standard "key" data in "line-answers.txt". And then calculate the accuracy and create a confusion matrix (for error analysis) 
based on the difference between the sense list and the "key" data.

2. Algorithm 
    -> 1) import "my-line-answers.txt" and and the "key" data "line-answers.txt".
    
    -> 2) extract the sense lists from these 2 files respectively.
    
    -> 3)compare these two lists.
         -> calculate how many products have been correctly identified as products
         -> calculate how many products have been misidentified as phones
         -> calculate how many phones have been correctly identified as phones
         -> calculate how many phones have been misidentified as products
    
    -> 3) calculate the accuracy by adding up the number of correctly identified products and phones, 
          and divide it by the total number of senses
    
    -> 4) create the confusion matrix based on step 3).  
        
    -> 5) calculate the baseline accuracy. 
          From "decision-list.py", we can see "product" has higher occurrence than "phone" in the training dataset. 
          Therefore, all senses in the test dataset should be identified as "product", which means in "key" data "line-answers.txt", 
          all senses equal "product" should be considered correctly identified, and all senses equal "phone" should be considered misidentified. 

3. How to run the scripts
- please place "decision-list.py" and "scorer.py" in the same directory
3.1 decision-list.py
    - Create a folder called "PA3" in the same directory with "decision-list.py"
    - Put these 3 files under the folder "PA3": line-train.xml, line-test.xml, line-answers.txt
    - Open terminator, for example: Windows "Command Prompt"
    - Navigate to the directory of "decision-list.py"
    - Enter "python decision-list.py line-train.xml line-test.xml my-decision-list.txt my-line-answers.txt" and execute the script
    Output: 
        - In the terminator, it will show answer tags of the test data.
          For example: <answer instance="line-n.w7_057:1203:" senseid ="phone"/>
        - It will create a file named "my-line-answers.txt" under the folder "PA3". It stores the answers(senses) for test data.
        - It will also create a file named "my-decision-list.txt" under the folder "PA3". It stores each feature,
          the log-likelihood score associated with it, the sense it predicts and the instance ID. 

3.2 scorer.py
    - Open terminator, for example: Windows "Command Prompt"
    - Navigate to the directory of "scorer.py"
    - Enter "python scorer.py my-line-answers.txt line-answers.txt" and execute the script
    Output: 
        - It will show the base line accuracy and my overall accuray
        - It will also show the confusion matrix 
        
"""
import sys
import pandas as pd
from nltk.tokenize import word_tokenize
import logging

log = "decision-list-log.txt"   #log file
logging.basicConfig(filename = log, level = logging.DEBUG, format = '%(message)s')



# get and return input file names            
def get_inputs():
    input_len = len(sys.argv) - 1    # subtract scorer.py
    
    if input_len < 2:        # check whether there are less than 4 arguments, if not, exit the script
        print("Please enter at least 2 arguments.")
        sys.exit()
      
    myAnswers = str(sys.argv[1])             # get the 1st argument: line-train.xml
    goldKey = str(sys.argv[2])              # get the 2nd argument: line-test.xml

    return myAnswers, goldKey

# get and return the sense list based on the input files
def get_sense_list(file_name):    
    sense_list = []
    with open (('./PA3/' + file_name), errors = 'ignore') as f:       
        lines = f.read().splitlines()
        for line in lines:
            line = line.replace("\"", " ")
            line = line.replace("/", "")
            line = line.replace(">", "")
            token = word_tokenize(line)
            sense = token[len(token) - 1]
            sense_list.append(sense)
           
    #print(sense_list)
    return sense_list


# calculate the base line accuracy. 
# based on the training dataset, product has higher occurrence than phone
# so product should be assigned to each senseID in test dataset as the baseline
# in test dataset, the sense = product would be classified as correct identification     
def cal_base_accuracy(goldKey_list):
    product_count = goldKey_list.count("product")
    #phone_count = goldKey_list.count("phone")
    
    baseline_accuracy = product_count/len(goldKey_list)

    print()
    print("Baseline accuracy: ", '{:.3%}'.format(baseline_accuracy))
    print()


# calculate my overall accuracy
def cal_my_accuracy(myAnswers_list, goldKey_list):

    product_correct = 0
    product_error = 0
    
    phone_correct = 0
    phone_error = 0
    
    for i in range(0, len(goldKey_list)):
        if goldKey_list[i] == myAnswers_list[i]:
            if goldKey_list[i] == 'product':
                product_correct += 1
            else:
                phone_correct += 1
        else:
            if goldKey_list[i] == 'product':
                product_error += 1
            else:
                phone_error += 1
                
#    print(product_correct, product_error)
#    print(phone_correct, phone_error)
                
    myAccuracy = (product_correct + phone_correct)/len(goldKey_list)
    
    matrix = {
            ' ': ['product', 'phone'],
            'product': [product_correct, phone_error],
            'phone': [product_error, phone_correct]
            }
    

    matrix_df = pd.DataFrame(matrix, columns = [' ', 'product', 'phone'])  

    print("My overall accuracy: ", '{:.3%}'.format(myAccuracy))
    print()                
    print("Confusion matrix: ")
    print(matrix_df)
    

def main():  
    myAnswers, goldKey = get_inputs() 
    
    logging.info('$ python scorer.py %s %s', str(myAnswers), str(goldKey)) 
        
    myAnswers_list = get_sense_list(myAnswers)
    goldKey_list = get_sense_list(goldKey)
    
    cal_base_accuracy(goldKey_list)
    cal_my_accuracy(myAnswers_list, goldKey_list)    
        
    logging.info('$ exit') 
        

      
if __name__ == "__main__":
    main()