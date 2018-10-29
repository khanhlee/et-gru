# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 19:36:48 2018

@author: khanhle
"""
import time
import pandas as pd

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def load_text_file(file_text):
    start_time = time.time()
    with open(file_text) as f:
        lines = f.readlines()
    print("Loading time: ", time.time() - start_time)
    #print "nb lines: ", len(lines)
    return lines  

def created_chunk(source_file, out_file):
    lst_file = load_text_file(source_file)
    a = chunks(lst_file, 5)
    result = ''
    for x in a:
        for y in x:
            result += y.rstrip('\n') + ','
        result += '\n'
    f = open(out_file, 'w')
    f.write(result)
    f.close()

def separate_cv(chunk_file, label):
    df = pd.read_csv(chunk_file, header=None)
    for i in range(0, 5):
        try:
            # write training file
            train = df[df.columns.difference([i])]
            train1 = pd.Series(train.values.ravel('F'))
            print('Training: ', train1)
            train1.to_csv(label + '_trn' + str(i) + '.csv', header=False, index=False)
            test = df[df.columns[i]]
            print('Testing: ', test)
            test.to_csv(label + '_tst' + str(i) + '.csv', header=False, index=False)
        except:
            pass