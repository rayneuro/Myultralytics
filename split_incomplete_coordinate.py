import os
import pandas as pd 


df = pd.DataFrame(columns=['label' , 'x1' ,'x2' ,'x2' ,'y2' , 'x3' , 'y3' , 'x4' ,'y4'])  # 在這個例子中我們只有一個欄位 'Text'
df = pd.DataFrame(columns=[  'x1' ,'x2' ,'x2' ,'y2' , 'x3' , 'y3' , 'x4' ,'y4'])  # 在這個例子中我們只有一個欄位 'Text'

#train_files = [f for f in os.listdir('/home/ray/datasets/10%data/labels/val')]

'''
for file in train_files:
    with open(f'/home/ray/datasets/10%data/labels/val/{file}', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            if line[0] == '1' or line[0] == '0':
                df.loc[len(df)] = [int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]), float(line[7]), float(line[8])]
'''
#print(df)
with open(f'/home/ray/Myultralytics/incomplete_data/test.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            if line[0] == '1' or line[0] == '0':
                df.loc[len(df)] = [int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]), float(line[7]), float(line[8])]



df.to_csv('./test.csv', index=False)
                