'''
Author: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
Date: 2023-03-28 22:39:55
LastEditors: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
LastEditTime: 2023-04-08 22:17:47
FilePath: \知识图谱作业2\dispose_data.py
Description: coded by ZhangYang@BUPT, my email is zhangynag0207@bupt.edu.cn

Copyright (c) 2023 by zhangynag0207@bupt.edu.cn, All Rights Reserved. 
'''
import os
import re
import json
import jieba



def transfer_data(): 
    data = []
    """
    转换地址数据
    """    
    piece = {}
    files = os.listdir("./data/location")  # 得到文件夹下的所有文件名称
    for file in files: # 遍历文件夹
        with open("./data/location/"+file, 'r', encoding='utf-8') as f: # 打开文件
            text = f.read()                 # 读取文件
            text = re.sub("\n", "", text)  # 去除换行符
            piece["label"] = list(range(len(text)))  # 标签、文本长度一致

            # 先将所有标签置为O
            for i in range(len(text)): #
                piece["label"][i] = "O"

            # 进行标签转换
            human_labels_index = [] # 人工标注的标签索引
            for m in re.finditer(r'\s([^\s]*)/LOC', text): # 1.匹配空格 2.匹配非空格内容 3.匹配/LOC
                piece["label"][m.start()+1] = "B-LOCATION" # 地址开始的标签为B-LOCATION
                for i in range(m.start()+2, m.end()-4): # 地址中间的标签为I-LOCATION
                    piece["label"][i] = "I-LOCATION"
                for i in range(m.end()-4, m.end()): # 人工标注的标签索引
                    human_labels_index.append(i)

            # 去除人工标注的标签
            piece["text"] = list(text) # 文本 
            for i in human_labels_index: 
                piece["text"][i] = " " # 将人工标注的标签置为空格
            data.append(piece) # 添加到数据中
            piece = {} 
            
    """
    转换时间数据
    """
    files = os.listdir("./data/time")  
    for file in files:  # 遍历文件夹
        with open("./data/time/"+file, 'r', encoding='utf-8') as f: # 打开文件
            text = f.read() # 读取文件
            text = re.sub("\n", "", text) # 去除换行符
            piece["label"] = list(range(len(text))) # 标签、文本长度一致
            human_labels_index = [] # 人工标注的标签索引

            # 先将所有标签置为O
            for i in range(len(text)): 
                piece["label"][i] = "O"         # 其他标签为O

            # 进行标签转换
            for m in re.finditer(r'\s([^\s]*)/[DT][SO]', text): #1.匹配空格 2.匹配非空格内容 3.匹配/LOC
                piece["label"][m.start()+1] = "B-TIME"       # 时间开始的标签为B-TIME
                for i in range(m.start()+2, m.end()-3):
                    piece["label"][i] = "I-TIME"            # 时间中间的标签为I-TIME
                for i in range(m.end()-3, m.end()):
                    human_labels_index.append(i)           # 人工标注的标签索引

            # 去除人工标注的标签
            piece["text"] = list(text) # 文本
            for i in human_labels_index: #
                piece["text"][i] = " " # 将人工标注的标签置为空格
            
            data.append(piece) # 添加到数据中
            piece = {} # 清空
    
    # 保存数据为json格式
    # json.dump(data, open("./data/data.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    # 保存数据为txt格式
    with open("./data/data.txt", 'w', encoding='utf-8') as f: # 打开文件
        for piece in data: # 遍历数据
            for i in range(len(piece["text"])): # 遍历文本
                f.write(piece["text"][i] + " " + piece["label"][i] + "\n") # 写入文本和标签
            f.write("\n") # 写入换行符

    return data # 返回数据

def add_line_sp():
    # 添加分隔符(每个30行加一次)
    f = open('./data/data.txt','r',encoding='utf-8')
    f2 = open('./data/total_ner2.txt','w',encoding='utf-8')
    tag_list = ['。','！','？','；']
    for line in f:
        f2.write(line)
        line_sps = line.strip().split(' ')
        if line_sps[0] in tag_list:
            line_sp = line.strip().split(' ')
            if len(line_sp) == 2:
                if line_sp[1] == 'O':
                    f2.write('\n')
                    line_num = 0
            else:
                    f2.write('\n')
                    line_num = 0
    print('成功添加分隔符！')

def spilt_data():
    print('\n')
    print('开始切割ner的total_ner2.txt数据：')
    # f_path = 'ner_data/2/test' + str(k) + '.txt'
    f_path = './data/total_ner2.txt'
    train_path = './split_data/train.txt'
    valid_path = './split_data/dev.txt'
    test_path = './split_data/test.txt'
    f = open(f_path,'r',encoding='utf-8')
    f1 = open(f_path, 'r', encoding='utf-8')
    f_train = open(train_path,'w',encoding='utf-8')
    f_valid = open(valid_path, 'w', encoding='utf-8')
    f_test = open(test_path, 'w', encoding='utf-8')
    line_num = 0
    for line in f:
        line_num += 1
    print(line_num)
    train_num = int(line_num * 0.7)
    # train_num = int(line_num * 0.8)  # 训练集为8：1：1时使用该句
    print(train_num)
    valid_num = int(line_num * 0.1)
    print(valid_num)
    # test_num =
    count_num = 0
    count_1 = 0
    for line_train in f1:
        count_num += 1
        if count_num <= train_num :
            # count_1 +=1
            f_train.write(line_train)
        else:
            line_sp = line_train.strip().split(' ')
            if len(line_sp) == 3:
                    f_train.write(line_train)
            else:
                break
    for line_valid in f1:
        count_1 += 1
        if count_1 <= valid_num :
            # count_1 +=1
            f_valid.write(line_valid)
        else:
            line_sp = line_valid.strip().split(' ')
            if len(line_sp) == 3:
                f_valid.write(line_valid)
            else:
                break
    for line_test in f1:
        f_test.write(line_test)
    print('切割完成！')

def modify_data():
    
    transfer_data()       

    add_line_sp() 

    spilt_data()
    