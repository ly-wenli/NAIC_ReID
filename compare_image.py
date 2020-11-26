from PIL import Image
from PIL import ImageChops
import os.path as osp
from tqdm import tqdm
import base64
from collections import defaultdict
import json

def compare_str(str_one,str_two,flag='cs_2020'):
    if str_one['b64str'] == str_two['b64str']:
        f = flag + 'equal_path.txt'
        with open(f,"a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
                
                file.write('\n'+str_two['path'])

def compare_images(path_one, path_two):
    """
    比较图片
    :param path_one: 第一张图片的路径
    :param path_two: 第二张图片的路径
    :return: 相同返回 success
    """
    image_one = Image.open(path_one)
    image_two = Image.open(path_two)
    try:
        diff = ImageChops.difference(image_one, image_two)

        if diff.getbbox() is None:
            # 图片间没有任何不同则直接退出
            print("比较成功")
            f = "equal_image.txt"

            with open(f,"a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
                
                file.write(path_one)
                

            
        else:
            pass
            #print("比较失败")
            # f = "equal_image.txt"

            # with open(f,"a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
                
            #     file.write(path_one)
            

    except ValueError as e:
        print("宽高不一致")
        


if __name__ == '__main__':
    """
    #---------------------2019 cs--------------------------
    cs_2019_images_name = []
    # 比较2020年初赛数据和复赛数据相同的部分
    train_2019_cs_path = "/home/wenli/data/MyDataSet/train_2019_cs"
    with open(osp.join(train_2019_cs_path, 'train_list.txt'), 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                img_name, _ = [i for i in lines.split(' ')]
                if img_name == 'train/105180993.png' or img_name=='train/829283568.png' or img_name=='train/943445997.png': # remove samples with wrong label
                    continue
                
                #img_name = osp.join('images', img_name)
                img_name = osp.join(train_2019_cs_path, img_name)
                cs_2019_images_name.append(img_name)
    
    b64_2019_cs = []
    for cs_p in tqdm(cs_2019_images_name):
        f = open(cs_p,'rb')
        b64_data = base64.b64encode(f.read())
        s = b64_data.decode()
        b64_2019_cs.append({'path': cs_p ,"b64str": s})
    cs = "cs_b64_2019.json"
    with open(cs,"a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
        
        file.write(json.dumps(b64_2019_cs))
    print(len(cs_2019_images_name))

    #---------------------2019 fs--------------------------
    fs_2019_images_name = []
    # 比较2020年初赛数据和复赛数据相同的部分
    train_2019_fs_path = "/home/wenli/data/MyDataSet/train_2019_fs"
    with open(osp.join(train_2019_fs_path, 'train_list.txt'), 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                img_name, _ = [i for i in lines.split(' ')]
                if img_name == 'train/105180993.png' or img_name=='train/829283568.png' or img_name=='train/943445997.png': # remove samples with wrong label
                    continue
                
                #img_name = osp.join('images', img_name)
                img_name = osp.join(train_2019_fs_path, img_name)
                fs_2019_images_name.append(img_name)
    
    b64_2019_fs = []
    for fs_p in tqdm(fs_2019_images_name):
        f = open(cs_p,'rb')
        b64_data = base64.b64encode(f.read())
        s = b64_data.decode()
        b64_2019_fs.append({'path': fs_p ,"b64str": s})
    fs = "fs_b64_2019.json"
    with open(fs,"a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
        
        file.write(json.dumps(b64_2019_fs))
    print(len(fs_2019_images_name))

    """
    """
    #---------------------2020 cs--------------------------
    cs_images_name = []
    # 比较2020年初赛数据和复赛数据相同的部分
    train_2020_cs_path = "/home/wenli/data/MyDataSet/train_2020_cs"
    with open(osp.join(train_2020_cs_path, 'train_list.txt'), 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                img_name, _ = [i for i in lines.split(':')]
                # if img_name == 'train/105180993.png' or img_name=='train/829283568.png' or img_name=='train/943445997.png': # remove samples with wrong label
                #     continue
                
                img_name = osp.join('images', img_name)
                img_name = osp.join(train_2020_cs_path, img_name)
                cs_images_name.append(img_name)
    
    b64_cs = []
    for cs_p in tqdm(cs_images_name):
        f = open(cs_p,'rb')
        b64_data = base64.b64encode(f.read())
        s = b64_data.decode()
        b64_cs.append({'path': cs_p ,"b64str": s})
    cs = "cs_b64.json"
    with open(cs,"a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
        
        file.write(json.dumps(b64_cs))
    print(len(cs_images_name))


    #---------------------2020 fs--------------------------

    fs_images_name = []
    # 比较2020年初赛数据和复赛数据相同的部分
    train_2020_fs_path = "/home/wenli/data/MyDataSet/train_2020_fs"
    with open(osp.join(train_2020_fs_path, 'train_list.txt'), 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                img_name, _ = [i for i in lines.split(':')]
                # if img_name == 'train/105180993.png' or img_name=='train/829283568.png' or img_name=='train/943445997.png': # remove samples with wrong label
                #     continue
                
                img_name = osp.join('images', img_name)
                img_name = osp.join(train_2020_fs_path, img_name)
                fs_images_name.append(img_name)
    fs_b64_file = "fs_b64_file"

    b64_fs = []
    for fs_p in tqdm(fs_images_name):
        f = open(fs_p,'rb')
        b64_data = base64.b64encode(f.read())
        s = b64_data.decode()
        b64_fs.append({'path': fs_p ,"b64str": s})
    fs = "fs_b64.json"
    with open(fs,"a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
        
        file.write(json.dumps(b64_fs))

    print(len(fs_images_name))

"""
    # for fs_path in tqdm(fs_images_name):
    #     for cs_path in tqdm(cs_images_name):


    #         compare_images(fs_path,cs_path)
    
    
    # image_one = Image.open(fs_images_name[0])
    # image_two = Image.open(cs_images_name[0])
    # f = open("C:\\Users\\wenli\\Desktop\\a.png",'rb')
    # b64_data = base64.b16encode(f.read())
    # s = b64_data.decode()
    # f = open("C:\\Users\\wenli\\Desktop\\aa.png",'rb')
    # b64_data = base64.b16encode(f.read())
    # ss = b64_data.decode()
    # fs = "test_txt.json"

    # with open(fs,"a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
        
    #     file.write(json.dumps("C:\\Users\\wenli\\Desktop\\a.png"+": "+s))

    # print("图片相同" if(s == ss) else "图片不相同")

    #compare_images("C:\\Users\\wenli\\Desktop\\a.png","C:\\Users\\wenli\\Desktop\\aa.png")

    fs_2020_json_path = open('fs_b64.json',encoding='utf-8')
    fs_2020_json = json.load(fs_2020_json_path)
    print("fs_2020_json len is " , len(fs_2020_json))


    cs_2020_json_path = open('cs_b64.json',encoding='utf-8')
    cs_2020_json = json.load(cs_2020_json_path)
    print("cs_2020_json len is " , len(cs_2020_json))


    fs_2019_json_path = open('fs_b64_2019.json',encoding='utf-8')
    fs_2019_json = json.load(fs_2019_json_path)
    print("fs_2019_json len is " , len(fs_2019_json))
    
    cs_2019_json_path = open('cs_b64_2019.json',encoding='utf-8')
    cs_2019_json = json.load(cs_2019_json_path)
    print("cs_2019_json len is " , len(cs_2019_json))
    # base is 2020 fs
    # for fs in tqdm(fs_2020_json):
    #     for cs in cs_2020_json: 
    #         compare_str(fs,cs,flag='cs_2020')

    for fs in tqdm(fs_2020_json):
        for cs in fs_2019_json: 
            compare_str(fs,cs,flag='fs_2019')

    for fs in tqdm(fs_2020_json):
        for cs in cs_2019_json: 
            compare_str(fs,cs,flag='cs_2019')
    # base is 2020 cs
    for fs in tqdm(cs_2020_json):
        for cs in fs_2019_json: 
            compare_str(fs,cs,flag='fs_2019')
    for fs in tqdm(cs_2020_json):
        for cs in cs_2019_json: 
            compare_str(fs,cs,flag='cs_2019')
    
    
    # base is 2019 fs
    for fs in tqdm(fs_2019_json):
        for cs in cs_2019_json: 
            compare_str(fs,cs,flag='cs_2019')

    # dont op


