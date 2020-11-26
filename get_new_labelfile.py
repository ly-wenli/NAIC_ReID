import os.path as osp
from IPython import embed

def get_new_label(data_dir):
    train_2019_cs_path = osp.join(data_dir,'train_2019_cs')

    filename_train_2019_cs = osp.join(train_2019_cs_path, 'train_list.txt')

    cs_2019equal_list  = []
    # load cs_2020equal.txt
    with open('cs_2019equal_path.txt','r') as fr:
        while True:
            lines = fr.readline()
            if not lines:
                break
            eq_img_name = lines.split("/")[-1].replace("\n","")
            cs_2019equal_list.append(eq_img_name)
    
    new_cs_2019_label_list = []
    # load 2020 cs dataset
    with open(filename_train_2019_cs, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            img_name,img_label = [i for i in lines.split(' ')]
            # if img_name == 'train/105180993.png' or img_name=='train/829283568.png' or img_name=='train/943445997.png': # remove samples with wrong label
            #     continue
            same_falg = False
            for s in cs_2019equal_list:
    
               
                if s == img_name.split('/')[-1]:
                    same_falg = True
                    break
            if same_falg:
                continue
            else:
                new_cs_2019_label_list.append(img_name+" "+img_label)
    print(len(new_cs_2019_label_list))
    
    new_train_list = "new_2019_cs_train_list.txt"
    with open(new_train_list,'a') as f:
        f.writelines(new_cs_2019_label_list)
            


if __name__ == "__main__":
    get_new_label('/home/wenli/data/MyDataSet')
