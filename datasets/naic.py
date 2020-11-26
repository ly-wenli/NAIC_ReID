from .bases import BaseImageDataset
import os.path as osp
from collections import defaultdict


class NAIC(BaseImageDataset):
    def __init__(self,cfg, root='../data', verbose = True):
        super(NAIC, self).__init__()
        self.cfg = cfg
        self.dataset_dir = root
        self.mydataset_dir = osp.join(self.dataset_dir,'MyDataSet')
        # self.dataset_dir_train = osp.join(self.mydataset_dir, 'train')
        # self.dataset_dir_train_2019_cs = osp.join(self.mydataset_dir, 'train_2019_cs')
        # self.dataset_dir_train_2019_fs = osp.join(self.mydataset_dir, 'train_2019_fs')
        self.dataset_dir_test = osp.join(self.mydataset_dir, 'image_B_v1_1')

        train = self._process_dir(self.mydataset_dir, relabel=True)
        query_green, query_normal = self._process_dir_test(self.dataset_dir_test,  query = True)
        gallery_green, gallery_normal = self._process_dir_test(self.dataset_dir_test, query = False)
        if verbose:
            print("=> NAIC Competition data loaded")
            self.print_dataset_statistics(train, query_green+query_normal, gallery_green+gallery_normal)

        self.train = train
        self.query_green = query_green
        self.gallery_green = gallery_green
        self.query_normal = query_normal
        self.gallery_normal = gallery_normal

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)


    def _process_dir(self, data_dir, relabel=True): # train_2020_cs_path

        train_2020_cs_path = osp.join(data_dir,'train_2020_cs')
        train_2020_fs_path = osp.join(data_dir,'train_2020_fs')
        train_2019_cs_path = osp.join(data_dir,'train_2019_cs')
        train_2019_fs_path = osp.join(data_dir,'train_2019_fs')
        train_2020_fs_unlabel_path = osp.join(data_dir,'unlabel')
        filename_train_2020_cs = osp.join(train_2020_cs_path, 'new_train_list.txt')
        filename_train_2020_fs = osp.join(train_2020_fs_path, 'train_list.txt')
        filename_train_2019_cs = osp.join(train_2019_cs_path, 'new_2019_cs_train_list.txt')
        filename_train_2019_fs = osp.join(train_2019_fs_path, 'train_list.txt')
        filename_train_2020_fs_unlabel = osp.join(train_2020_fs_unlabel_path, 'label.txt')
        dataset = []
        camid = 1
        count_image=defaultdict(list)
        #load 2020 cs dataset
        with open(filename_train_2020_cs, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                img_name,img_label = [i for i in lines.split(':')]
                # if img_name == 'train/105180993.png' or img_name=='train/829283568.png' or img_name=='train/943445997.png': # remove samples with wrong label
                #     continue
                img_label = 'train_2020_cs_' + str(img_label)
                img_name = osp.join('images',img_name)
                img_name = osp.join(train_2020_cs_path,img_name)
                count_image[img_label].append(img_name)
        ccil_2020 = len(count_image)
        print("wenli:2020cs len is",ccil_2020)
        # load 2020 fs dataset
        with open(filename_train_2020_fs, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                img_name, img_label = [i for i in lines.split(':')]
                # if img_name == 'train/105180993.png' or img_name=='train/829283568.png' or img_name=='train/943445997.png': # remove samples with wrong label
                #     continue
                img_label = 'train_2020_fs_' + str(img_label)
                img_name = osp.join('images', img_name)
                img_name = osp.join(train_2020_fs_path, img_name)
                count_image[img_label].append(img_name)
        fcil_2020 = len(count_image)-ccil_2020
        print("wenli:2020fs len is",fcil_2020)
        # load 2019 cs dataset
        with open(filename_train_2019_cs, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                img_name,img_label = [i for i in lines.split(':')]
                # if img_name == 'train/105180993.png' or img_name=='train/829283568.png' or img_name=='train/943445997.png': # remove samples with wrong label
                #     continue
                img_label = 'train_2019_cs_' + str(img_label)
                img_name = osp.join(train_2019_cs_path, img_name)
                count_image[img_label].append(img_name)
        ccil_2019 = len(count_image)-ccil_2020-fcil_2020
        print("wenli:2019cs len is",ccil_2019)



        # loade 2019 fs dataset
        with open(filename_train_2019_fs, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                img_name,img_label = [i for i in lines.split(' ')]
                if img_name == 'train/105180993.png' or img_name=='train/829283568.png' or img_name=='train/943445997.png': # remove samples with wrong label
                    continue
                img_label = 'train_2019_fs_' + str(img_label)
                img_name = osp.join(train_2019_fs_path, img_name)
                count_image[img_label].append(img_name)
        fcil_2019 = len(count_image)-ccil_2020-fcil_2020-ccil_2019
        print("wenli:2019fs len is",fcil_2019)

        if self.cfg.DATASETS.USE_UNLABEL:
            # load 2020 fs unlabel dataset
            with open(filename_train_2020_fs_unlabel, 'r') as file_to_read:
                while True:
                    lines = file_to_read.readline()
                    if not lines:
                        break
                    img_name, img_label = [i for i in lines.split(':')]
                    # if img_name == 'train/105180993.png' or img_name=='train/829283568.png' or img_name=='train/943445997.png': # remove samples with wrong label
                    #     continue
                    img_label = 'train_2020_fs_unlabel_' + str(img_label)
                    img_name = osp.join('images', img_name)
                    img_name = osp.join(train_2020_fs_unlabel_path, img_name)
                    count_image[img_label].append(img_name)
            unlabel_2020fs = len(count_image)-ccil_2020-fcil_2020-ccil_2019-fcil_2019
            print("wenli:unlabel len is", unlabel_2020fs)

        val_imgs = {}
        pid_container = set()
        for pid, img_name in count_image.items():
            if len(img_name) < 2:
                pass
            else:
                val_imgs[pid] = count_image[pid]
                pid_container.add(pid)
        pid_container = sorted(pid_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        for pid, img_name in val_imgs.items():
            pid = pid2label[pid]
            for img in img_name:
                dataset.append((img, pid, camid))

        return dataset

    def _process_dir_test(self, data_dir, query=True):
        if query:
            subfix = 'query'
        else:
            subfix = 'gallery'

        datatype = ['green', 'normal']
        for index, type in enumerate(datatype):
            filename = osp.join(data_dir, '{}_{}.txt'.format(subfix, type))
            dataset = []
            with open(filename, 'r') as file_to_read:
                while True:
                    lines = file_to_read.readline()
                    if not lines:
                        break
                    for i in lines.split():
                        img_name = i

                    dataset.append((osp.join(self.dataset_dir_test, subfix, img_name), 1, 1))
            if index == 0:
                dataset_green = dataset
        return dataset_green, dataset

