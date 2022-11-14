import os
import shutil
import numpy as np
import xml.etree.ElementTree as ET

def convert(size, box):
    dw = 1./(size[0])  
    dh = 1./(size[1])  
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
 
def convert_annotation(in_file_p, out_file_p):
    in_file = open(in_file_p, 'r')
    out_file = open(out_file_p, 'w')

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
 
    for obj in root.iter('object'):
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(0) + " " + " ".join([str(a) for a in bb]) + '\n')

    in_file.close()
    out_file.close()

def pre_dataset():
    """
    按照voc固定格式准备数据集
    输出格式为
    |-set
      |-images
      | |-train
      | |-val
      |-labels
        |-train
        |-val
    """

    # 设定使用平台数据集版本609或者608
    data_path = '/home/data/609' 

    # 1. 读取平台数据
    img_xml_list = os.listdir(data_path)
    img_list = []
    xml_list = []

    for ix in img_xml_list:
        if ix.endswith('.xml'):
            xml_list.append(ix)
        else:
            img_list.append(ix)

    xml_list.sort()
    img_list.sort()


    # 2. 设置输出文件夹
    out_root = '/home/data/set/'
    img_train = out_root + 'images/train/'
    img_val = out_root + 'images/val/'

    txt_train = out_root + 'labels/train/'
    txt_val = out_root + 'labels/val/'

    new_dir = [img_train, img_val, txt_train, txt_val]
    for n_dir in new_dir:
        if not os.path.exists(n_dir):
            os.makedirs(n_dir)


    # 3. TODO: 请自行设定验证集比例----------------------------------------------------
    val_pro = 0.0 # 验证集所占比例,范围(0,1)
    val_num = int(val_pro * len(xml_list))
    val_index_list = np.random.randint(0,len(xml_list),val_num)
    # 3. TODO: 补充结束---------------------------------------------------------------


    # 4. 拷贝图片到验证集或者训练集
    index_l = 0
    for i in img_list:
        # print(os.path.join(img_train, i))
        if index_l in val_index_list:
            shutil.copy(os.path.join(data_path, i), os.path.join(img_val, i))
        else:
            shutil.copy(os.path.join(data_path, i), os.path.join(img_train, i))
        index_l += 1


    # 5. 对应将标注文件转换到验证集或者训练集
    index_l = 0
    for x in xml_list:
        xml_path = os.path.join(data_path, x)
        if index_l in val_index_list:
            val_txt_file = os.path.join(txt_val, x.split('.')[0] + '.txt')
            convert_annotation(xml_path, val_txt_file)
        else:
            train_txt_file = os.path.join(txt_train, x.split('.')[0] + '.txt')
            convert_annotation(xml_path, train_txt_file)
        index_l += 1
      

    print('-------------------------------Dataset already!--------------------------------------')