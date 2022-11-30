#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File          :  tmp.py
@Time          :  2022/01/14 18:00:32
@Author        :  zc12345 
@Version       :  1.0
@Contact       :  zhangjingyuezjy@163.com
@Description   :  None
'''

# here put the import lib
import os
import shutil

def cp_files(src_dir, dst_dir):
    for root, dirs, files in os.walk(src_dir):
        for curr_dir in dirs:
            part_dir = os.path.join(root, curr_dir)
            for f in os.listdir(part_dir):
                if os.path.splitext(f)[-1] == '.log':
                    out_dir = os.path.join(dst_dir, curr_dir)
                    if not os.path.exists(out_dir):
                        os.mkdir(out_dir)
                    in_path = os.path.join(part_dir, f)
                    out_path = os.path.join(out_dir, f)
                    print('mv {} file'.format(f))
                    shutil.copy(in_path, out_path)

if __name__ == "__main__":
    src_path = '/data1/zc12345/my_code/models'
    dst_path = './models'
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    cp_files(src_dir=src_path, dst_dir=dst_path)