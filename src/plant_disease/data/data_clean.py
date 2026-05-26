"""Deduplicate training images and drop ambiguously labelled samples."""

import logging
import os
from collections import Counter

from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_files(file_path):
    """
    载入文件
    :param file_path: 图像路径
    :return 返回文件名list，文件名及其路径的dict
    """
    filename_lists = list()
    filepath_dicts = dict()
    for directory, folder, filenames in os.walk(file_path):
        for filename in tqdm(filenames):
            if os.path.splitext(filename)[-1] not in [".jpg", ".JPG", ".JPEG", ".jpeg", ".png"]:
                continue

            filename_lists.append(filename)
            filepath_dicts[filename] = os.path.join(directory, filename)

    return filename_lists, filepath_dicts


def derepeat(name_list):
    """
    删除重复的图像名
    :param name_list: 图像名列表
    :return:
    """
    new_name_list = list()
    for n in name_list:
        if "副本" in n:
            logger.warning("found '副本' filename: %s", n)
            n = n.split("-")[0].strip(" ") + os.path.splitext(n)[-1]

        new_name_list.append(n)

    # 统计每张图片出现的次数,删除出现重复次数的图像名
    count = Counter(new_name_list)
    logger.info("source nums: %d", len(new_name_list))
    new_name_list = list(set(new_name_list))
    logger.info("dst nums: %d", len(new_name_list))
    for key, value in count.items():
        if value > 1:
            new_name_list.remove(key)

    return new_name_list


def remove_file(name_list, path_list):
    """
    删除重复出现的图像
    :param name_list: 图像名
    :param path_list: 图像及对应路径的dict
    :return: 删除后的图像及路径名字典
    """
    new_dict = dict()
    delete_name_list = list()
    for k, v in path_list.items():
        if k not in name_list:
            delete_name_list.append(v)  # 后面还需要对待副本字样的图像进行检查其是否存在相同的副本
        else:
            new_dict[k] = v

    # 对于有些带副本的图像不一定存在重复的图像
    tmp = set(name_list).difference(set(new_dict.keys()))
    logger.debug("residual names: %s", tmp)
    for value in tmp:
        for k in path_list.keys():
            if os.path.splitext(value)[0] in k:
                # 在操作前先判断，因为有些图像名可能是部分匹配
                if path_list[k] in delete_name_list:
                    new_dict[k] = path_list[k]
                    delete_name_list.remove(path_list[k])
                    name_list[name_list.index(value)] = k

    # 删除图像文件
    for t in delete_name_list:
        os.remove(t)

    return new_dict, name_list


def dereplicate(train_file_path):
    """
    单个数据集(如训练集，测试集)去重复
    :param train_file_path:
    :return:
    """
    train_name_list, train_path_dict = load_files(train_file_path)
    logger.info("train img count=%d, path count=%d", len(train_name_list), len(train_path_dict))
    train_name_list = derepeat(train_name_list)
    train_path_dict, train_name_list = remove_file(train_name_list, train_path_dict)
    logger.info("clean img count=%d, path count=%d", len(train_name_list), len(train_path_dict))

    return train_name_list, train_path_dict


def process_repeat(train_file_path, val_file_path):
    """
    查找训练集和测试集合中重复的图像名
    :param train_file_path: 训练集路径
    :param val_file_path: 验证集路径
    :return:
    """
    # 训练集，验证集分别去重
    train_name_list, train_path_dict = dereplicate(train_file_path)
    val_name_list, val_path_dict = dereplicate(val_file_path)

    # 测试集和验证集取交集
    same = set(train_name_list) & set(val_name_list)
    logger.info("train/val intersection size: %d", len(same))

    # 删除对应的图像路径
    for v in same:
        if v in train_path_dict.keys():
            logger.warning("removing duplicate path: %s", v)
            os.remove(train_path_dict[v])
            os.remove(val_path_dict[v])


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    args = parser.parse_args()
    process_repeat(args.train, args.val)
