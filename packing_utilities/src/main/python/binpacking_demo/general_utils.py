"""
   Copyright (c) 2020. Huawei Technologies Co., Ltd.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from . import config


def choose_box_direction_len(length, width, height, direction):
    """
        根据箱子的长宽高和方向确定箱子的x,y,z方向的长度
        :param length: 箱子的长
        :param width: 箱子的宽
        :param height: 箱子的高
        :param direction: 箱子的方向
    """
    if direction == 0:
        lx = length
        ly = width
        lz = height
    elif direction == 1:
        lx = width
        ly = length
        lz = height
    elif direction == 2:
        lx = length
        ly = height
        lz = width
    elif direction == 3:
        lx = height
        ly = length
        lz = width
    elif direction == 4:
        lx = width
        ly = height
        lz = length
    elif direction == 5:
        lx = height
        ly = width
        lz = length
    else:
        raise Exception('direction is out of range')
    return lx, ly, lz


def get_box_direction(length, width, height, lx, ly, lz):
    """
        根据长宽高以及箱子实际x,y,z方向长度确认箱子的旋转
        :param length: 长
        :param width: 宽
        :param height: 高
        :param lx: x方向长
        :param ly: y方向长
        :param lz: z方向长
        :return:
    """
    if lx == length and ly == width and lz == height:
        return 0
    if lx == width and ly == length and lz == height:
        return 1
    if lx == length and ly == height and lz == width:
        return 2
    if lx == height and ly == length and lz == width:
        return 3
    if lx == width and ly == height and lz == length:
        return 4
    if lx == height and ly == width and lz == length:
        return 5


def is_avail_space(space):
    """
        判断一个空间是否是真实存在的（即长宽高必须都大于0）
        :param space: 待判断的空间
    """
    # 空间的x,y,z方向的长度存在小于0的，返回False
    if space.max_coord[0] - space.min_coord[0] <= 0 or space.max_coord[
            1] - space.min_coord[1] <= 0 or space.max_coord[
                2] - space.min_coord[2] <= 0:
        return False
    # 若有高度约束，空间的起始高度大于等于该约束，返回False
    if config.constrains_height and space.min_coord[
            2] >= config.constrains_height:
        return False
    return True


def is_overlap(min_coord1, max_coord1, min_coord2, max_coord2):
    """
        判断两个空间是否重合，若重合返回True，反之返回False
        :param min_coord1: 第一个区域的最小坐标
        :param max_coord1: 第一个区域的最大坐标
        :param min_coord2: 第二个区域的最小坐标
        :param max_coord2: 第二个区域的最大坐标
    """
    length = min(len(min_coord1), len(min_coord2))
    for i in range(length):
        if min_coord1[i] >= max_coord2[i] or max_coord1[i] <= min_coord2[i]:
            return False
    return True


def is_combine(min_coord1, max_coord1, min_coord2, max_coord2):
    """
        判断第一个区域是否包含第二个区域
        :param min_coord1: 第一个区域的最小坐标
        :param max_coord1: 第一个区域的最大坐标
        :param min_coord2: 第二个区域的最小坐标
        :param max_coord2: 第二个区域的最大坐标
    """
    for i in range(len(min_coord1)):
        if min_coord1[i] > min_coord2[i] or max_coord1[i] < max_coord2[i]:
            return False
    return True


def is_same(min_coord1, max_coord1, min_coord2, max_coord2):
    """
    判断两个区域是否完全相等
        :param min_coord1: 第一个区域的最小坐标
        :param max_coord1: 第一个区域的最大坐标
        :param min_coord2: 第二个区域的最小坐标
        :param max_coord2: 第二个区域的最大坐标
    """
    assert len(min_coord1) == len(
        min_coord2), "Inconsistent dimensions of the two areas"
    for i in range(len(min_coord1)):
        if min_coord1[i] != min_coord2[i] or max_coord1[i] != max_coord2[i]:
            return False
    return True
