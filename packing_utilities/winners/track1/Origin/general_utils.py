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

import copy
import config


class Area:
    def __init__(self,
                 lx=0,
                 ly=0,
                 min_coord=None,
                 max_coord=None,
                 max_layer=0,
                 max_weight=0):
        """
        docstring here
            :param lx=0: 区域x方向长
            :param ly=0: 区域y方向长
            :param min_coord=None: 最小坐标点
            :param max_coord=None: 最大坐标点
            :param max_layer=0: 最大承载层数（底面积相同）
            :param max_weight=0: 最大承载重量（底面积不同）
        """
        self.lx = lx
        self.ly = ly
        self.min_coord = min_coord
        self.max_coord = [min_coord[0]+lx, min_coord[1]+ly]  # 自己改的
        self.max_layer = max_layer
        self.max_weight = max_weight

    @classmethod
    def by_length(cls, lx, ly, min_coord, max_layer=None, max_weight=None):
        """
        docstring here
            :param lx: 区域x方向长
            :param ly: 区域y方向长
            :param min_coord: 最小坐标点
            :param max_layer: 最大承载层数（底面积相同）
            :param max_weight: 最大承载重量（底面积不同）
        """
        max_coord = [min_coord[0] + lx, min_coord[1] + ly]
        return cls(lx, ly, min_coord, max_coord, max_layer, max_weight)

    @classmethod
    def by_coordinate(cls,
                      min_coord,
                      max_coord,
                      max_layer=None,
                      max_weight=None):
        """
        docstring here
            :param min_coord: 最小坐标点
            :param max_coord: 最大坐标点
            :param max_layer: 最大承载层数（底面积相同）
            :param max_weight: 最大承载重量（底面积不同）
        """
        size = []
        for i in range(len(min_coord)):
            size.append(max_coord[i] - min_coord[i])
        return cls(*size, min_coord, max_coord, max_layer, max_weight)


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


def find_coord(packed_box_list, projecting_direction, refer_coord):
    """
        自己加的
        为新EP space的最小坐标点找到合适的坐标
        :param packed_box_list: 除最新装载的box以外的所有已装载到该bin中的boxes
        :param projecting_direction: 0: 按平行x轴方向投影; 1: y轴; 2: z轴
        :param reference_coord: 一个x/y/z坐标
    """
    suitable_coord = None
    if projecting_direction == 0:
        reference_box_list = [box for box in packed_box_list \
                              if (box.x + box.lx) <= refer_coord[0] and \
                                 (box.y + box.ly) > refer_coord[1] and \
                                 box.y <= refer_coord[1] and \
                                 box.z <= refer_coord[2] and \
                                 (box.z + box.lz) > refer_coord[2]]
        if reference_box_list:
            # 降序排列
            reference_box_list.sort(key=lambda box: (box.x+box.lx), reverse=True)
            suitable_coord = reference_box_list[0].x + reference_box_list[0].lx
        else:
            suitable_coord = 0
    elif projecting_direction == 1:
        reference_box_list = [box for box in packed_box_list \
                              if (box.y + box.ly) <= refer_coord[1] and \
                                 (box.x + box.lx) > refer_coord[0] and \
                                 box.x <= refer_coord[0] and \
                                 box.z <= refer_coord[2] and \
                                 (box.z + box.lz) > refer_coord[2]]
        if reference_box_list:
            # 降序排列
            reference_box_list.sort(key=lambda box: (box.y+box.ly), reverse=True)
            suitable_coord = reference_box_list[0].y + reference_box_list[0].ly
        else:
            suitable_coord = 0
    elif projecting_direction == 2:
        reference_box_list = [box for box in packed_box_list \
                              if (box.z + box.lz) <= refer_coord[2] and \
                                 (box.y + box.ly) > refer_coord[1] and \
                                 box.y <= refer_coord[1] and \
                                 box.x <= refer_coord[0] and \
                                 (box.x + box.lx) > refer_coord[0]]
        if reference_box_list:
            # 降序排列
            reference_box_list.sort(key=lambda box: (box.z+box.lz), reverse=True)
            suitable_coord = reference_box_list[0].z + reference_box_list[0].lz
        else:
            suitable_coord = 0
    return suitable_coord


def get_hold_surface(space, packed_box_list):
    """
        遍历已装载的box, 为space寻找所有的支撑平面
        :param space: Space类
        :param packed_box_list: 已装载的boxes, 每个元素都是PackedBox类
    """
    hold_surface_list = []
    space_min_coord = space.min_coord
    space_max_coord = space.max_coord
    if space_max_coord[0] != (space_min_coord[0] + space.lx) or \
       space_max_coord[1] != (space_min_coord[1] + space.ly) or \
       space_max_coord[2] != (space_min_coord[2] + space.lz):
        raise Exception("The max coord needs to be updated.")
    # 若该space的最小坐标点的z坐标=0
    if space_min_coord[2] == 0:
        # 空list表示支撑平面为地
        hold_surface_list = []
        return hold_surface_list
    # 找出能为该space提供支撑平面的boxes(满足：box的最大z坐标=space的最小z坐标，且xy面有重合)
    selc_box_list = [box for box in packed_box_list \
                     if (box.z+box.lz) == space_min_coord[2] and \
                        is_overlap(min_coord1=[box.x, box.y], max_coord1=[box.x+box.lx, box.y+box.ly], min_coord2=space_min_coord, max_coord2=space_max_coord)]
    # 若没有box可以提供支撑，说明该空间是悬浮的，丢弃该空间
    if not selc_box_list:
        hold_surface_list = "Float"
    # 对每个box，确定其提供的hold_area
    # 一个box和space各有一个最小/最大坐标点，共4个x坐标和4个y坐标,
    # 将坐标升序排列，中间的两个坐标所确定的区域即为该box能为space提供的支撑区域
    for box in selc_box_list:
        x_coord = [space_min_coord[0], space_max_coord[0], box.x, box.x+box.lx]
        y_coord = [space_min_coord[1], space_max_coord[1], box.y, box.y+box.ly]
        x_coord.sort()  # 升序排列
        y_coord.sort()
        # 因为box是固体不可能在同一高度上重叠，故支撑平面也应该是互不重合的
        hold_surface_list.append(Area(lx=(x_coord[2] - x_coord[1]),\
                                      ly=(y_coord[2] - y_coord[1]), \
                                      min_coord=[x_coord[1], y_coord[1]], \
                                      max_coord=[x_coord[2], y_coord[2]]))
    if hold_surface_list is None:  # 每个空间都应该有支撑平面，若找不到，则说明存在错误
        raise Exception(" A space should have hold areas.")
    # 去重
    # if hold_surface_list != "Float":
    #     hold_surface_list = del_repeat_elem(hold_surface_list)
    return hold_surface_list


def update_epspace_by_box(packed_box_list, epspace_list):
    """
        给mypack.py用的
        根据box的位置和尺寸更新ep space的尺寸、hold_surface等数据
        :param packed_box: PackedBox类
        :param space: 某个extreme point所能利用的空间, Space类
    """
    revised_space_list = []
    wasted_space_list = []
    space_list = copy.deepcopy(epspace_list)
    for space in space_list:
        # if space.gen_by_which_box_order == 23:
        #     print("here")
        if space.min_coord[0]+space.lx != space.max_coord[0] or \
           space.min_coord[1]+space.ly != space.max_coord[1] or \
           space.min_coord[2]+space.lz != space.max_coord[2]:
            raise Exception("Space max coord needs to be revised.")
        # 选出可以为该空间提供支撑平面的box更新其hold_surface
        support_box_list = [box for box in packed_box_list \
                            if (box.z+box.lz) == space.min_coord[2] and \
                                is_overlap(space.min_coord, space.max_coord, \
                                           [box.x, box.y], [box.x+box.lx, box.y+box.ly])]
        if len(support_box_list) == 1:
            hold_surface_list = get_hold_surface(space, support_box_list)
            if hold_surface_list != "Float":
                space.hold_surface.extend(hold_surface_list)
        # 选出与该空间重合的box更新空间大小
        selc_box_list = [box for box in packed_box_list \
                        if is_overlap(space.min_coord, space.max_coord, \
                                      [box.x, box.y, box.z],
                                      [box.x+box.lx, box.y+box.ly, box.z+box.lz])]
        for box in selc_box_list:
            # 获得box和space的尺寸和最小最大坐标
            box_min_coord = [box.x, box.y, box.z]
            box_lx = box.lx
            box_ly = box.ly
            box_lz = box.lz
            box_max_coord = [box_min_coord[0]+box_lx, \
                             box_min_coord[1]+box_ly, \
                             box_min_coord[2]+box_lz]
            space_min_coord = space.min_coord
            space_lx = space.lx
            space_ly = space.ly
            space_lz = space.lz
            space_max_coord = [space_min_coord[0]+space_lx, \
                               space_min_coord[1]+space_ly, \
                               space_min_coord[2]+space_lz]
            if space_max_coord[0] != space.max_coord[0] or \
               space_max_coord[1] != space.max_coord[1] or \
               space_max_coord[2] != space.max_coord[2]:
                raise Exception("Space max coord needs to be revised.")
            # 依据重叠情况修正space的尺寸和hold_surface
            # x方向
            if is_overlap(space.min_coord, space.max_coord, \
                          [box.x, box.y, box.z],
                          [box.x+box.lx, box.y+box.ly, box.z+box.lz]):
                if box_min_coord[1] <= space_min_coord[1] and box_min_coord[2] <= space_min_coord[2]:
                    # choose1和choose2可选一个，choose1理论上可以更好地利用空间，choose2会减少空间数量从而加快运行速度节省运行时间
                    #choose1: 取消了所谓的浪费空间
                    # 最小x,z坐标不变，最小y坐标变为该box的最最大y坐标
                    space.min_coord[1] = box.y + box.ly
                    # 更新y方向上space的宽度
                    space.ly = space.max_coord[1] - space.min_coord[1]
                    space.hold_surface = revise_hold_surface(space)
                    #choose1 end
                    #choose2: 使用“浪费空间”的概念
                    # if box_min_coord[0] <= space_min_coord[0]:
                    #     # 按box的最小坐标点与space的最小坐标点的关系，该EP空间应被舍弃
                    #     space.valid_space = "Float"
                    #     break
                    #     # raise Exception("The min_x_coord of box should be greater than space's.")
                    # space.valid_space = False  # 该space无效，属于被浪费的空间
                    # if space.lx > box_min_coord[0] - space_min_coord[0]:
                    #     # 修正该空间在x方向上的长度(以备后续利用该空间)
                    #     space.lx = box_min_coord[0] - space_min_coord[0]
                    #     # 更新space.max_coord
                    #     space.max_coord = [space.min_coord[0] + space.lx,
                    #                     space.min_coord[1] + space.ly,
                    #                     space.min_coord[2] + space.lz]
                    #     # 修正该空间的hold_surface
                    #     space.hold_surface = revise_hold_surface(space)
                    # choose2 end
                # y方向
                if box_min_coord[1] > space_min_coord[1]:
                    if box_min_coord[2] <= space_min_coord[2]:  # 修正y方向的长度
                        if space.ly > (box_min_coord[1] - space_min_coord[1]):
                            space.ly = box_min_coord[1] - space_min_coord[1]
                            # 更新space.max_coord
                            space.max_coord = [space.min_coord[0] + space.lx,
                                            space.min_coord[1] + space.ly,
                                            space.min_coord[2] + space.lz]
                            # 修正该空间的hold_surface
                            space.hold_surface = revise_hold_surface(space)
                    if box_min_coord[2] > space_min_coord[2]:  # 可以修正z或y方向的长度
                        space1_ly = box_min_coord[1] - space_min_coord[1]
                        space2_lz = box_min_coord[2] - space_min_coord[2]
                        if space.ly > space1_ly and space.lz > space2_lz:
                            # 若修改y方向的长度有更大体积，则修改y方向
                            if space.lx*space1_ly*space.lz >= space.lx*space.ly*space2_lz:
                                space.ly = space1_ly
                                # 更新space.max_coord
                                space.max_coord[1] = space.min_coord[1] + space.ly
                                # 更新space.hold_surface
                                space.hold_surface = revise_hold_surface(space=space)
                            # 否则，修改z方向的长度
                            else:
                                space.lz = space2_lz
                                # 更新space.max_coord. z方向上的改变不影响hold_surface,故无需修正支撑平面
                                space.max_coord[2] = space.min_coord[2] + space.lz
                        # elif space.ly > space1_ly:
                        #     space.ly = space1_ly
                        #     # 更新space.max_coord
                        #     space.max_coord[1] = space.min_coord[1] + space.ly
                        #     # 更新space.hold_surface
                        #     space.hold_surface = revise_hold_surface(space=space)
                        # elif space.lz > space2_lz:
                        #     space.lz = space2_lz
                        #     # 更新space.max_coord. z方向上的改变不影响hold_surface,故无需修正支撑平面
                        #     space.max_coord[2] = space.min_coord[2] + space.lz
                # z方向
                if ((box_min_coord[1] <= space_min_coord[1]) or (box_min_coord[1] <= space_min_coord[1])) and box_min_coord[2] > space_min_coord[2]:
                    if space.lz > box_min_coord[2] - space_min_coord[2]:
                        space.lz = box_min_coord[2] - space_min_coord[2]
                        # 更新space.max_coord.z方向上的改变不影响hold_surface,故无需修正支撑平面
                        space.max_coord[2] = space.min_coord[2] + space.lz
        # 若支撑平面列表为空且最小纵坐标不为0，则该空间为悬浮空间，舍弃
        if not space.hold_surface and space.min_coord[2] != 0:
            space.valid_space = "Float"
        if space.valid_space != "Float":
            if space.valid_space:
                revised_space_list.append(space)
            else:
                wasted_space_list.append(space)
    # if not revised_space_list:
    #     raise Exception("revised_space_list should not be empty.")
    return revised_space_list, wasted_space_list


def update_epspace_by_box_2(packed_box_list, epspace_list):
    """
        给改进后的华为装箱算法用的(packing_algorithm3.py, packing_algorithm4.py)
        根据box的位置和尺寸更新ep space的尺寸、hold_surface等数据
        :param packed_box: PackedBox类
        :param space: 某个extreme point所能利用的空间, Space类
    """
    revised_space_list = []
    # wasted_space_list = []
    space_list = copy.deepcopy(epspace_list)
    for space in space_list:
        # if space.min_coord[2] != 0:
        #     revised_space_list.append(space)
        #     continue
        # if space.min_coord[0]+space.lx != space.max_coord[0] or \
        #    space.min_coord[1]+space.ly != space.max_coord[1] or \
        #    space.min_coord[2]+space.lz != space.max_coord[2]:
        #     raise Exception("Space max coord needs to be revised.")
        # # 选出可以为该空间提供支撑平面的box更新其hold_surface
        # support_box_list = [box for box in packed_box_list \
        #                     if (box.z+box.lz) == space.min_coord[2] and \
        #                         is_overlap(space.min_coord, space.max_coord, \
        #                                    [box.x, box.y], [box.x+box.lx, box.y+box.ly])]
        # if len(support_box_list) == 1:
        #     hold_surface_list = get_hold_surface(space, support_box_list)
        #     if hold_surface_list != "Float":
        #         space.hold_surface.extend(hold_surface_list)
        # 选出与该空间重合的box更新空间大小
        selc_box_list = [box for box in packed_box_list \
                        if is_overlap(space.min_coord, space.max_coord, \
                                      [box.x, box.y, box.z],
                                      [box.x+box.lx, box.y+box.ly, box.z+box.lz])]
        for box in selc_box_list:
            # 获得box和space的尺寸和最小最大坐标
            box_min_coord = [box.x, box.y, box.z]
            # box_lx = box.lx
            # box_ly = box.ly
            # box_lz = box.lz
            # box_max_coord = [box_min_coord[0]+box_lx, \
            #                  box_min_coord[1]+box_ly, \
            #                  box_min_coord[2]+box_lz]
            space_min_coord = space.min_coord
            # space_lx = space.lx
            # space_ly = space.ly
            # space_lz = space.lz
            # space_max_coord = [space_min_coord[0]+space_lx, \
            #                    space_min_coord[1]+space_ly, \
            #                    space_min_coord[2]+space_lz]
            # if space_max_coord[0] != space.max_coord[0] or \
            #    space_max_coord[1] != space.max_coord[1] or \
            #    space_max_coord[2] != space.max_coord[2]:
            #     raise Exception("Space max coord needs to be revised.")
            # 依据重叠情况修正space的尺寸和hold_surface
            if is_overlap(space.min_coord, space.max_coord, \
                          [box.x, box.y, box.z],
                          [box.x+box.lx, box.y+box.ly, box.z+box.lz]):
                # x方向
                if box_min_coord[1] <= space_min_coord[1] and box_min_coord[2] <= space_min_coord[2]:
                    # 最小x,z坐标不变，最小y坐标变为该box的最最大y坐标
                    space.min_coord[1] = box.y + box.ly
                    # 更新y方向上space的宽度
                    space.ly = space.max_coord[1] - space.min_coord[1]
                # y方向
                if box_min_coord[1] > space_min_coord[1]:
                    # if box_min_coord[2] <= space_min_coord[2]:  # 修正y方向的长度
                    if space.ly > (box_min_coord[1] - space_min_coord[1]):
                        space.ly = box_min_coord[1] - space_min_coord[1]
                        # 更新space.max_coord
                        space.max_coord[1] = space.min_coord[1] + space.ly
                        # space.max_coord = [space.min_coord[0] + space.lx,
                        #                 space.min_coord[1] + space.ly,
                        #                 space.min_coord[2] + space.lz]
                        # # 修正该空间的hold_surface
                        # space.hold_surface = revise_hold_surface(space)
                    # if box_min_coord[2] > space_min_coord[2]:  # 可以修正z或y方向的长度
                    #     space1_ly = box_min_coord[1] - space_min_coord[1]
                    #     space2_lz = box_min_coord[2] - space_min_coord[2]
                    #     if space.ly > space1_ly and space.lz > space2_lz:
                    #         # 若修改y方向的长度有更大体积，则修改y方向
                    #         if space.lx*space1_ly*space.lz >= space.lx*space.ly*space2_lz:
                    #             space.ly = space1_ly
                    #             # 更新space.max_coord
                    #             space.max_coord[1] = space.min_coord[1] + space.ly
                    #             # 更新space.hold_surface
                    #             # space.hold_surface = revise_hold_surface(space=space)
                    #         # 否则，修改z方向的长度
                    #         else:
                    #             space.lz = space2_lz
                    #             # 更新space.max_coord. z方向上的改变不影响hold_surface,故无需修正支撑平面
                    #             space.max_coord[2] = space.min_coord[2] + space.lz
                # z方向
                # if ((box_min_coord[1] <= space_min_coord[1]) or (box_min_coord[1] <= space_min_coord[1])) and box_min_coord[2] > space_min_coord[2]:
                #     if space.lz > box_min_coord[2] - space_min_coord[2]:
                #         space.lz = box_min_coord[2] - space_min_coord[2]
                #         # 更新space.max_coord. z方向上的改变不影响hold_surface,故无需修正支撑平面
                #         space.max_coord[2] = space.min_coord[2] + space.lz
        # 若支撑平面列表为空且最小纵坐标不为0，则该空间为悬浮空间，舍弃
        # if not space.hold_surface and space.min_coord[2] != 0:
        #     space.valid_space = "Float"
        # if space.valid_space != "Float":
        #     if space.valid_space:
        #         revised_space_list.append(space)
        #     else:
        #         wasted_space_list.append(space)
        if space.valid_space:
            revised_space_list.append(space)
    # if not revised_space_list:
    #     raise Exception("revised_space_list should not be empty.")
    return revised_space_list


def revise_hold_surface(space):
    """
        根据space更新后的长宽修正其的hold_surface
        :param space: Space类
    """
    # 修正该空间的hold_surface
    revised_hold_area_list = []
    hold_area_list = space.hold_surface
    if hold_area_list == "TBC":
        raise Exception("A space should have one or more hold areas.")
    if not hold_area_list:
        # 支撑面是地的，仍然为地
        return revised_hold_area_list
    else:
        for area in hold_area_list:
            if type(area) is not list:
                if area.max_coord[0] != area.min_coord[0]+area.lx or\
                area.max_coord[1] != area.min_coord[0]+area.ly:
                    area.max_coord[0] = area.min_coord[0]+area.lx
                    area.max_coord[1] = area.min_coord[0]+area.ly
                    # raise Exception("The area max coord needs to be revised.")
                # 如果有重合，则更新; 否则放弃该area
                if is_overlap(area.min_coord, area.max_coord,\
                            space.min_coord, space.max_coord):
                    if area.min_coord[0]+area.lx > space.max_coord[0]:
                        area.lx = space.max_coord[0] - area.min_coord[0]
                    if area.min_coord[1]+area.ly > space.max_coord[1]:
                        area.ly = space.max_coord[1] - area.min_coord[1]
                    revised_hold_area_list.append(area)
    return revised_hold_area_list


def del_repeat_space(space_list):
    space_list_no_repeat = []
    for space in space_list:
        if space not in space_list_no_repeat:
            space_list_no_repeat.append(space)
    return space_list_no_repeat


def del_repeat_hold_area(hold_surface):
    """
        为space去除重复的支撑平面
        :param hold_surface: 是一个list, 每个元素都是Area类型
    """
    area_list_no_repeat = []
    for area1 in hold_surface:
        flag = True
        for area2 in area_list_no_repeat:
            if area1.min_coord == area2.min_coord:
                flag = False
                break
        if flag:
            area_list_no_repeat.append(area1)
    return area_list_no_repeat


def trans_sol_2_route(sol, platform_list):
    """
        将sol转化为route
        :param sol: platform的索引列表
        :param platform_list: ListDict类型
        :retrun route: 每个元素都是一个platformCode
    """
    route = []
    for index in sol:
        platformCode = platform_list[int(index) - 1]
        route.append(platformCode)
    return route


def classify_box_by_platform(box_list):
    box_list = copy.deepcopy(box_list)
    box_by_platform_2DList = []
    for box in box_list:
        if not box_by_platform_2DList:
            box_by_platform_2DList.append([box])
        else:
            for i in range(len(box_by_platform_2DList)):
                if box.platform == box_by_platform_2DList[i][0].platform:
                    box_by_platform_2DList[i].append(box)
                    if i == len(box_by_platform_2DList) - 1:
                        # 如果box被分配给box_by_platform_2DList的最后一个list中，把i再加一以作标记
                        i += 1
                    break
            if i == len(box_by_platform_2DList) - 1:
                # 说明该box属于一个新platform
                box_by_platform_2DList.append([box])
    return box_by_platform_2DList


def add_box_2_box_by_platform(box, box_by_platform):
    """
        将新的box按照platformCode分配到box_by_platform中
        :param box_by_platform: 2D list, 每一维都是同一个platform的box list
    """
    if not box_by_platform:
        box_by_platform.append([box])
    else:
        for i in range(len(box_by_platform)):
            if box.platform == box_by_platform[i][0].platform:
                box_by_platform[i].append(box)
                if i == len(box_by_platform) - 1:
                    # 如果box被分配给box_by_platform_2DList的最后一个list中，把i再加一以作标记
                    i += 1
                break
        if i == len(box_by_platform) - 1:
            # 说明该box属于一个新platform
            box_by_platform.append([box])
    return box_by_platform


def get_total_volume_weight(box_list):
    volume = 0
    weight = 0
    for box in box_list:
        volume += box.volume
        weight += box.weight
    return volume, weight


def get_space_volume_sum(space_list):
    """
        计算所有space的体积和
        :param space_list: 每个元素都是Space类型
    """
    v_sum = 0
    for space in space_list:
        v_sum += space.lx * space.ly * space.lz
    return v_sum
