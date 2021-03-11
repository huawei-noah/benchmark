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

import general_utils as utils
from entity import PackedBox, AlgorithmBox, SimpleBlock


def assign_box_2_bin(box, space, packed_bin, box_direction):
    """
        将箱子放置到容器中
        :param box: the box that has been packed
        :param space: the space packs the box
        :param packed_bin: the bin packs the box
        :param box_direction: 箱子的摆放方向
    """
    packed_bin.order += 1
    lx, ly, lz = utils.choose_box_direction_len(box.length, box.width,
                                                box.height, box_direction)
    packed_box = PackedBox(*space.min_coord, lx, ly, lz, box, packed_bin.order,
                           box_direction, 0)
    copy_box = AlgorithmBox.copy_algorithm_box(box, 1)
    if packed_bin.packed_box_list:
        packed_bin.packed_box_list.append(packed_box)
        packed_bin.box_list.append(copy_box)
    else:
        packed_bin.packed_box_list = [packed_box]
        packed_bin.box_list = [copy_box]
    packed_bin.load_volume += lx * ly * lz
    packed_bin.load_weight += box.weight
    packed_bin.load_amount += box.amount


def assign_rectangle_box_in_block(space, block, packed_bin):
    """
        将块放入空间中
        :param space: 放入的空间
        :param block: 需要放入的块
        :param packed_bin: 放入的容器
    """
    lx, ly, lz = block.item_size
    base_x, base_y, base_z = space.min_coord
    order = packed_bin.order
    paceked_box_list = block.packed_box_list
    i = 0
    if isinstance(block, SimpleBlock):
        for num_z in range(block.nz):
            for num_x in range(block.nx):
                for num_y in range(block.ny):
                    box = paceked_box_list[i]
                    i += 1
                    order += 1
                    direction = utils.get_box_direction(box.length, box.width,
                                                        box.height, lx, ly, lz)
                    copy_box = AlgorithmBox.copy_algorithm_box(box, 1)
                    packed_box = PackedBox(
                        base_x + num_x * lx, base_y + num_y * ly,
                        base_z + num_z * lz, lx, ly, lz, box, order, direction, 0)
                    if packed_bin.packed_box_list:
                        packed_bin.packed_box_list.append(packed_box)
                        packed_bin.box_list.append(copy_box)
                    else:
                        packed_bin.packed_box_list = [packed_box]
                        packed_bin.box_list = [copy_box]
    else:

        #TODO next
        y_coor = [base_y]
        for pattern in block.pt:
            if pattern == 0:
                y_coor.append(y_coor[-1]+ly)
            else:
                y_coor.append(y_coor[-1]+lx)

        for num_z in range(1,block.nz+1):
            for num_y in range(len(block.pt)):
                for num_x in range(block.nm[num_y]):
                    box = paceked_box_list[i]
                    if block.pt[num_y] == 0:
                        box_x = base_x + num_x * lx
                    else:
                        box_x = base_x + num_x * ly
                    box_y = y_coor[num_y]
                    box_z = base_z + (num_z-1)*lz
                    i += 1
                    if order == 93:
                        aaa = 1.0
                    order += 1
                    direction = utils.get_box_direction(box.length, box.width,
                                                        box.height, lx, ly, lz)
                    if block.pt[num_y] == 1: 
                        direction = 1 - direction                            
                    copy_box = AlgorithmBox.copy_algorithm_box(box, 1)

                    if block.pt[num_y] == 0:
                        packed_box = PackedBox(
                            box_x, box_y,
                            box_z, lx, ly, lz, box, order, direction, 0)
                    else:
                        packed_box = PackedBox(
                            box_x, box_y,
                            box_z, ly, lx, lz, box, order, direction, 0)

                    if packed_bin.packed_box_list:
                        packed_bin.packed_box_list.append(packed_box)
                        packed_bin.box_list.append(copy_box)
                    else:
                        packed_bin.packed_box_list = [packed_box]
                        packed_bin.box_list = [copy_box]

    packed_bin.order = order
    packed_bin.load_volume += block.vol
    packed_bin.load_weight += block.weight
    packed_bin.load_amount += block.amount
