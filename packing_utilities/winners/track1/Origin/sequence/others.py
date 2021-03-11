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

from entity import SimpleBlock
import general_utils as utils
import constrains
import config


class BlockHeuristic:
    def __init__(self, bin_obj, box_list):
        assert box_list, 'There is no box need to be packed'
        self.box_list = box_list
        self.bin_obj = bin_obj
        self.box_size_map = self.gen_box_size_map()

    def gen_rectangle_block(self):
        block_table = []
        for size, elements in self.box_size_map.items():
            num = 0
            for ind in elements:
                num += self.box_list[ind].box_num
            lx, ly, lz = size
            nz = nx = ny = 1
            block_table.append(SimpleBlock(nx, ny, nz, size))
        return block_table

    def gen_box_size_map(self):
        box_size_map = {}
        for ind, item in enumerate(self.box_list):
            sizes = set()
            for direction in item.all_directions:
                sizes.add(
                    utils.choose_box_direction_len(item.length, item.width,
                                                   item.height, direction))
            for size in sizes:
                box_size_map.setdefault(size, []).append(ind)
        for value in box_size_map.values():
            value.sort(key=lambda x: self.box_list[x].weight, reverse=True)
        return box_size_map

    def gen_avail_block(self, block_table, bin_obj, space, avail):
        for block in block_table:
            if block.lx + space.min_coord[0] > bin_obj.length - config.distance_to_door:
                if block.lz + space.min_coord[2] > bin_obj.door_height:
                    continue
            size = block.item_size
            box_num = block.box_num
            # 有足够数量箱子构成块
            # 块能够放在当前空间
            # 如果有支撑约束，判断是否能够获取满足支撑约束的块
            if (avail[size] >= box_num and constrains.can_in_space(
                [block.lx, block.ly, block.lz], [space.lx, space.ly, space.lz])
                    and
                (not config.constrain_hold or constrains.can_get_hold_block(
                    block, self.box_size_map, self.box_list, self.bin_obj,
                    space))):
                return block
        return None
