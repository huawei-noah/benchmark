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

from entity import SimpleBlock, MixedBlock
import packing_algorithms as alg
import general_utils as utils
import constrains
import config


class BlockHeuristic:
    def __init__(self, bin_obj, box_list):
        assert box_list, 'There is no box need to be packed'
        self.box_list = box_list
        self.bin_obj = bin_obj
        self.box_size_map = self.gen_box_size_map()



    def gen_mixed_block(self, size, elements):
        mixed_block_table = []

        num = 0
        for ind in elements:
            num += self.box_list[ind].box_num
        lx, ly, lz = size
        box_num_patten_dict = {}
        for box_num in range(2, num+1):
            pattern_tuple = utils.generate_multip_patten(self.bin_obj.length, self.bin_obj.width, lx, ly, box_num)
            box_num_patten_dict[box_num] = pattern_tuple

        for nz in range(1, min(num, int(self.bin_obj.height // lz)) + 1):
            for num_xy in range(2, num//nz+1):
                if box_num_patten_dict[num_xy][0]:
                    patten_list, patten_num_list = box_num_patten_dict[num_xy]
                    for i in range(len(patten_list)):
                        mixed_block_table.append(MixedBlock(patten_list[i], patten_num_list[i], nz, size))

        return mixed_block_table

    def gen_rectangle_block(self):
        block_table = []
        for size, elements in self.box_size_map.items():
            num = 0
            for ind in elements:
                num += self.box_list[ind].box_num
            lx, ly, lz = size

            for nz in range(1, min(num, int(self.bin_obj.height // lz)) + 1):
                for nx in range(
                        1,
                        int(min(num // nz, self.bin_obj.length // lx)) + 1):
                    for ny in range(
                            1,
                            int(min(num / nz // nx, self.bin_obj.width // ly))
                            + 1):
                        block_table.append(SimpleBlock(nx, ny, nz, size))
            box_length = self.box_list[elements[0]].length
            box_width = self.box_list[elements[0]].width
            box_height = self.box_list[elements[0]].height

            if ly > 0.1 * self.bin_obj.width and lx != ly and utils.get_box_direction(box_length, box_width, box_height, lx, ly, lz) == 0:
                mixed_block_table = self.gen_mixed_block(size, elements)
                block_table.extend(mixed_block_table)

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

            # if (avail[size] >= box_num and constrains.can_in_space(
            #     [block.lx, block.ly, block.lz], [space.lx, space.ly, space.lz])
            #         and
            #     (not config.constrain_hold or constrains.can_get_hold_block(
            #         block, self.box_size_map, self.box_list, self.bin_obj,
            #         space))
            #         and
            #         (not alg.filter_space_by_block(bin_obj,bin_obj.space_obj.space_list, block, space))):
                return block
        return None
