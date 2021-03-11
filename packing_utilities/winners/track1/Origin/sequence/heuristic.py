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


def sort_by_same_basal_area_vol(box_list):
    """
    按照长和宽相同的箱子的总体积和进行排序，相同长和宽的箱子按照重量及高度排序。
    :param box_list: 待排序箱子列表
    """
    box_size_map = {}
    box_size_map_vol = {}
    for box in box_list:
        box_size_map.setdefault((box.length, box.width), []).append(box)
    for key, boxes in box_size_map.items():
        vol = 0
        boxes.sort(key=lambda x: (x.weight, x.height), reverse=True)
        for box in boxes:
            vol += box.length * box.width * box.height * box.box_num
        box_size_map_vol[key] = vol
    key_list = sorted(
        list(box_size_map_vol.keys()),
        key=lambda x: box_size_map_vol[x],
        reverse=True)
    new_box_list = []
    for key in key_list:
        new_box_list.extend(box_size_map[key])
    return new_box_list
