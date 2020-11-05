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

class Simulator:
    @staticmethod
    def transform(packed_bin_list):
        all_res = []
        for packed_bin in packed_bin_list:
            res = {
                "bin_type": packed_bin.bin_type,
                "bin_size": [packed_bin.length,
                             packed_bin.width,
                             packed_bin.height]
            }
            boxes = []
            for packed_box in packed_bin.packed_box_list:
                x = packed_box.z - 0.5 * packed_box.lx + 0.5 * packed_bin.length
                y = packed_box.x - 0.5 * packed_box.ly + 0.5 * packed_bin.width
                z = packed_box.y - 0.5 * packed_box.lz + 0.5 * packed_bin.height
                boxes.append({
                    "box_type": packed_box.box_id,
                    "size": [packed_box.lx,
                                 packed_box.ly,
                                 packed_box.lz],
                    "min_coordinate": [x, y, z]
                })
            res['data'] = boxes
            all_res.append([res])
        return all_res
