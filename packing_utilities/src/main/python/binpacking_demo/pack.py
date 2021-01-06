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

import json
import shutil
import statistics
import sys
import os
import time
import traceback

from .entity import AlgorithmBox, Bin
from .packing_algorithms import SSBH
from .simulator import Simulator


class Pack:
    def __init__(self,
                 message_str,
                 spu_ids=None,
                 truck_code=None,
                 route=None,
                 output_path=None,
                 simulator_path=None):
        """
        3D bin packing initialization, with given boxes, route and truck.
        三维装载初始化。可以指定装载箱子、路径、卡车。
        :param message_str: input params string; 入参字符串；
        :param spu_ids: designated boxes' ids. Default: all boxes;
            指定装载箱子的spu id集合，默认为所有箱子；
        :param truck_code: designated truck code. Default: the largest truck;
            指定卡车代码，默认为最大卡车；
        :param route: designated route, according to which the boxes are
            packed in order. Default: the default platform order given by input
            files; 指定路径，按照路径顺序进行箱子装载，默认为入参中提货点列表顺序；
        :param output_path: designated output directory. Default: do not save
        output params in files; 指定出参的输出文件夹, 默认不保存出参在文件中；
        """
        self.data = json.loads(message_str)
        self.truck = None
        self.bin = self._gen_bin(truck_code)
        self.box_list = self._gen_box_list(spu_ids)
        if route is None:
            self.routes = []
            route = []
            bonded_warehouses = []
            for platform in self.data[
                    'algorithmBaseParamDto']['platformDtoList']:
                if 'mustFirst' in platform and platform['mustFirst']:
                    if platform['platformCode'] not in bonded_warehouses:
                        bonded_warehouses.append(platform['platformCode'])
                elif platform['platformCode'] not in route:
                    route.append(platform['platformCode'])
            if bonded_warehouses:
                self.routes.append([bonded_warehouses[0]] + route)
                for i in range(1, len(bonded_warehouses)):
                    self.routes.append([bonded_warehouses[i]])
            else:
                self.routes.append(route)
        else:
            self.routes = route
        self.output_path = output_path
        self.file_name = self.data['estimateCode']
        self.simulator_path = simulator_path

    def run(self):
        if not self.box_list:
            raise Exception("Box list is empty.")
        packed_bin_list = self._pack_by_platform()
        res = self._gen_res(packed_bin_list)
        if self.output_path is not None:
            if not os.path.exists(self.output_path):
                os.mkdir(self.output_path)
            output_file = os.path.join(self.output_path, self.file_name)
            with open(
                    output_file, 'w', encoding='utf-8', errors='ignore') as f:
                json.dump(res, f, ensure_ascii=False)
        if self.simulator_path is not None:
            simulator_path = os.path.join(self.simulator_path, self.file_name)
            if not os.path.exists(self.simulator_path):
                os.mkdir(self.simulator_path)
            if os.path.exists(simulator_path):
                shutil.rmtree(simulator_path)
            os.mkdir(simulator_path)
            all_res_sim = Simulator.transform(packed_bin_list)
            num = 1
            for res_sim in all_res_sim:
                simulator_file = os.path.join(
                    simulator_path,
                     str(num) + '_' + res_sim[0]['bin_type'] + '.txt')
                with open(simulator_file, 'w') as f:
                    json.dump(res_sim, f)
                num += 1
        return res

    def _pack_by_platform(self):
        pack_bin_list = []
        for route in self.routes:
            last_bin = None
            for platform in route:
                box_in_platform = [
                    box for box in self.box_list if box.platform == platform]
                if not box_in_platform:
                    continue
                pack_bins = SSBH.pack(self.bin, box_in_platform, last_bin=last_bin)
                pack_bin_list.extend(pack_bins[:-1])
                last_bin = pack_bins[-1]
            if last_bin:
                pack_bin_list.append(last_bin)
            else:
                raise Exception("Packing Fails.")
        return pack_bin_list

    def _pack(self):
        return SSBH.pack(self.bin, self.box_list)

    def _compare(self, packed_bin_list_1, packed_bin_list_2):
        packed_box_ratio_1 = statistics.mean(
            [packed_bin.ratio for packed_bin in packed_bin_list_1])
        packed_box_ratio_2 = statistics.mean(
            [packed_bin.ratio for packed_bin in packed_bin_list_2])
        return packed_bin_list_1 \
            if packed_box_ratio_1 >= packed_box_ratio_2 \
            else packed_bin_list_2

    def _gen_box_list(self, spu_ids=None):
        raw_box_list = self.data["boxes"]
        box_map_list = []
        for raw_box in raw_box_list:
            if spu_ids is not None and raw_box['spuBoxId'] not in spu_ids:
                continue
            box = {}
            box['box_id'] = raw_box['spuBoxId']
            box['box_type'] = ""
            box['length'] = raw_box['length']
            box['width'] = raw_box['width']
            box['height'] = raw_box['height']
            box['weight'] = raw_box['weight']
            box['all_directions'] = [0, 1]
            box['box_num'] = 1
            if 'platformCode' in raw_box:
                box['platform'] = raw_box['platformCode']
            else:
                box['platform'] = 'same'
            box['max_layer'] = sys.maxsize
            box['max_weight'] = float('inf')
            box['is_cylinder'] = False
            box_map_list.append(box)
        box_list = []
        for box in box_map_list:
            box_list.append(AlgorithmBox(**box))
        return box_list

    def _gen_bin(self, truck_code=None):
        raw_trucks = self.data["algorithmBaseParamDto"]["truckTypeDtoList"]
        if truck_code:
            raw_trucks = [truck for truck in raw_trucks
                          if truck['truckTypeCode'] == truck_code]
            if not raw_trucks:
                raise Exception("Truck code not found.")
            self.truck = raw_trucks[0]
        else:
            truck_vols = [
                raw_truck["length"] * raw_truck["width"] * raw_truck["height"]
                for raw_truck in raw_trucks
            ]
            truck_ind = truck_vols.index(max(truck_vols))
            self.truck = raw_trucks[truck_ind]
        bin_map = {}
        bin_map["bin_type"] = self.truck["truckTypeName"]
        bin_map["length"] = self.truck["length"]
        bin_map["width"] = self.truck["width"]
        bin_map["height"] = self.truck["height"]
        bin_map["volume"] = (self.truck["length"]
                             * self.truck["width"]
                             * self.truck["height"])
        bin_map["max_weight"] = self.truck["maxLoad"]
        bin_map["truck_type_id"] = self.truck["truckTypeId"]
        bin_map["truck_type_code"] = self.truck["truckTypeCode"]
        return Bin(**bin_map)

    def _gen_res(self, packed_bin_list):
        res = {"estimateCode": self.data["estimateCode"], "solutionArray": []}
        truck_list = []
        if packed_bin_list:
            for packed_bin in packed_bin_list:
                truck = {}
                truck["truckTypeId"] = packed_bin.truck_type_id
                truck["truckTypeCode"] = packed_bin.truck_type_code
                truck["piece"] = len(packed_bin.packed_box_list)
                truck["volume"] = packed_bin.load_volume
                truck["weight"] = packed_bin.load_weight
                truck["innerLength"] = packed_bin.length
                truck["innerWidth"] = packed_bin.width
                truck["innerHeight"] = packed_bin.height
                truck["maxLoad"] = self.truck["maxLoad"]
                spu_list = []
                for packed_box in packed_bin.packed_box_list:
                    spu = {}
                    packed_box.change_ordinate(packed_bin)
                    spu["spuId"] = packed_box.box_id
                    spu["direction"] = packed_box.direction
                    spu["x"] = packed_box.x
                    spu["y"] = packed_box.y
                    spu["z"] = packed_box.z
                    spu["order"] = packed_box.order
                    spu["length"] = packed_box.length
                    spu["width"] = packed_box.width
                    spu["height"] = packed_box.height
                    spu["weight"] = packed_box.weight
                    spu['platformCode'] = packed_box.platform
                    spu_list.append(spu)
                spu_list.sort(key=lambda box: box['order'])
                truck["spuArray"] = spu_list
                platform_list = []
                for spu in spu_list:
                    if spu['platformCode'] not in platform_list:
                        platform_list.append(spu['platformCode'])
                truck["platformArray"] = platform_list
                truck_list.append(truck)
        res["solutionArray"].append(truck_list)
        return res
