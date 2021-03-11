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
import copy
import traceback
import general_utils as utils
from entity import AlgorithmBox, Bin
from packing_algorithms1 import SSBH as SSBH1
from packing_algorithms4 import SSBH as SSBH4
import mypack
from order import Order


class Pack:
    def __init__(self,
                 message_str,
                 spu_ids=None,
                 truck_code=None,
                 route=None,
                 output_path=None,
                 simulator_path=None,
                 mypack_obj=None):
        """
        3D bin packing initialization, with given boxes, route and truck.
        三维装载初始化。可以指定装载箱子、路径、卡车。
        :param message_str: input params string; 入参字符串；(Han:包含了一个订单的所有信息)
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
        self.origin_bin = self._gen_bin(truck_code)
        self.bin = self.origin_bin
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
        if mypack_obj:
            self.mypack_obj = mypack_obj
        else:
            order = Order(message_str)
            self.mypack_obj = mypack.Pack(order)

    def run(self):
        if not self.box_list:
            raise Exception("Box list is empty.")
        packed_bin_list = self._pack_by_platform()  # (Han:利用的信息是某一类型的车以及该类型的车的路径;返回的是该类型的车按路径上的顺序全部装载完路径上的货物的arrangement,如要几辆该类型的车，每一辆该类型的车访问路径上的哪些points,以及如何放置boxes等;packed_bin_list中的每个entry即是一辆车的相关信息)
        res, truck_list = self._gen_res(packed_bin_list)  # Han:res["estimateCode"]存储了一个订单的json文件名,res["solutionArray"]存储了该订单的解solution
        # if self.output_path is not None:
        #     if not os.path.exists(self.output_path):
        #         os.mkdir(self.output_path)
        #     output_file = os.path.join(self.output_path, self.file_name)
        #     with open(
        #             output_file, 'w', encoding='utf-8', errors='ignore') as f:
        #         json.dump(res, f, ensure_ascii=False, indent=4)  # Han:将解保存为json文件
        return res, truck_list

    def _pack_by_platform(self):
        pack_bin_list = []
        for route in self.routes:
            last_bin = None
            for platform in route:
                box_in_platform = [  # (Han: 在某个platform上的所要装载的boxes的集合)
                    box for box in self.box_list if box.platform == platform]
                if not box_in_platform:
                    continue  # 下一个platform
                # packing_algorithms1和packing_algorithms2选一个
                pack_bins = SSBH1.pack(self.bin, box_in_platform, last_bin=last_bin)  # （Han:返回的是装载车辆相关信息）
                # pack_bins = packing_algorithms4.SSBH.pack(self.bin, box_in_platform, last_bin=last_bin)  # （Han:返回的是装载车辆相关信息）
                pack_bin_list.extend(pack_bins[:-1])  # (Han:不能继续装载的车的信息存入pack_bin_list中)
                last_bin = pack_bins[-1]
            if last_bin:
                pack_bin_list.append(last_bin)
            else:
                raise Exception("Packing Fails.")
        # avg_ratio = 0
        # for packed_bin in pack_bin_list:
        #     avg_ratio += packed_bin.ratio
        #     print(packed_bin.ratio, flush=True)
        # print("average:", avg_ratio/len(pack_bin_list))
        """ replace机制 """
        # pack_bin_list = self.replace_big_bin_by_small_bin(pack_bin_list, mark_ratio=0.8)  # 用华为装箱算法
        pack_bin_list = self.replace_big_bin_by_small_bin_2(pack_bin_list, mark_ratio=0.8)  # 用华为装箱算法
        # pack_bin_list = self.mypack_obj.replace_big_bin_by_small_bin(pack_bin_list, mark_ratio=0.7)  # 用自己的装箱算法
        """ begin:用packing_algorithms4进行repack以及再replace """
        # avg_ratio = 0
        # for packed_bin in pack_bin_list:
        #     avg_ratio += packed_bin.ratio
        # #     print(packed_bin.ratio, flush=True)
        # # print("average:", avg_ratio/len(pack_bin_list))
        # old_avg_ratio = avg_ratio / len(pack_bin_list)
        # old_pack_bin_list = copy.deepcopy(pack_bin_list)

        # pack_bin_list, repack = self.repack(pack_bin_list)

        # # avg_ratio = 0
        # # for packed_bin in pack_bin_list:
        # #     avg_ratio += packed_bin.ratio
        # #     print(packed_bin.ratio, flush=True)
        # # print("average:", avg_ratio/len(pack_bin_list))

        # if repack:
        #     # pack_bin_list = self.replace_big_bin_by_small_bin(pack_bin_list, mark_ratio=0.8)  # 用华为装箱算法
        #     pack_bin_list = self.replace_big_bin_by_small_bin_2(pack_bin_list, mark_ratio=0.8)  # 用华为装箱算法
        #     # pack_bin_list = self.mypack_obj.replace_big_bin_by_small_bin(pack_bin_list, mark_ratio=0.7)  # 用自己的装箱算法
        #     # avg_ratio = 0
        #     # for packed_bin in pack_bin_list:
        #     #     avg_ratio += packed_bin.ratio
        #     #     print(packed_bin.ratio, flush=True)
        #     # print("average:", avg_ratio/len(pack_bin_list))
        
        # avg_ratio = 0
        # for packed_bin in pack_bin_list:
        #     avg_ratio += packed_bin.ratio
        # #     print(packed_bin.ratio, flush=True)
        # # print("average:", avg_ratio/len(pack_bin_list))
        # new_avg_ratio = avg_ratio / len(pack_bin_list)
        # if new_avg_ratio < old_avg_ratio:
        #     pack_bin_list = old_pack_bin_list
        """ end:用packing_algorithms4进行repack以及再replace"""
        avg_ratio = 0
        for packed_bin in pack_bin_list:
            avg_ratio += packed_bin.ratio
            print(packed_bin.ratio, flush=True)
        print("average:", avg_ratio/len(pack_bin_list))
        return pack_bin_list

    def repack(self, old_packed_bin_list):
        repack = False
        pack_bin_list = []
        if old_packed_bin_list[-1].ratio < 0.4:
            repack = True
            # 用packing_algorithms4重新装载
            for route in self.routes:
                last_bin = None
                for platform in route:
                    box_in_platform = [  # (Han: 在某个platform上的所要装载的boxes的集合)
                        box for box in self.box_list if box.platform == platform]
                    if not box_in_platform:
                        continue  # 下一个platform
                    # packing_algorithms1和packing_algorithms2选一个
                    pack_bins = SSBH4.pack(self.bin, box_in_platform, last_bin=last_bin)  # （Han:返回的是装载车辆相关信息）
                    # pack_bins = packing_algorithms1.SSBH.pack(self.bin, box_in_platform, last_bin=last_bin)  # （Han:返回的是装载车辆相关信息）
                    pack_bin_list.extend(pack_bins[:-1])  # (Han:不能继续装载的车的信息存入pack_bin_list中)
                    last_bin = pack_bins[-1]
                if last_bin:
                    pack_bin_list.append(last_bin)
                else:
                    raise Exception("Packing Fails.")
        else:
            pack_bin_list = old_packed_bin_list
        return pack_bin_list, repack

    def replace_big_bin_by_small_bin(self, pack_bin_list, mark_ratio=0.5):
        """
            用选好的bin装载完成后，在满足一定条件下，尝试用小的bin去替代，如果小bin能完全装载当前bin所装载的box，
            则用小bin装载，否则不进行替换
        """
        new_packed_bin_list = []
        # 存储两个bin的容量比
        two_bin_ratio_ListDict = []
        packed_bin_v = pack_bin_list[0].volume
        packed_bin_w = pack_bin_list[0].max_weight
        for truck in self.data["algorithmBaseParamDto"]["truckTypeDtoList"]:
            truck["volume"] = truck["length"] * truck["width"] * truck["height"]
            two_bin_ratio = min(truck["volume"] / packed_bin_v, truck["maxLoad"] / packed_bin_w)
            two_bin_ratio_ListDict.append({"truckTypeCode": truck["truckTypeCode"], \
                                           "two_bin_ratio": two_bin_ratio})
        for packed_bin in pack_bin_list:
            if packed_bin.ratio <= mark_ratio:
                redundancy = 0.1
            else:  # 若装载率>=0.8,不进行替换
                new_packed_bin_list.append(packed_bin)
                continue
            candi_replace_bin = [bin for bin in two_bin_ratio_ListDict
                                    if bin["two_bin_ratio"]>(packed_bin.ratio+redundancy) and bin["two_bin_ratio"]<1]
            candi_replace_bin.sort(key=lambda bin: bin["two_bin_ratio"], reverse=False)  # 升序
            flag = 0
            for bin in candi_replace_bin:
                truck_code = bin["truckTypeCode"]
                # self.bin在self.pack()中用于初始化一个新的容器
                self.bin = self._gen_bin(truck_code)
                box_list = packed_bin.box_list
                box_by_platform_2dList = utils.classify_box_by_platform(box_list)
                last_bin = None
                for box_in_platform in box_by_platform_2dList:
                    pack_bins = SSBH4.pack(self.bin, box_in_platform, last_bin=last_bin)
                    # pack_bins = packing_algorithms1.SSBH.pack(self.bin, box_in_platform, last_bin=last_bin)
                    if len(pack_bins) == 1 and pack_bins[0].ratio > 0:
                        last_bin = pack_bins[0]
                    else:  # 不使用该bin进行替换
                        flag += 1
                        break
                if len(pack_bins) == 1 and pack_bins[0].ratio > 0:
                    # 进行替换
                    new_packed_bin_list.append(pack_bins[0])
                    # print("Replacement succeeded.")
                    break
            # 若遍历完candi_replace_bin仍没有合适的替换，则使用原bin
            if flag == len(candi_replace_bin):
                new_packed_bin_list.append(packed_bin)
                # print("Replacement failed.")
        self.bin = self.origin_bin
        return new_packed_bin_list

    def replace_big_bin_by_small_bin_2(self, pack_bin_list, mark_ratio=0.5):
        """
            用选好的bin装载完成后，在满足一定条件下，尝试用小的bin去替代，如果小bin能完全装载当前bin所装载的box，
            则用小bin装载，否则不进行替换;
            先用packing_algorithms4去replace, 若不成功，则再用packing_algorithms2去replace;
        """
        new_packed_bin_list = []
        # 存储两个bin的容量比
        two_bin_ratio_ListDict = []
        packed_bin_v = pack_bin_list[0].volume
        packed_bin_w = pack_bin_list[0].max_weight
        for truck in self.data["algorithmBaseParamDto"]["truckTypeDtoList"]:
            truck["volume"] = truck["length"] * truck["width"] * truck["height"]
            two_bin_ratio = min(truck["volume"] / packed_bin_v, truck["maxLoad"] / packed_bin_w)
            two_bin_ratio_ListDict.append({"truckTypeCode": truck["truckTypeCode"], \
                                           "two_bin_ratio": two_bin_ratio})
        for packed_bin in pack_bin_list:
            if packed_bin.ratio <= mark_ratio:
                redundancy = 0.1
            else:  # 若装载率>=mark_ratio,不进行替换
                new_packed_bin_list.append(packed_bin)
                continue
            candi_replace_bin = [bin for bin in two_bin_ratio_ListDict
                                    if bin["two_bin_ratio"]>(packed_bin.ratio+redundancy) and bin["two_bin_ratio"]<1]
            candi_replace_bin.sort(key=lambda bin: bin["two_bin_ratio"], reverse=False)  # 升序
            flag = 0
            for bin in candi_replace_bin:
                truck_code = bin["truckTypeCode"]
                # self.bin在self.pack()中用于初始化一个新的容器
                self.bin = self._gen_bin(truck_code)
                box_list = packed_bin.box_list
                box_by_platform_2dList = utils.classify_box_by_platform(box_list)
                last_bin = None
                for box_in_platform in box_by_platform_2dList:
                    pack_bins = SSBH4.pack(self.bin, box_in_platform, last_bin=last_bin)
                    # pack_bins = packing_algorithms1.SSBH.pack(self.bin, box_in_platform, last_bin=last_bin)
                    if len(pack_bins) == 1 and pack_bins[0].ratio > 0:
                        last_bin = pack_bins[0]
                    else:  # 不使用该bin进行替换
                        flag += 1
                        break
                if len(pack_bins) == 1 and pack_bins[0].ratio > 0:
                    # 进行替换
                    new_packed_bin_list.append(pack_bins[0])
                    # print("Replacement succeeded.")
                    break
            # 若遍历完candi_replace_bin仍没有合适的替换，则再用packing_algorithms2来replace
            if flag == len(candi_replace_bin):
                flag = 0
                for bin in candi_replace_bin:
                    truck_code = bin["truckTypeCode"]
                    # self.bin在self.pack()中用于初始化一个新的容器
                    self.bin = self._gen_bin(truck_code)
                    box_list = packed_bin.box_list
                    box_by_platform_2dList = utils.classify_box_by_platform(box_list)
                    last_bin = None
                    for box_in_platform in box_by_platform_2dList:
                        pack_bins = SSBH1.pack(self.bin, box_in_platform, last_bin=last_bin)
                        # pack_bins = packing_algorithms1.SSBH.pack(self.bin, box_in_platform, last_bin=last_bin)
                        if len(pack_bins) == 1 and pack_bins[0].ratio > 0:
                            last_bin = pack_bins[0]
                        else:  # 不使用该bin进行替换
                            flag += 1
                            break
                    if len(pack_bins) == 1 and pack_bins[0].ratio > 0:
                        # 进行替换
                        new_packed_bin_list.append(pack_bins[0])
                        # print("Replacement succeeded.")
                        break
                # 若遍历完candi_replace_bin仍没有合适的替换，则使用原bin
                if flag == len(candi_replace_bin):
                    new_packed_bin_list.append(packed_bin)
                    # print("Replacement failed.")
        self.bin = self.origin_bin
        return new_packed_bin_list

    def _pack(self):
        return SSBH1.pack(self.bin, self.box_list)

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
                continue  # Han:如果spu_ids不为None,则只从订单中提取spu_ids中存储的box的信息;否则提取所有boxes的信息
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
        if packed_bin_list:  # (Han:packed_bin_list中的每个entry存储的是每辆车的路径信息和装载信息)
            for packed_bin in packed_bin_list:  # (Han:对于每一辆车)
                truck = {}
                truck["truckTypeId"] = packed_bin.truck_type_id
                truck["truckTypeCode"] = packed_bin.truck_type_code
                truck["piece"] = len(packed_bin.packed_box_list)  # (Han:一共装载了多少boxes)
                truck["volume"] = packed_bin.load_volume  # (Han:装载的货物的体积)
                truck["weight"] = packed_bin.load_weight
                truck["innerLength"] = packed_bin.length
                truck["innerWidth"] = packed_bin.width
                truck["innerHeight"] = packed_bin.height
                truck["maxLoad"] = self.truck["maxLoad"]  # （Han:最大装载重量）
                spu_list = []
                for packed_box in packed_bin.packed_box_list:  # (Han:对于每一辆车的每一个box)
                    spu = {}
                    packed_box.change_ordinate(packed_bin)  # (Han:转换该box的坐标，坐标系中心改为集装箱中心,packed_box:entity.PackedBox类型)
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
                    spu_list.append(spu)  # Han:spu_list中的每个entry都是一个box的相关信息
                spu_list.sort(key=lambda box: box['order'])  # Han:按‘order’字段进行排序
                truck["spuArray"] = spu_list  # Han:truck['spuArray']存储该卡车所有装载的boxes的信息
                platform_list = []
                for spu in spu_list:
                    if spu['platformCode'] not in platform_list:
                        platform_list.append(spu['platformCode'])
                truck["platformArray"] = platform_list  # Han:truck['platformArray']存储该车的访问路径
                truck_list.append(truck)  # Han:truck_list存储每一辆卡车的相关信息
        res["solutionArray"].append(truck_list)  # Han:res["estimateCode"]存储了一个订单的json文件名,res["solutionArray"]存储了该订单的解solution
        return res, truck_list

def main(argv):
    # input_dir = argv[1] # 原来的语句
    # output_dir = argv[2] # 原来的语句
    input_dir = "E:\\VSCodeSpace\\HuaweiCompetitionSpace\\mycodes\\SDVRP-Improve_hwpacking\\data\\inputs"  # 为方便调试我加上的语句
    output_dir = "E:\\VSCodeSpace\\HuaweiCompetitionSpace\\mycodes\\SDVRP-Improve_hwpacking\\data\\outputs"  # 为方便调试我加上的语句
    i = 0
    for file_name in os.listdir(input_dir):
        # i += 1
        # if i == 2:
        #     break
        print("----------", file_name, "---------------")
        input_path = os.path.join(input_dir, file_name)
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            message_str = f.read()
        # route = [["platform27", "platform04", "platform32", "platform08", "platform02", "platform09", "platform07", "platform34", "platform25"]]
        pack = Pack(message_str,
                    truck_code=None,
                    # route=route,
                    output_path=output_dir)
        pack.run()
        # raw_trucks = pack.data["algorithmBaseParamDto"]["truckTypeDtoList"]
        # trucks_code = []
        # for truck in raw_trucks:
        #     trucks_code.append(truck["truckTypeCode"])
        # # print(i, ": ", trucks_code)
        # box_num = len(pack.box_list)
        # platform_num = len(pack.data['algorithmBaseParamDto']['platformDtoList'])
        # print(i, "box: ", box_num, " platform: ", platform_num)
        # i += 1



if __name__ == "__main__":
    t1 = time.process_time()
    main(sys.argv)
    t2 = time.process_time()
    print("cpu time:", t2 - t1)