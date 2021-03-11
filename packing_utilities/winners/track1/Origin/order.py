import json
import shutil
import statistics
import sys
import os
import heapq
import numpy as np

from entity import AlgorithmBox, Bin


class Order:
    def __init__(self, message_str):
        """
        The data of the order/ problem instance, including pickup points, boxes, trucks, etc.
        : param message_str: input params string, extracted from input json file.
        """
        super().__init__()
        self.data = json.loads(message_str)
        self.bin = self._gen_bin()
        self.trucks_listdict = self.data["algorithmBaseParamDto"]["truckTypeDtoList"]
        self.file_name = self.data['estimateCode']
        self.allplatform_listdict = self.data["algorithmBaseParamDto"]["platformDtoList"]
        self.platform_num = len(self.allplatform_listdict)
        self.boxes_listAlgorithmBox = self._gen_box_list()
        self.boxes_by_platform_dictlistAlgorithmBox,\
            self.boxes_volume_sum_by_platform_dict,\
                self.boxes_weight_sum_by_platform_dict = self._gen_boxes_by_platform()
        self.distanceMap_2dMatrix = self._gen_distance()
        self.pack_pattern_each_platform_dictlistdictlistAlgorithmBox =\
            self._gen_pack_pattern()
        self.nb_proportion_of_platform = 0.3  # 超参，可与当前platform组2C-SP的platform比例
        # self.potential_CP_of_all_platform_dictliststr = self._gen_2C_SP()

    def _gen_distance(self):
        """ 索引0表示starting point,索引N+1表示ending point """
        raw_distanceMap_dict = self.data["algorithmBaseParamDto"]["distanceMap"]
        distanceMap_2dMatrix = np.full(
            (self.platform_num+2, self.platform_num+2), np.inf)
        for i in range(1, self.platform_num+1):
            platformCode_str =\
                self.allplatform_listdict[i-1]["platformCode"]  # 这里索引要用i-1
            distanceMap_2dMatrix[0][i] =\
                raw_distanceMap_dict["start_point+"+platformCode_str]
            distanceMap_2dMatrix[i][self.platform_num+1] =\
                raw_distanceMap_dict[platformCode_str+"+end_point"]
            for j in range(1, self.platform_num+1):
                if i == j:
                    continue
                else:
                    another_platformCode_str =\
                        self.allplatform_listdict[j-1]["platformCode"]  # 这里索引要用j-1
                    distanceMap_2dMatrix[i][j] =\
                        raw_distanceMap_dict[platformCode_str+"+"+another_platformCode_str]
                    distanceMap_2dMatrix[j][i] =\
                        raw_distanceMap_dict[another_platformCode_str+"+"+platformCode_str]
        return distanceMap_2dMatrix

    def _gen_bin(self):
        """ 从可使用的trucks中返回要使用的最大体积的truck/bin """
        raw_trucks_listdict = self.data["algorithmBaseParamDto"]["truckTypeDtoList"]
        truck_vols_list = [
            raw_truck_dict["length"] * raw_truck_dict["width"] * raw_truck_dict["height"]
            for raw_truck_dict in raw_trucks_listdict
        ]
        max_vol_truck_index = truck_vols_list.index(max(truck_vols_list))
        self.truck = raw_trucks_listdict[max_vol_truck_index]
        bin_dict = {}
        bin_dict["bin_type"] = self.truck["truckTypeName"]
        bin_dict["length"] = self.truck["length"]
        bin_dict["width"] = self.truck["width"]
        bin_dict["height"] = self.truck["height"]
        bin_dict["volume"] = (self.truck["length"]
                              * self.truck["width"]
                              * self.truck["height"])
        bin_dict["max_weight"] = self.truck["maxLoad"]
        bin_dict["truck_type_id"] = self.truck["truckTypeId"]
        bin_dict["truck_type_code"] = self.truck["truckTypeCode"]

        return Bin(**bin_dict)

    def _gen_box_list(self):
        raw_box_listdict = self.data["boxes"]
        box_list = []
        for raw_box in raw_box_listdict:
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
            # box['volume'] = raw_box['length']*raw_box['width']*raw_box['height']
            box['is_cylinder'] = False
            box_list.append(box)
        box_listAlgorithmBox = []
        for box in box_list:
            box_listAlgorithmBox.append(AlgorithmBox(**box))
        return box_listAlgorithmBox

    def _gen_boxes_by_platform(self):
        boxes_in_diff_platform_dictlistAlgorithmBox = {}
        boxes_volume_sum_by_platform_dict = {}
        boxes_weight_sum_by_platform_dict = {}
        for platform_dict in self.allplatform_listdict:
            boxes_in_diff_platform_dictlistAlgorithmBox[platform_dict["platformCode"]] = []
            boxes_volume_sum_by_platform_dict[platform_dict["platformCode"]] = 0.0
            boxes_weight_sum_by_platform_dict[platform_dict["platformCode"]] = 0.0
        for algobox in self.boxes_listAlgorithmBox:
            boxes_in_diff_platform_dictlistAlgorithmBox[algobox.platform].append(algobox)
            boxes_volume_sum_by_platform_dict[algobox.platform] +=\
                (algobox.length * algobox.width * algobox.height)
            boxes_weight_sum_by_platform_dict[algobox.platform] +=\
                algobox.weight
        return boxes_in_diff_platform_dictlistAlgorithmBox,\
            boxes_volume_sum_by_platform_dict, boxes_weight_sum_by_platform_dict

    def _gen_pack_pattern(self):
        """ 对每个platform构造1C-FLPs和1C-SP """
        pack_pattern_each_platform_dictlistdictlistAlgorithmBox = {}
        for i in range(self.platform_num):            
            platformCode_str = self.allplatform_listdict[i]["platformCode"]
            pack_pattern_each_platform_dictlistdictlistAlgorithmBox[platformCode_str] = []
            full_load_truck_num_by_v =\
                self.boxes_volume_sum_by_platform_dict[platformCode_str] / self.bin.volume
            full_load_truck_num_by_w =\
                self.boxes_weight_sum_by_platform_dict[platformCode_str] / self.bin.max_weight
            full_load_truck_num_by_vORw = 0
            pack_by_v_bool = True
            if full_load_truck_num_by_v > full_load_truck_num_by_w and\
                full_load_truck_num_by_v > 1:  # 按volume生成1C-FLP和1C-SP
                full_load_truck_num_by_vORw = full_load_truck_num_by_v
                pack_by_v_bool = True
            elif full_load_truck_num_by_w > full_load_truck_num_by_v and\
                full_load_truck_num_by_w > 1:  # 按weight生成1C-FLP和1C-SP
                full_load_truck_num_by_vORw = full_load_truck_num_by_w
                pack_by_v_bool = False
            else:  # 只生成一个1C-SP
                full_load_truck_num_by_vORw = 0
            FLPs_dict, SP_dict = self._gen_packpattern_by_platform(
                platformCode_str, pack_by_v_bool, full_load_truck_num_by_vORw)
            pack_pattern_each_platform_dictlistdictlistAlgorithmBox[
                platformCode_str].append(FLPs_dict)
            pack_pattern_each_platform_dictlistdictlistAlgorithmBox[\
                    platformCode_str].append(SP_dict)
        return pack_pattern_each_platform_dictlistdictlistAlgorithmBox

    def _gen_packpattern_by_platform(self, platformCode_str, pack_by_v, truck_num_by_vORw):
        """
        为某个platform构造1C-FLPs和1C-SP.
        @return: oneC_FLPs, oneC_SP均为dictionary类型
        """
        truck_capacity = 0
        total_load = 0
        if pack_by_v is True:
            self.boxes_by_platform_dictlistAlgorithmBox[platformCode_str].sort(
                key=sort_boxes_by_volume, reverse=True)
            truck_capacity = self.bin.volume
        else:
            self.boxes_by_platform_dictlistAlgorithmBox[platformCode_str].sort(
                key=sort_boxes_by_weight, reverse=True)
            truck_capacity = self.bin.max_weight
        oneC_FLPs = {}
        oneC_FLPs["1C-FLPNum"] = int(truck_num_by_vORw)
        oneC_FLPs["1C-FLPs"] = [[] for i in range(int(truck_num_by_vORw))]
        oneC_FLPs["total_boxes_volume"] = [0 for i in range(int(truck_num_by_vORw))]
        oneC_FLPs["total_boxes_weight"] = [0 for i in range(int(truck_num_by_vORw))]
        oneC_FLPs["boxes_num"] = [0 for i in range(int(truck_num_by_vORw))]
        oneC_SP = {}
        oneC_SP["1C-SPNum"] = 1  # 每个pickup point只有1个1C-SP
        oneC_SP["1C-SP"] = []
        oneC_SP["total_boxes_volume"] = 0
        oneC_SP["total_boxes_weight"] = 0
        oneC_SP["boxes_num"] = 0
        full_load_truck_num = 0
        boxes_num_this_platform =\
            len(self.boxes_by_platform_dictlistAlgorithmBox[platformCode_str])
        if pack_by_v:  # 按boxes‘ volume装载
            j = 0
            while j < boxes_num_this_platform:
                if full_load_truck_num < oneC_FLPs["1C-FLPNum"]:  # 进行1C-FLP划分
                    total_load += self.boxes_by_platform_dictlistAlgorithmBox[
                        platformCode_str][j].volume
                    if total_load < truck_capacity:
                        oneC_FLPs["1C-FLPs"][full_load_truck_num].append(
                            self.boxes_by_platform_dictlistAlgorithmBox[platformCode_str][j])
                        oneC_FLPs["total_boxes_volume"][full_load_truck_num] = total_load
                        oneC_FLPs["total_boxes_weight"][full_load_truck_num] +=\
                            self.boxes_by_platform_dictlistAlgorithmBox[
                                platformCode_str][j].weight
                        oneC_FLPs["boxes_num"][full_load_truck_num] += 1
                        j += 1
                    else:
                        total_load = 0
                        full_load_truck_num += 1
                else:  # 剩下的boxes均属于1C-SP
                    oneC_SP["1C-SP"].append(
                        self.boxes_by_platform_dictlistAlgorithmBox[platformCode_str][j])
                    oneC_SP["total_boxes_volume"] +=\
                        self.boxes_by_platform_dictlistAlgorithmBox[platformCode_str][j].volume
                    oneC_SP["total_boxes_weight"] +=\
                        self.boxes_by_platform_dictlistAlgorithmBox[platformCode_str][j].weight
                    oneC_SP["boxes_num"] += 1
                    j += 1
        else:  # 按boxes‘ weight装载
            j = 0
            while j < boxes_num_this_platform:
                if full_load_truck_num < oneC_FLPs["1C-FLPNum"]:  # 对boxes进行1C-FLP划分
                    total_load += self.boxes_by_platform_dictlistAlgorithmBox[
                        platformCode_str][j].weight
                    if total_load < truck_capacity:
                        oneC_FLPs["1C-FLPs"][full_load_truck_num].append(
                            self.boxes_by_platform_dictlistAlgorithmBox[platformCode_str][j])
                        oneC_FLPs["total_boxes_weight"][full_load_truck_num] = total_load
                        oneC_FLPs["total_boxes_volume"][full_load_truck_num] +=\
                            self.boxes_by_platform_dictlistAlgorithmBox[
                                platformCode_str][j].volume
                        oneC_FLPs["boxes_num"][full_load_truck_num] += 1
                        j += 1
                    else:
                        total_load = 0
                        full_load_truck_num += 1
                else:  # 剩下的boxes均属于1C-SP
                    oneC_SP["1C-SP"].append(
                        self.boxes_by_platform_dictlistAlgorithmBox[platformCode_str][j])
                    oneC_SP["total_boxes_volume"] +=\
                        self.boxes_by_platform_dictlistAlgorithmBox[platformCode_str][j].volume
                    oneC_SP["total_boxes_weight"] +=\
                        self.boxes_by_platform_dictlistAlgorithmBox[platformCode_str][j].weight
                    oneC_SP["boxes_num"] += 1
                    j += 1
        return oneC_FLPs, oneC_SP

    def _gen_2C_SP(self):
        potential_CP_of_all_platform_dictliststr = {}
        for platform in self.allplatform_listdict:
            potential_CP_of_all_platform_dictliststr[platform["platformCode"]] = []
        dist_map = self.distanceMap_2dMatrix
        N = self.platform_num
        potential_CP_num_every_platform =\
            int((N-1) * self.nb_proportion_of_platform) + 1  # 向上取整
        insertion_cost_2dMatrix = np.full((self.platform_num, self.platform_num), np.inf)
        for i in range(1, self.platform_num):
            for j in range(i+1, self.platform_num + 1):
                insertion_cost_2dMatrix[i-1][j-1] =\
                    dist_map[0][i] + dist_map[i][j] - dist_map[j][N+1]
                insertion_cost_2dMatrix[j-1][i-1] =\
                    dist_map[0][j] + dist_map[j][i] - dist_map[i][N+1]
        for i in range(N):
            flag_value = heapq.nsmallest(
                potential_CP_num_every_platform, insertion_cost_2dMatrix[i])[-1]
            platformCode_str = self.allplatform_listdict[i]["platformCode"]
            for j in range(N):
                if insertion_cost_2dMatrix[i][j] <= flag_value:
                    another_platformCode = self.allplatform_listdict[j]["platformCode"]
                    potential_CP_of_all_platform_dictliststr[platformCode_str].append(
                        another_platformCode)
        return potential_CP_of_all_platform_dictliststr


def sort_boxes_by_volume(AlgorithmBox):
    return AlgorithmBox.volume


def sort_boxes_by_weight(AlgorithmBox):
    return AlgorithmBox.weight


if __name__ == "__main__":
    input_dir = "E:\\VSCodeSpace\\HuaweiCompetitionSpace\\mycodes\\3L-SDVRP\\data\\inputs"
    output_dir = "E:\\VSCodeSpace\\HuaweiCompetitionSpace\\mycodes\\3L-SDVRP\\data\\outputs"
    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            message_str = f.read()
        one_order = Order(message_str)
        print(len(one_order.boxes_by_platform_dictlistAlgorithmBox))
        print(one_order.boxes_by_platform_dictlistAlgorithmBox['platform03'][0].all_directions)
        print(type(one_order.data['boxes'][0]))
        print("end")
