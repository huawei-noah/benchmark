""" same as mycode/3L-SDVRP/pack.py """
import sys
import os
import copy
import time
import myconstrains
from order import Order
# from routing import Routing
from entity import AlgorithmBox, Bin, PackedBin, PackedBox, Space, Area
from extremepoint_space import ExtremePointSpace
import general_utils as utils
import config
import json
import math


class Pack:
    def __init__(self, order, route=None, truck_code=None, spu_ids=None) -> None:
        """
            :param order: Order类对象
            :param route: 一条解码前的路径, e.g., [1,3,5,4,2...]
            :param truck_code: 卡车代码
            :parem spu_ids: 要装载的箱子的spu id集合
        """
        self.order = order
        self.data = order.data
        self.platform_num = self.order.platform_num
        self.allplatformlist_ListDict = self.order.allplatform_listdict
        self.platformCode_list = [platform["platformCode"] for platform in self.allplatformlist_ListDict]
        self.pack_patterns_dictlistdictlistAlgorithmBox =\
            self.order.pack_pattern_each_platform_dictlistdictlistAlgorithmBox
        self.truck = None
        self.truck_list = []
        self.all_box_list, self.all_box_volume_sum, self.all_box_weight_sum = self._gen_box_list()
        self.trucks_listdict = self.data["algorithmBaseParamDto"]["truckTypeDtoList"]
        # self.origin_bin = self._gen_bin(self.selc_bin())
        self.origin_bin = self._gen_bin()
        self.revs_bin = self.revise_selc_bin()
        self.bin = self.revs_bin
        # self.bin = self._gen_bin(truck_code)
        # self.box_list = self._gen_box_list(spu_ids)

    def selc_bin(self):
        """ 根据box总体积重量和各bin/truck的容量，选择所需数量最少的车型作为主要装载车型；
            若所需数量最少的车型有多种车型，则选择容量小的
        """
        if len(self.trucks_listdict) == 1:
            truck = self.trucks_listdict[0]
            truck["volume"] = truck["length"] * truck["width"] * truck["height"]
            truck_code = self.trucks_listdict[0]["truckTypeCode"]
            return truck_code
        else:
            for truck in self.trucks_listdict:
                truck["volume"] = truck["length"] * truck["width"] * truck["height"]
                # 只用该车型装载，按体积、重量该车型至少需要多少辆t_num
                v_num = math.ceil(self.all_box_volume_sum / truck["volume"])
                w_num = math.ceil(self.all_box_weight_sum / truck["maxLoad"])
                t_num = max(v_num, w_num)
                truck["num"] = t_num
            # 按所需数量升序
            self.trucks_listdict.sort(key=lambda truck: truck["num"], reverse=False)
            # 选出所有所需数量最少的车型
            candidate_truck = [truck for truck in self.trucks_listdict
                               if truck["num"] == self.trucks_listdict[0]["num"]]
            if len(candidate_truck) == 1:
                truck_code = candidate_truck[0]["truckTypeCode"]
                return truck_code
            else:
                # 按体积升序排列选体积最小的
                candidate_truck.sort(key= lambda truck: truck["volume"], reverse=False)
                truck_code = candidate_truck[0]["truckTypeCode"]
                return truck_code

    def revise_selc_bin(self):
        """ 如果有些box超过了self.origin_bin的尺寸，就选尺寸最大的bin来装载 """
        all_boxes = copy.deepcopy(self.all_box_list)
        all_boxes.sort(key= lambda box: (box.height, box.length, box.width), reverse=True)  # 降序
        for box in all_boxes:
            if box.height > self.origin_bin.height or box.length > self.origin_bin.length or box.width > self.origin_bin.width:
                self.origin_bin = self.order.bin
                return self.origin_bin  # 最大的车辆
        return self.origin_bin

    def _pack_by_platform(self, route, selc_box_list=None):  # route上各platform的SP模式货物由Order控制
        """
            按route上的platform顺序进行装载; pack()的pack_method也是一个可调的参数
            :param route: 每个元素都是一个platformCode;
            :param selc_box_list: 从selc_box_list中(按route上的platform顺序)取box进行装载;
                若为None,则从self.all_box_list(全部boxes)中(按route上的platform顺序)取box进行装载
        """
        pack_bin_list = []
        last_bin = None
        box_w_v_Dict = {}  # 存储每个platform所有boxes的总体积总重量
        box_list = []
        box_by_platform_2dList = []  # 每个元素都是一个list，存储同一个platform上的boxes
        if selc_box_list:
            boxes = selc_box_list
        else:
            boxes = self.all_box_list
        for platform in route:
            box_w_v_Dict[platform] = {"volume": 0, "weight": 0}
            # 按platform在route中的顺序对boxes重新整理
            box_in_platform = [  # (Han: 在某个platform上的所要装载的boxes的集合)
                box for box in boxes if box.platform == platform]
            # 对boxes按一定底面积和体积降序排列
            box_in_platform.sort(
                key=lambda AB: (AB.length * AB.width, AB.height, AB.width, AB.length), reverse=True)
            if not box_in_platform:
                continue  # 下一个platform
            else:
                box_list.extend(box_in_platform)
                box_by_platform_2dList.append(box_in_platform)
                volume, weight = utils.get_total_volume_weight(box_in_platform)
                box_w_v_Dict[platform]["volume"] = volume
                box_w_v_Dict[platform]["weight"] = weight
        while box_list:
            print("------------------------------")
            pack_bins, remaining_box_Dict = self.pack(box_list, box_by_platform_2dList, pack_method=1, last_bin=last_bin, box_w_v_Dict=box_w_v_Dict, p=0)  # （Han:返回的是装载车辆相关信息）
            if pack_bins:
                if pack_bins[0].ratio != 0:
                    pack_bin_list.append(pack_bins[0])  # (不能继续装载的车的信息存入pack_bin_list中)
            else:
                raise Exception("Packing fails.")
            last_bin = None
            box_list = remaining_box_Dict["box_list"]
            box_by_platform_2dList = remaining_box_Dict["box_by_platform_2DList"]
        # 如果最后一辆车的装载率很小（如<0.1）, 看能否装到其他车上
        # pack_bin_list = self.transfer_box_for_low_loading_bin(pack_bin_list)
        pack_bin_list = self.replace_big_bin_by_small_bin(pack_bin_list)
        return pack_bin_list

    def _pack_FLP(self, platformCode, FLP_of_this_platform_Dict):
        packed_bin_List = []
        FLPs_2DList = FLP_of_this_platform_Dict["1C-FLPs"]
        FLP_num = FLP_of_this_platform_Dict["1C-FLPNum"]
        last_bin = None
        for i in range(FLP_num):
            box_in_platform = FLPs_2DList[i]
            pack_bins, remaining_boxes_Dict = self.pack(box_in_platform, last_bin=last_bin)
            # 将未装箱的boxes放入下一个FLP(若有)或者放入同platform的SP
            if FLP_num > 1 and i != (FLP_num + 1):
                FLPs_2DList[i+1].extend(remaining_boxes_Dict["boxes_list"])
            else:
                self.pack_patterns_dictlistdictlistAlgorithmBox[
                    platformCode][1]["1C-SP"].extend(remaining_boxes_Dict["boxes_list"])
                self.pack_patterns_dictlistdictlistAlgorithmBox[
                    platformCode][1]["boxes_num"] += remaining_boxes_Dict["boxes_num"]
                self.pack_patterns_dictlistdictlistAlgorithmBox[
                    platformCode][1]["total_boxes_volume"] += remaining_boxes_Dict["boxes_volume"]
                self.pack_patterns_dictlistdictlistAlgorithmBox[
                    platformCode][1]["total_boxes_weight"] += remaining_boxes_Dict["boxes_weight"]

            if len(pack_bins) == 1:
                packed_bin_List.append(pack_bins[0])
            else:
                raise Exception("Here, the length of pack_bins should be 1.")
        return packed_bin_List

    def pack(self, box_List, box_by_platform, pack_method, last_bin=None, box_w_v_Dict=None, p=0):
        """
            该方法只装载一个bin
            :param box_List: 需要装载的箱子列表, 每个元素都是AlgorithmBox类型
            :param box_by_platform: 2D list, 每个元素都是一个list，存储同一个platform上的boxes
            :param pack_method: 1: pack_selc_space_by_box; 2: pack_selc_box_by_space
            :param last_bin: 上次装载的容器, packed_bin类型
            :param box_w_v_Dict: 每个platform原始box的总重量体积
            :param p: 每个platform被装载的box占该platform的比例，
                      两种方式：当box_w_v_Dict=None或p=1时，该比例以现存未装载的box为分母；
                      否则，以platform初始所有box为分母
        """
        pack_bins_list = []
        remaining_box_Dict = {"box_list": [],
                              "box_by_platform_2DList": [],
                              "box_num": 0,
                              "box_volume": 0,
                              "box_weight": 0}  # 用于存放没有装载上的boxes
        # 如果上一次装载的容器last_bin不为None，使用上一次装载的容器
        if last_bin:
            packed_bin = copy.deepcopy(last_bin)
            last_bin = None
            EPspace_obj = packed_bin.space_obj  # ExtremePointSpace类
            EPspace_list = EPspace_obj.space_list  # 计划用来存储ExtremePoint可以使用的空间,每个元素都是Space类
            """ 此处可能TO BE CONTINUED """
        # 否则初始化一个新的容器, cf. PackedBin.create_by_bin()
        else:
            packed_bin = PackedBin.create_by_bin(self.bin)
            extreme_point = [0, 0, 0]  # bin的backleftlow角
            packed_bin.space_obj = ExtremePointSpace(space_list=[
                Space(packed_bin.length, packed_bin.width, packed_bin.height, min_coord=extreme_point)
            ])
            EPspace_obj = packed_bin.space_obj  # ExtremePointSpace类
            EPspace_list = EPspace_obj.space_list  # 计划用来存储ExtremePoint可以使用的空间,每个元素都是Space类
        # 对box_List中的所有boxes按一定底面积和体积降序排列(这里不可，会违反FIFO)
        # box_List.sort(
        #     key= lambda AB: (AB.length * AB.width, AB.height, AB.width, AB.length), reverse=True)
        # load box one by one
        # packed_bin, remaining_box_Dict = self.pack_selc_space_by_box(packed_bin, box_List)
        if pack_method == 1:
            backup_packed_bin = copy.deepcopy(packed_bin)
            packed_bin, remaining_box_Dict = self.pack_selc_space_by_box(packed_bin, box_List, box_by_platform, box_w_v_Dict, p=p)
            ratio_volume = packed_bin.load_volume / packed_bin.volume
            ratio_weight = packed_bin.load_weight / packed_bin.max_weight
            packed_bin.ratio = max(ratio_volume, ratio_weight)
            if p > 0:
                if packed_bin.load_volume == 0 or packed_bin.ratio < 0.8:
                    # 说明当前各platfomr剩余的box不足以达到p, 或车辆装载率很低(比如低于0.5)，可以继续往该车上装
                    packed_bin, remaining_box_Dict = self.pack_selc_space_by_box(packed_bin, remaining_box_Dict["box_list"], remaining_box_Dict["box_by_platform_2DList"], box_w_v_Dict, p=p/2)  # p=0
                    ratio_volume = packed_bin.load_volume / packed_bin.volume
                    ratio_weight = packed_bin.load_weight / packed_bin.max_weight
                    packed_bin.ratio = max(ratio_volume, ratio_weight)
                if packed_bin.ratio < 0.5:
                    # p=1时，以现存box的体积/重量为分母
                    packed_bin, remaining_box_Dict = self.pack_selc_space_by_box(packed_bin, remaining_box_Dict["box_list"], remaining_box_Dict["box_by_platform_2DList"], box_w_v_Dict, p=p/4)  # p=0
                    ratio_volume = packed_bin.load_volume / packed_bin.volume
                    ratio_weight = packed_bin.load_weight / packed_bin.max_weight
                    packed_bin.ratio = max(ratio_volume, ratio_weight)
                if packed_bin.load_volume == 0:  # packed_bin.ratio == 0
                    packed_bin, remaining_box_Dict = self.pack_selc_space_by_box(backup_packed_bin, box_List, box_by_platform, box_w_v_Dict, p=0)
        elif pack_method == 2:
            iter_num = 0
            backup_packed_bin = copy.deepcopy(packed_bin)
            backup_box_by_platform = copy.deepcopy(box_by_platform)
            packed_bin, remaining_box_Dict = self.pack_selc_box_by_space(packed_bin, box_by_platform, box_w_v_Dict, p=p)
            ratio_volume = packed_bin.load_volume / packed_bin.volume
            ratio_weight = packed_bin.load_weight / packed_bin.max_weight
            packed_bin.ratio = max(ratio_volume, ratio_weight)
            if p > 0:
                if packed_bin.load_volume == 0 or packed_bin.ratio < 0.8:
                    # p=1时，以现存box的体积/重量为分母
                    packed_bin, remaining_box_Dict = self.pack_selc_box_by_space(packed_bin, remaining_box_Dict["box_by_platform_2DList"], box_w_v_Dict, p=p/2)  # p=0,p/2
                    # packed_bin, remaining_box_Dict = self.pack_selc_space_by_box(packed_bin, remaining_box_Dict["box_list"], remaining_box_Dict["box_by_platform_2DList"], box_w_v_Dict, p=p/2)  # p=0,p/2
                    ratio_volume = packed_bin.load_volume / packed_bin.volume
                    ratio_weight = packed_bin.load_weight / packed_bin.max_weight
                    packed_bin.ratio = max(ratio_volume, ratio_weight)
                # while (packed_bin.load_volume == 0 or packed_bin.ratio < 0.7) and iter_num < 3:
                #     # 说明当前各platfomr剩余的box不足以达到p, 或车辆装载率很低(比如低于0.5)，可以继续往该车上装
                #     packed_bin, remaining_box_Dict = self.pack_selc_space_by_box(backup_packed_bin, box_List, backup_box_by_platform, box_w_v_Dict, p=p/2)
                #     ratio_volume = packed_bin.load_volume / packed_bin.volume
                #     ratio_weight = packed_bin.load_weight / packed_bin.max_weight
                #     packed_bin.ratio = max(ratio_volume, ratio_weight)
                #     iter_num += 1
                if packed_bin.ratio < 0.5:
                    packed_bin, remaining_box_Dict = self.pack_selc_box_by_space(packed_bin, remaining_box_Dict["box_by_platform_2DList"], box_w_v_Dict, p=p/4) # p=0
                    # packed_bin, remaining_box_Dict = self.pack_selc_space_by_box(packed_bin, remaining_box_Dict["box_list"], remaining_box_Dict["box_by_platform_2DList"], box_w_v_Dict, p=1) # p=0
                    ratio_volume = packed_bin.load_volume / packed_bin.volume
                    ratio_weight = packed_bin.load_weight / packed_bin.max_weight
                    packed_bin.ratio = max(ratio_volume, ratio_weight)
                if packed_bin.load_volume == 0:
                    packed_bin, remaining_box_Dict = self.pack_selc_box_by_space(backup_packed_bin, backup_box_by_platform, box_w_v_Dict, p=0)
        """ 此处需再思考 """
        ratio_volume = packed_bin.load_volume / packed_bin.volume
        ratio_weight = packed_bin.load_weight / packed_bin.max_weight
        packed_bin.ratio = max(ratio_volume, ratio_weight)
        print("Load ratio:", packed_bin.ratio)
        pack_bins_list.append(packed_bin)
        if packed_bin.ratio == 0:
            # 当前车辆的空车容量仍无法装载，选体积最大的车辆
            self.bin = self._gen_bin()
        return pack_bins_list, remaining_box_Dict

    def pack_selc_space_by_box(self, packed_bin, box_List, box_by_platform=None, box_w_v_Dict=None, p=0):
        """
            对一个box，从packed_bin.space_obj.space_list的space中选择能装下该box的space进行装载
            :param box_List: 所有待装载的boxes
            :param box_by_platform: 2D list, 每一行是同一个platform的box的list
            :param box_w_v_Dict: 存储每个platform未装载时的所有boxes的体积重量，不会随着装载而改变
            :param p: 装载率，platform上被装载的box占该platform原始所有box的比率不能小于p
        """
        vst_following_platform = True  # 用以判断是否访问后续的platform
        remaining_box_Dict = {"box_list": [],
                              "box_by_platform_2DList": [],
                              "box_num": 0,
                              "box_volume": 0,
                              "box_weight": 0}  # 用于存放没有装载上的boxes
        for box_in_this_platform in box_by_platform:
            if not vst_following_platform:
                remaining_box_Dict["box_list"].extend(box_in_this_platform)
                remaining_box_Dict["box_by_platform_2DList"].append(box_in_this_platform)
                remaining_box_Dict["box_num"] += len(box_in_this_platform)
                continue
            backup_packed_bin = copy.deepcopy(packed_bin)
            packed_box_Dict = {"box_list": [], "box_volume": 0, "box_weight": 0}  # 存储该platform被装载的box
            not_packed_box_Dict = {"box_list": [], "box_volume": 0, "box_weight": 0}  # 存储未被装载的box
            for box in box_in_this_platform:
                # 两个方向
                packed_success_0 = False
                packed_success_1 = False
                # 将space_list中的space按照min_coor的x y z升序排序
                packed_bin.space_obj.space_list.sort(key=lambda s: (s.min_coord[0], s.min_coord[1], s.min_coord[2]), reverse=False)
                # 按space的排序开始装，能装到前面的space则不对后面的space进行装载
                for EPspace in packed_bin.space_obj.space_list:
                    # 对space的hold_surface进行去重，hold_surface是一个list，存储Area类型的支撑平面
                    if EPspace.hold_surface:
                        EPspace.hold_surface = utils.del_repeat_hold_area(EPspace.hold_surface)
                    # 两个方向装载box，注意packed_bin_0和packed_bin_1与packed_bin都是相互独立,
                    # 不能彼此影响
                    direction0 = 0  # 0对应100，box长度与卡车长度平行
                    direction1 = 1  # 1对应200， box宽度与卡车长度平行
                    if myconstrains.can_in_space(box, packed_bin, EPspace, direction0) and\
                        myconstrains.can_hold_box(box, EPspace, direction0) and \
                        myconstrains.can_in_bin(packed_bin, box.weight):
                        packed_success_0 = True
                        # if packed_bin.order == 100 and box.box_id == "94265910102H947":
                        #     print("here")
                        packed_bin_0, waste_space_0 =\
                            self.assign_box_2_space(packed_bin, EPspace, box, direction0)
                        packed_bin = packed_bin_0
                        packed_box_Dict["box_list"].append(box)
                        packed_box_Dict["box_volume"] += box.volume
                        packed_box_Dict["box_weight"] += box.weight
                        break
                    elif myconstrains.can_in_space(box, packed_bin, EPspace, direction1) and\
                        myconstrains.can_hold_box(box, EPspace, direction1) and \
                        myconstrains.can_in_bin(packed_bin, box.weight):
                        packed_success_1 = True
                        packed_bin_1, waste_space_1 =\
                            self.assign_box_2_space(packed_bin, EPspace, box, direction1)
                        packed_bin = packed_bin_1
                        packed_box_Dict["box_list"].append(box)
                        packed_box_Dict["box_volume"] += box.volume
                        packed_box_Dict["box_weight"] += box.weight
                        break
                    # if packed_success_0 and packed_success_1:
                    #     if waste_space_0 <= waste_space_1:
                    #         # 更新packed_bin
                    #         packed_bin = packed_bin_0
                    #         packed_box_Dict["box_list"].append(box)
                    #         packed_box_Dict["box_volume"] += box.volume
                    #         packed_box_Dict["box_weight"] += box.weight
                    #     else:
                    #         packed_bin = packed_bin_1
                    #         packed_box_Dict["box_list"].append(box)
                    #         packed_box_Dict["box_volume"] += box.volume
                    #         packed_box_Dict["box_weight"] += box.weight
                    #     break
                    # elif packed_success_0:
                    #     packed_bin = packed_bin_0
                    #     packed_box_Dict["box_list"].append(box)
                    #     packed_box_Dict["box_volume"] += box.volume
                    #     packed_box_Dict["box_weight"] += box.weight
                    #     break
                    # elif packed_success_1:
                    #     packed_bin = packed_bin_1
                    #     packed_box_Dict["box_list"].append(box)
                    #     packed_box_Dict["box_volume"] += box.volume
                    #     packed_box_Dict["box_weight"] += box.weight
                    #     break
                if (not packed_success_0) and (not packed_success_1):
                    if p == 0:
                        vst_following_platform = False
                    # 若遍历所有当前的EP space都没有装载成功，则保存
                    not_packed_box_Dict["box_list"].append(box)
                    not_packed_box_Dict["box_volume"] += box.volume
                    not_packed_box_Dict["box_weight"] += box.weight
            # 计算该platform的boxes的被装载率, 两种方式
            if box_w_v_Dict and packed_box_Dict["box_list"] and p != 1:  # 用该platform原始所有boxes的总体积重量做分母
                platformCode = packed_box_Dict["box_list"][0].platform
                v_load_rate = packed_box_Dict["box_volume"] / box_w_v_Dict[platformCode]["volume"]
                w_load_rate = packed_box_Dict["box_weight"] / box_w_v_Dict[platformCode]["weight"]
            else:  # 用该platform 到目前为止剩下的boxes的总体积重量做分母
                v_load_rate = packed_box_Dict["box_volume"] / \
                    (packed_box_Dict["box_volume"] + not_packed_box_Dict["box_volume"])
                w_load_rate = packed_box_Dict["box_weight"] / \
                    (packed_box_Dict["box_weight"] + not_packed_box_Dict["box_weight"])
            platform_load_rate = max(v_load_rate, w_load_rate)
            # 若该platform被装载的box未达到一定比率，则truck不访问该platform
            if platform_load_rate < p:  # 不接受该platform的box
                packed_bin = backup_packed_bin
                remaining_box_Dict["box_list"].extend(box_in_this_platform)
                remaining_box_Dict["box_by_platform_2DList"].append(box_in_this_platform)
                remaining_box_Dict["box_num"] += len(box_in_this_platform)
                remaining_box_Dict["box_volume"] += 0  # 暂时用不到，先不更新
                remaining_box_Dict["box_weight"] += 0  # 暂时用不到，先不更新
            else:  # 访问该platform
                if not_packed_box_Dict["box_list"]:
                    remaining_box_Dict["box_list"].extend(not_packed_box_Dict["box_list"])
                    remaining_box_Dict["box_by_platform_2DList"].append(not_packed_box_Dict["box_list"])
                    remaining_box_Dict["box_num"] += len(not_packed_box_Dict["box_list"])
                    remaining_box_Dict["box_volume"] += not_packed_box_Dict["box_volume"]
                    remaining_box_Dict["box_weight"] += not_packed_box_Dict["box_weight"]
        return packed_bin, remaining_box_Dict

    def pack_selc_box_by_space(self, packed_bin, box_by_platform, box_w_v_Dict=None, p=0):
        """
            对一个packed_bin.space_obj.space_list中的space中选择能装在该space中的box进行装载
            :param box_by_platform: 2D list, 每个元素都是一个list，存储同一个platform上的boxes
            :param box_w_v_Dict: 存储每个platform未装载时的所有boxes的体积重量，不会随着装载而改变
            :param p: 装载率，platform上被装载的box占该platform原始所有box的比率不能小于p
        """
        vst_following_platform = True
        remaining_box_Dict = {"box_list": [],
                              "box_by_platform_2DList":[],
                              "box_num": 0,
                              "box_volume": 0,
                              "box_weight": 0}  # 用于存放没有装载上的boxes
        for box_in_this_platform in box_by_platform:
            if not vst_following_platform:
                remaining_box_Dict["box_list"].extend(box_in_this_platform)
                remaining_box_Dict["box_by_platform_2DList"].append(box_in_this_platform)
                remaining_box_Dict["box_num"] += len(box_in_this_platform)
                continue
            backup_packed_bin = copy.deepcopy(packed_bin)
            backup_box_in_this_platform = copy.deepcopy(box_in_this_platform)
            packed_box_Dict = {"box_list": [], "box_volume": 0, "box_weight": 0}  # 存储该platform被装载的box
            not_packed_box_Dict = {"box_list": [], "box_volume": 0, "box_weight": 0}  # 存储未被装载的box
            # 两个方向
            packed_success_0 = False
            packed_success_1 = False
            # 将space_list中的space按照min_coor的x y z升序排序
            packed_bin.space_obj.space_list.sort(key=lambda s: (s.min_coord[0], s.min_coord[1], s.min_coord[2]), reverse=False)
            # 按space的排序开始装，能装到前面的space则不对后面的space进行装载
            i = 0
            space_num = len(packed_bin.space_obj.space_list)
            while i < space_num:
                EPspace = packed_bin.space_obj.space_list[i]
                #for EPspace in packed_bin.space_obj.space_list:
                j = 0  # 统计没有装入bin的箱子数
                # 对space的hold_surface进行去重，hold_surface是一个list，存储Area类型的支撑平面
                if EPspace.hold_surface:
                    EPspace.hold_surface = utils.del_repeat_hold_area(EPspace.hold_surface)
                if not box_in_this_platform:
                    break
                for box in box_in_this_platform:
                    box_num = len(box_in_this_platform)
                    # 两个方向装载box，注意packed_bin_0和packed_bin_1与packed_bin都是相互独立,
                    # 不能彼此影响
                    direction0 = 0  # 0对应100，box长度与卡车长度平行
                    direction1 = 1  # 1对应200， box宽度与卡车长度平行
                    if myconstrains.can_in_space(box, packed_bin, EPspace, direction0) and\
                        myconstrains.can_hold_box(box, EPspace, direction0) and \
                        myconstrains.can_in_bin(packed_bin, box.weight):
                        packed_success_0 = True
                        packed_bin_0, waste_space_0 =\
                            self.assign_box_2_space(packed_bin, EPspace, box, direction0)
                        packed_bin = packed_bin_0
                        packed_box_Dict["box_list"].append(box)
                        packed_box_Dict["box_volume"] += box.volume
                        packed_box_Dict["box_weight"] += box.weight
                        box_in_this_platform.remove(box)
                        # 将space_list中的space按照min_coor的x y z升序排序
                        packed_bin.space_obj.space_list.sort(key=lambda s: (s.min_coord[0], s.min_coord[1], s.min_coord[2]), reverse=False)
                        space_num = len(packed_bin.space_obj.space_list)
                        i = 0
                        break
                    elif myconstrains.can_in_space(box, packed_bin, EPspace, direction1) and\
                        myconstrains.can_hold_box(box, EPspace, direction1) and \
                        myconstrains.can_in_bin(packed_bin, box.weight):
                        packed_success_1 = True
                        packed_bin_1, waste_space_1 =\
                            self.assign_box_2_space(packed_bin, EPspace, box, direction1)
                        packed_bin = packed_bin_1
                        packed_box_Dict["box_list"].append(box)
                        packed_box_Dict["box_volume"] += box.volume
                        packed_box_Dict["box_weight"] += box.weight
                        box_in_this_platform.remove(box)
                        # 将space_list中的space按照min_coor的x y z升序排序
                        packed_bin.space_obj.space_list.sort(key=lambda s: (s.min_coord[0], s.min_coord[1], s.min_coord[2]), reverse=False)
                        space_num = len(packed_bin.space_obj.space_list)
                        i = 0
                        break
                    else:
                        j += 1
                    # if packed_success_0 and packed_success_1:
                    #     if waste_space_0 <= waste_space_1:
                    #         # 更新packed_bin
                    #         packed_bin = packed_bin_0
                    #         packed_box_Dict["box_list"].append(box)
                    #         packed_box_Dict["box_volume"] += box.volume
                    #         packed_box_Dict["box_weight"] += box.weight
                    #     else:
                    #         packed_bin = packed_bin_1
                    #         packed_box_Dict["box_list"].append(box)
                    #         packed_box_Dict["box_volume"] += box.volume
                    #         packed_box_Dict["box_weight"] += box.weight
                    #     box_in_this_platform.remove(box)
                    #     # 将space_list中的space按照min_coor的x y z升序排序
                    #     packed_bin.space_obj.space_list.sort(key=lambda s: (s.min_coord[0], s.min_coord[1], s.min_coord[2]), reverse=False)
                    #     space_num = len(packed_bin.space_obj.space_list)
                    #     i = 0
                    #     break
                    # elif packed_success_0:
                    #     packed_bin = packed_bin_0
                    #     packed_box_Dict["box_list"].append(box)
                    #     packed_box_Dict["box_volume"] += box.volume
                    #     packed_box_Dict["box_weight"] += box.weight
                    #     box_in_this_platform.remove(box)
                    #     # 将space_list中的space按照min_coor的x y z升序排序
                    #     packed_bin.space_obj.space_list.sort(key=lambda s: (s.min_coord[0], s.min_coord[1], s.min_coord[2]), reverse=False)
                    #     space_num = len(packed_bin.space_obj.space_list)
                    #     i = 0
                    #     break
                    # elif packed_success_1:
                    #     packed_bin = packed_bin_1
                    #     packed_box_Dict["box_list"].append(box)
                    #     packed_box_Dict["box_volume"] += box.volume
                    #     packed_box_Dict["box_weight"] += box.weight
                    #     box_in_this_platform.remove(box)
                    #     # 将space_list中的space按照min_coor的x y z升序排序
                    #     packed_bin.space_obj.space_list.sort(key=lambda s: (s.min_coord[0], s.min_coord[1], s.min_coord[2]), reverse=False)
                    #     space_num = len(packed_bin.space_obj.space_list)
                    #     i = 0
                    #     break
                    # else:
                    #     j += 1
                    if j == box_num:
                        i += 1
                packed_success_0 = False
                packed_success_1 = False
            # 该platform上未装载的个体
            volume, weight = utils.get_total_volume_weight(box_in_this_platform)
            not_packed_box_Dict["box_list"].extend(box_in_this_platform)
            not_packed_box_Dict["box_volume"] += volume
            not_packed_box_Dict["box_weight"] += weight
            # 计算该platform的boxes的被装载率, 两种方式
            if box_w_v_Dict and packed_box_Dict["box_list"] and p != 1:  # 用该platform原始所有boxes的总体积重量做分母
                platformCode = packed_box_Dict["box_list"][0].platform
                v_load_rate = packed_box_Dict["box_volume"] / box_w_v_Dict[platformCode]["volume"]
                w_load_rate = packed_box_Dict["box_weight"] / box_w_v_Dict[platformCode]["weight"]
            else:  # 用该platform 到目前为止剩下的boxes的总体积重量做分母
                v_load_rate = packed_box_Dict["box_volume"] / \
                    (packed_box_Dict["box_volume"] + not_packed_box_Dict["box_volume"])
                w_load_rate = packed_box_Dict["box_weight"] / \
                    (packed_box_Dict["box_weight"] + not_packed_box_Dict["box_weight"])
            platform_load_rate = max(v_load_rate, w_load_rate)
            if p == 0 and platform_load_rate != 1:
                # 当前platform上的box没有被完全装完，该车辆不再访问后续的platform
                vst_following_platform = False
            # 若该platform被装载的box未达到一定比率，则truck不访问该platform
            if platform_load_rate < p:  # 不接受该platform的box
                packed_bin = backup_packed_bin
                remaining_box_Dict["box_list"].extend(backup_box_in_this_platform)
                remaining_box_Dict["box_by_platform_2DList"].append(backup_box_in_this_platform)
                remaining_box_Dict["box_num"] += len(backup_box_in_this_platform)
                remaining_box_Dict["box_volume"] += 0  # 暂时用不到，先不更新
                remaining_box_Dict["box_weight"] += 0  # 暂时用不到，先不更新
            else:  # 访问该platform
                if box_in_this_platform:
                    # box_in_this_platform不空，剩余的归入remaining_box_Dict
                    remaining_box_Dict["box_list"].extend(box_in_this_platform)
                    remaining_box_Dict["box_by_platform_2DList"].append(box_in_this_platform)
                    remaining_box_Dict["box_num"] += len(box_in_this_platform)
                    remaining_box_Dict["box_volume"] += volume
                    remaining_box_Dict["box_weight"] += weight
        return packed_bin, remaining_box_Dict

    def assign_box_2_space(self, packed_bin, EPspace, box, box_direction):
        """
            将箱子放置到容器中 (cf. Huawei: space/general_tuils.py/assign_box_2_bin())
            同时要更新packed_bin里面的space_obj.space_list
            :param packed_bin: PackedBin类
            :param EPspace: 由一个extreme point所确定的space, Space类
            :param box: AlgorithmBox类
            :param direction: 0表示100方向，1表示200方向
        """
        new_packed_bin = copy.deepcopy(packed_bin)
        new_packed_bin.order += 1
        # if new_packed_bin.order == 63:
        #     print("Here!")
        # 根据方向确定box在x y z轴上的长度
        lx, ly, lz = utils.choose_box_direction_len(
            box.length, box.width, box.height, box_direction)
        packed_box = PackedBox(
            *EPspace.min_coord, lx, ly, lz, box, new_packed_bin.order, box_direction, 0)
        copy_box = AlgorithmBox.copy_algorithm_box(box=box, box_num=1)  # AlgorihtmBox类
        if new_packed_bin.packed_box_list:
            new_packed_bin.packed_box_list.append(packed_box)
            new_packed_bin.box_list.append(copy_box)
        else:
            new_packed_bin.packed_box_list = [packed_box]
            new_packed_bin.box_list = [copy_box]
        new_packed_bin.load_volume += lx * ly * lz
        new_packed_bin.load_weight += box.weight
        new_packed_bin.load_amount += box.amount

        """ 产生新的EPspace, 且根据新装载的box更新旧的EPspace(也要计算hold_surface); 计算waste_space; """
        # 去重
        new_packed_bin.space_obj.space_list = utils.del_repeat_space(new_packed_bin.space_obj.space_list)
        # 删除已占用的epspace
        new_packed_bin.space_obj.space_list.remove(EPspace)
        # 如果装载之前bin是空的，那么只需加入新的ep space
        if len(new_packed_bin.packed_box_list) == 1:
            updated_new_epspace_list, wasted_space_list = self.get_new_epspace(new_packed_bin)
            new_packed_bin.space_obj.space_list = updated_new_epspace_list
            wasted_space_v_sum = 0
        # 否则，更新新的和旧的ep space
        else:
            updated_new_epspace_list, wasted_space_list1 = self.get_new_epspace(new_packed_bin)
            # 新产生的ep space不会造成wasted_space
            # if wasted_space_list1:
            #     raise Exception("wasted_space_list1 should be empty.")
            # 利用最新装载的box更新packed_bin中已有的ep space
            updated_old_epspace_list, wasted_space_list2 = utils.update_epspace_by_box(
                packed_box_list=[new_packed_bin.packed_box_list[-1]], \
                epspace_list=new_packed_bin.space_obj.space_list)
            # 将所有space放到packed_bin.space_obj.space_list中
            new_packed_bin.space_obj.space_list = updated_old_epspace_list
            new_packed_bin.space_obj.space_list.extend(updated_new_epspace_list)

            #wasted_space_list = wasted_space_list1 + wasted_space_list2
            # 计算space体积和
            wasted_space_v_sum = 0  # utils.get_space_volume_sum(wasted_space_list)

        return new_packed_bin, wasted_space_v_sum 

    def get_new_epspace(self, new_packed_bin):
        """
            根据刚装载的box，向bin中加入该box产生的新的epspace
            :param new_packed_bin: 刚刚装载完一个新的box
        """
        new_epspace_list = []
        wasted_space_list = []
        packed_bin = new_packed_bin
        new_packed_box = packed_bin.packed_box_list[-1]
        epspace_list = packed_bin.space_obj.space_list
        box_lx = new_packed_box.lx
        box_ly = new_packed_box.ly
        box_lz = new_packed_box.lz
        box_min_coord = [new_packed_box.x, new_packed_box.y, new_packed_box.z]
        # 若packed_bin只装载了一个box,直接把三个extreme points所确定的三个epspace加入epspace_list
        if len(packed_bin.packed_box_list) == 1:
            if sum(box_min_coord) != 0:
                raise Exception("The sum of box_min_coord should be 0.")
            # 直接把三个extreme points所确定的三个epspace加入epspace_list
            space_x = Space(packed_bin.length-box_lx, packed_bin.width, \
                            packed_bin.height, \
                            min_coord=(box_min_coord[0]+box_lx, 0, 0), \
                            hold_surface=[])
            space_x.gen_by_which_box_order = new_packed_box.order
            space_y = Space(packed_bin.length, packed_bin.width-box_ly, \
                            packed_bin.height, \
                            min_coord=(0, box_min_coord[1]+box_ly, 0), \
                            hold_surface=[])
            space_y.gen_by_which_box_order = new_packed_box.order
            space_z_hold_area = Area(
                lx=box_lx, ly=box_ly, min_coord=[box_min_coord[0], box_min_coord[1]])
            space_z = Space(packed_bin.length, packed_bin.width, \
                            packed_bin.height-box_lz, \
                            min_coord=(0, 0, box_min_coord[2]+box_lz), \
                            hold_surface=[space_z_hold_area])
            space_z.gen_by_which_box_order = new_packed_box.order
            new_epspace_list.append(space_x)
            new_epspace_list.append(space_y)
            new_epspace_list.append(space_z)
            updated_new_epspace_list = new_epspace_list
        # 否则，遍历之前装载的每一个box以逐步确定新的ep space
        else:
            # 产生新ep space: BLH_x_space (backLeftHigh角沿x轴投影产生的space),BLH_y_space,
            # FLL_y_space (frontLeftLow), FLL_z_space, BRL_x_space(backRightLow), BRL_z_space
            hold_area = Area(  # blh_x_space和blh_y_space的初始hold_area
                lx=box_lx, ly=box_ly, min_coord=[box_min_coord[0], box_min_coord[1]])
            
            # 以blh点为最小坐标点确立一个space
            blh_min_coord = [box_min_coord[0], box_min_coord[1], box_min_coord[2]+box_lz]
            blh_space = Space(packed_bin.length - blh_min_coord[0], \
                                packed_bin.width - box_min_coord[1], \
                                packed_bin.height - box_lz - box_min_coord[2], \
                                min_coord=blh_min_coord, \
                                hold_surface=[hold_area])
            blh_space.gen_by_which_box_order = new_packed_box.order
            # 根据packed_box确定该ep space的所有支撑平面
            hold_surface_list = utils.get_hold_surface(blh_space, packed_bin.packed_box_list)
            if hold_surface_list != "Float":
                blh_space.hold_surface = hold_surface_list
                new_epspace_list.append(blh_space)

            # 第1个blh_x_space，只平行x轴投影确定最小坐标点
            x_coord = utils.find_coord(packed_bin.packed_box_list[:-1], \
                                       projecting_direction=0, \
                                       refer_coord=[box_min_coord[0], 
                                                    box_min_coord[1], 
                                                    box_min_coord[2]+box_lz])
            blh_x_min_coord=(x_coord, box_min_coord[1], box_min_coord[2]+box_lz)
            blh_x_space1 = Space(packed_bin.length - blh_x_min_coord[0], \
                                packed_bin.width - blh_x_min_coord[1], \
                                packed_bin.height - blh_x_min_coord[2], \
                                min_coord=blh_x_min_coord, \
                                hold_surface=[hold_area])
            blh_x_space1.gen_by_which_box_order = new_packed_box.order
            # 根据packed_box确定该ep space的所有支撑平面
            hold_surface_list = utils.get_hold_surface(blh_x_space1, packed_bin.packed_box_list)
            if hold_surface_list != "Float":
                blh_x_space1.hold_surface = hold_surface_list
                new_epspace_list.append(blh_x_space1)
            # 第2个blh_x_space，平行x投影后，继续平行y投影，确定最小坐标点
            y_coord = utils.find_coord(packed_bin.packed_box_list[:-1], \
                                       projecting_direction=1, \
                                       refer_coord=[x_coord, 
                                                    box_min_coord[1], 
                                                    box_min_coord[2]+box_lz])
            blh_x_min_coord=(x_coord, y_coord, box_min_coord[2]+box_lz)
            blh_x_space2 = Space(packed_bin.length - blh_x_min_coord[0], \
                                packed_bin.width - blh_x_min_coord[1], \
                                packed_bin.height - blh_x_min_coord[2], \
                                min_coord=blh_x_min_coord, \
                                hold_surface=[hold_area])
            blh_x_space2.gen_by_which_box_order = new_packed_box.order
            # 根据packed_box确定该ep space的所有支撑平面
            hold_surface_list = utils.get_hold_surface(blh_x_space2, packed_bin.packed_box_list)
            if hold_surface_list != "Float":
                blh_x_space2.hold_surface = hold_surface_list
                new_epspace_list.append(blh_x_space2)

            # 第1个blh_y_space，只平行y轴投影确立最小坐标点
            y_coord = utils.find_coord(packed_bin.packed_box_list[:-1], \
                                       projecting_direction=1, \
                                       refer_coord=[box_min_coord[0], 
                                                    box_min_coord[1], 
                                                    box_min_coord[2]+box_lz])
            blh_y_min_coord=(box_min_coord[0], y_coord, box_min_coord[2]+box_lz)
            if blh_y_min_coord != blh_x_min_coord:
                blh_y_space1 = Space(packed_bin.length - blh_y_min_coord[0], \
                                    packed_bin.width - blh_y_min_coord[1], \
                                    packed_bin.height - blh_y_min_coord[2], \
                                    min_coord=blh_y_min_coord, \
                                    hold_surface=[hold_area])
                blh_y_space1.gen_by_which_box_order = new_packed_box.order
                # 根据packed_box确定该ep space的所有支撑平面
                hold_surface_list = utils.get_hold_surface(blh_y_space1, packed_bin.packed_box_list)
                if hold_surface_list != "Float":
                    blh_y_space1.hold_surface = hold_surface_list
                    new_epspace_list.append(blh_y_space1)
            # 第2个blh_y_space，先平行y轴投影再平行x轴投影确定最小坐标点
            y_coord = utils.find_coord(packed_bin.packed_box_list[:-1], \
                                       projecting_direction=1, \
                                       refer_coord=[box_min_coord[0], 
                                                    box_min_coord[1], 
                                                    box_min_coord[2]+box_lz])
            x_coord = utils.find_coord(packed_bin.packed_box_list[:-1], \
                                       projecting_direction=0, \
                                       refer_coord=[box_min_coord[0], 
                                                    y_coord,
                                                    box_min_coord[2]+box_lz])
            blh_y_min_coord=(x_coord, y_coord, box_min_coord[2]+box_lz)
            if blh_y_min_coord != blh_x_min_coord:
                blh_y_space2 = Space(packed_bin.length - blh_y_min_coord[0], \
                                    packed_bin.width - blh_y_min_coord[1], \
                                    packed_bin.height - blh_y_min_coord[2], \
                                    min_coord=blh_y_min_coord, \
                                    hold_surface=[hold_area])
                blh_y_space2.gen_by_which_box_order = new_packed_box.order
                # 根据packed_box确定该ep space的所有支撑平面
                hold_surface_list = utils.get_hold_surface(blh_y_space2, packed_bin.packed_box_list)
                if hold_surface_list != "Float":
                    blh_y_space2.hold_surface = hold_surface_list
                    new_epspace_list.append(blh_y_space2)

            # fll_y_space，先平行y轴投影，再平行z轴投影确定最小坐标点
            y_coord = utils.find_coord(packed_bin.packed_box_list[:-1], \
                                       projecting_direction=1, \
                                       refer_coord=[box_min_coord[0] + box_lx,
                                                    box_min_coord[1],
                                                    box_min_coord[2]])
            z_coord = utils.find_coord(packed_bin.packed_box_list[:-1], \
                                       projecting_direction=2, \
                                       refer_coord=[box_min_coord[0] + box_lx,
                                                    y_coord,
                                                    box_min_coord[2]])
            fll_y_min_coord=(box_min_coord[0]+box_lx, y_coord, z_coord)
            fll_y_space = Space(packed_bin.length - fll_y_min_coord[0], \
                                packed_bin.width - fll_y_min_coord[1], \
                                packed_bin.height - fll_y_min_coord[2], \
                                min_coord=fll_y_min_coord, \
                                hold_surface="TBC")  # To Be Confirmed,注意悬空情况
            fll_y_space.gen_by_which_box_order = new_packed_box.order
            # 根据packed_box确定该ep space的所有支撑平面
            hold_surface_list = utils.get_hold_surface(fll_y_space, packed_bin.packed_box_list[:-1])
            if hold_surface_list != "Float":
                fll_y_space.hold_surface = hold_surface_list
                new_epspace_list.append(fll_y_space)

            # 第一个fll_z_space，先平行z轴投影，再平行y轴投影确定最小坐标点
            z_coord = utils.find_coord(packed_bin.packed_box_list[:-1], \
                                       projecting_direction=2, \
                                       refer_coord=[box_min_coord[0] + box_lx,
                                                    box_min_coord[1],
                                                    box_min_coord[2]])
            y_coord = utils.find_coord(packed_bin.packed_box_list[:-1], \
                                       projecting_direction=1, \
                                       refer_coord=[box_min_coord[0] + box_lx,
                                                    box_min_coord[1],
                                                    z_coord])
            fll_z_min_coord=(box_min_coord[0]+box_lx, y_coord, z_coord)
            if fll_z_min_coord != fll_y_min_coord:
                fll_z_space1 = Space(packed_bin.length - fll_z_min_coord[0], \
                                    packed_bin.width - fll_z_min_coord[1], \
                                    packed_bin.height - fll_z_min_coord[2], \
                                    min_coord=fll_z_min_coord, \
                                    hold_surface="TBC")  # To Be Confirmed,注意悬空情况
                fll_z_space1.gen_by_which_box_order = new_packed_box.order
                # 根据packed_box确定该ep space的所有支撑平面
                hold_surface_list = utils.get_hold_surface(fll_z_space1, packed_bin.packed_box_list[:-1])
                if hold_surface_list != "Float":
                    fll_z_space1.hold_surface = hold_surface_list
                    new_epspace_list.append(fll_z_space1)
            # 第二个fll_z_space，只平行于z轴投影
            fll_z_min_coord=(box_min_coord[0]+box_lx, box_min_coord[1], z_coord)
            if fll_z_min_coord != fll_y_min_coord:
                fll_z_space2 = Space(packed_bin.length - fll_z_min_coord[0], \
                                    packed_bin.width - fll_z_min_coord[1], \
                                    packed_bin.height - fll_z_min_coord[2], \
                                    min_coord=fll_z_min_coord, \
                                    hold_surface="TBC")  # To Be Confirmed,注意悬空情况
                fll_z_space2.gen_by_which_box_order = new_packed_box.order
                # 根据packed_box确定该ep space的所有支撑平面
                hold_surface_list = utils.get_hold_surface(fll_z_space2, packed_bin.packed_box_list[:-1])
                if hold_surface_list != "Float":
                    fll_z_space2.hold_surface = hold_surface_list
                    new_epspace_list.append(fll_z_space2)

            # brl_x_space，先平行x轴投影再平行z轴投影
            x_coord = utils.find_coord(packed_bin.packed_box_list[:-1], \
                                       projecting_direction=0, \
                                       refer_coord=[box_min_coord[0], 
                                                    box_min_coord[1] + box_ly, 
                                                    box_min_coord[2]])
            z_coord = utils.find_coord(packed_bin.packed_box_list[:-1], \
                                       projecting_direction=2, \
                                       refer_coord=[x_coord, 
                                                    box_min_coord[1] + box_ly, 
                                                    box_min_coord[2]])
            brl_x_min_coord=(x_coord, box_min_coord[1]+box_ly, z_coord)
            brl_x_space = Space(packed_bin.length - brl_x_min_coord[0], \
                                packed_bin.width - brl_x_min_coord[1],\
                                packed_bin.height - brl_x_min_coord[2], \
                                min_coord=brl_x_min_coord, \
                                hold_surface="TBC")  # To Be Confirmed,注意悬空情况
            brl_x_space.gen_by_which_box_order = new_packed_box.order
            # 根据packed_box确定该ep space的所有支撑平面
            hold_surface_list = utils.get_hold_surface(brl_x_space, packed_bin.packed_box_list[:-1])
            if hold_surface_list != "Float":
                brl_x_space.hold_surface = hold_surface_list
                new_epspace_list.append(brl_x_space)

            # 第一个brl_z_space，先平行z轴投影再平行x轴投影
            z_coord = utils.find_coord(packed_bin.packed_box_list[:-1], \
                                       projecting_direction=2, \
                                       refer_coord=[box_min_coord[0],
                                                    box_min_coord[1] + box_ly,
                                                    box_min_coord[2]])
            x_coord = utils.find_coord(packed_bin.packed_box_list[:-1], \
                                       projecting_direction=0, \
                                       refer_coord=[box_min_coord[0],
                                                    box_min_coord[1] + box_ly,
                                                    z_coord])
            brl_z_min_coord=(x_coord, box_min_coord[1]+box_ly, z_coord)
            if brl_z_min_coord != brl_x_min_coord:
                brl_z_space1 = Space(packed_bin.length - brl_z_min_coord[0], \
                                    packed_bin.width - brl_z_min_coord[1],\
                                    packed_bin.height - brl_z_min_coord[2], \
                                    min_coord=brl_z_min_coord, \
                                    hold_surface="TBC")  # To Be Confirmed,注意悬空情况
                brl_z_space1.gen_by_which_box_order = new_packed_box.order
                # 根据packed_box确定该ep space的所有支撑平面
                hold_surface_list = utils.get_hold_surface(brl_z_space1, packed_bin.packed_box_list[:-1])
                if hold_surface_list != "Float":
                    brl_z_space1.hold_surface = hold_surface_list
                    new_epspace_list.append(brl_z_space1)
            # 第2个brl_z_space，只平行z轴投影
            brl_z_min_coord=(box_min_coord[0], box_min_coord[1]+box_ly, z_coord)
            if brl_z_min_coord != brl_x_min_coord:
                brl_z_space2 = Space(packed_bin.length - brl_z_min_coord[0], \
                                    packed_bin.width - brl_z_min_coord[1],\
                                    packed_bin.height - brl_z_min_coord[2], \
                                    min_coord=brl_z_min_coord, \
                                    hold_surface="TBC")  # To Be Confirmed,注意悬空情况
                brl_z_space2.gen_by_which_box_order = new_packed_box.order
                # 根据packed_box确定该ep space的所有支撑平面
                hold_surface_list = utils.get_hold_surface(brl_z_space2, packed_bin.packed_box_list[:-1])
                if hold_surface_list != "Float":
                    brl_z_space2.hold_surface = hold_surface_list
                    new_epspace_list.append(brl_z_space2)
            
            # 利用之前已装载的且与space有重合的boxes更新新产生的ep space
            updated_new_epspace_list, wasted_space_list = utils.update_epspace_by_box(
                packed_box_list=packed_bin.packed_box_list[:-1], epspace_list=new_epspace_list)
        return updated_new_epspace_list, wasted_space_list

    def transfer_box_for_low_loading_bin(self, pack_bin_list):
        """ 如果最后一辆车的装载率很小（如<0.1）, 看能否装到其他车上 """
        backup_pack_bin_list = copy.deepcopy(pack_bin_list)
        new_pack_bin_list = []
        pack_bin_list.sort(key=lambda bin: bin.ratio, reverse=False)  # 升序
        last_bin = pack_bin_list[0]
        if last_bin.ratio < 0.1:
            i = 1
            box_list = last_bin.box_list
            box_by_platform_2dList = utils.classify_box_by_platform(box_list)
            while box_list and (i < len(pack_bin_list)):
                pack_bins, remaining_box_Dict = self.pack(box_list, box_by_platform_2dList, pack_method=2, last_bin=pack_bin_list[i], p=1)
                box_list = remaining_box_Dict["box_list"]
                box_by_platform_2dList = remaining_box_Dict["box_by_platform_2DList"]
                i += 1
                if pack_bins:
                    new_pack_bin_list.append(pack_bins[0])
                else:
                    raise Exception("Packing fails.")
            if box_list:  # 若没有装载完
                pack_bin_list = backup_pack_bin_list
            else:
                if i != len(pack_bin_list):
                    pack_bin_list = new_pack_bin_list + pack_bin_list[i:]
                else:
                    pack_bin_list = new_pack_bin_list
        else:
            pack_bin_list = backup_pack_bin_list
        return pack_bin_list

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
        for truck in self.trucks_listdict:
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
                pack_bins, remaining_box_Dict = self.pack(box_list, box_by_platform_2dList, pack_method=1, last_bin=None, p=0)
                if remaining_box_Dict["box_list"]:
                    # 不使用该bin进行替换
                    flag += 1
                    continue
                else:
                    #进行替换
                    new_packed_bin_list.append(pack_bins[0])
                    print("Replacement succeeded.")
                    break
            # 若遍历完candi_replace_bin仍没有合适的替换，则使用原bin
            if flag == len(candi_replace_bin):
                new_packed_bin_list.append(packed_bin)
                print("Replacement failed.")
        self.bin = self.revs_bin
        return new_packed_bin_list

    def simple_pack(self, space, spu_ids):
        """ 单独接收一块长方体空间，和要装载的boxes，进行装载，然后计算这个空间的装载率 """
        return

    def get_a_decodedsol(self, sol_list):  # sol_list就是解码前的route
        """
            sol_list是platform的索引list，如果共有n个platforms，
            则sol_list是一个1-n的排列，n表示在输入数据的"platformDtoList"中的第n个platform.
            该方法既包含direct routes,也包含SDVRP routes.
        """
        truck_list = []
        truck_list_direct_trip = self.direct_route()
        truck_list_SDVRP = self.decode_sol_SDVRP(sol_list)
        for truck in truck_list_direct_trip:
            truck_list.append(truck)
        for truck in truck_list_SDVRP:
            truck_list.append(truck)
        return truck_list

    def direct_route(self):
        direct_route_truck_listdict = []  # 存储该order中所有的direct route
        for i in range(self.platform_num):
            platformCode = self.allplatformlist_ListDict[i]["platformCode"]
            FLP_of_this_platform_dict =\
                self.pack_patterns_dictlistdictlistAlgorithmBox[platformCode][0]
            FLP_num_of_this_platform = FLP_of_this_platform_dict["1C-FLPNum"]

            if FLP_num_of_this_platform > 0:
                packed_bin_list = self._pack_FLP(platformCode, FLP_of_this_platform_dict)  # 待实现
            
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
                    spu_list.sort(key=lambda box: box['order'])  # Han:按‘order’字段进行升序排序
                    truck["spuArray"] = spu_list  # Han:truck['spuArray']存储该卡车所有装载的boxes的信息
                    truck["platformArray"] = [platformCode]  # direct route. Huawei文件pack.py的_gen_res()中提供的方法可参考
                    direct_route_truck_listdict.append(truck)
        return direct_route_truck_listdict

    def decode_sol_SDVRP(self, sol_list):
        """
        该方法解码split delivery route
        注意将解码后的解添加进self.res["solutionArray"]时是哪一级list
        :param sol_list: 一个platform索引(在输入文件中的索引，代码是从1开始)的permutation,
                        e.g., [1,2,3,4,5]表示input文件中(即self.data)["algorithmBaseParamDto"]["platformDtoList"]中
                        第1,2,3,4,5个platform, 而第1个platform的platformCode可能是platform06(而不是platform01)
        """
        truck_list = []
        sub_sol_listlist = self.get_sub_sol_according2_mustfirst_points(sol_list)
        for sub_sol_list in sub_sol_listlist:
            sub_truck_list = self.decode_sub_sol(sub_sol_list)
            for truck in sub_truck_list:
                truck_list.append(truck)
        return truck_list

    def get_sub_sol_according2_mustfirst_points(self, sol_list):
        """
        倒序检查sol_list中的bondedPoints,按bondedPoints对sol_list进行切分;
        i.e.,从倒数第一个bondedPoint到最后的point是一个sub_sol_list,
        从倒数第二个bondedPoint到倒数第一个bondedPoint(不含)是一个sub_sol_list.
        @return: sub_sol_listlist: 二维list，每行存储一个platform序列
        """
        route_2dlist = []
        sub_sol_listlist = []
        end_pointer = len(sol_list)
        platform_listdict = self.allplatformlist_ListDict
        for platform_index in reversed(sol_list):
            if platform_listdict[int(platform_index) - 1]["mustFirst"]:
                start_pointer = sol_list.index(platform_index)
                sub_sol_list = sol_list[start_pointer:end_pointer]
                sub_sol_listlist.append(sub_sol_list)
                end_pointer = start_pointer
        if platform_listdict[sol_list[0]-1]["mustFirst"] is not True:
            sub_sol_listlist.append(sol_list[0:end_pointer])
        
        return sub_sol_listlist

    def decode_sub_sol(self, sub_sol_list):
        """
            对一条路径，返回可行的车辆装载方案，
            可能会不止一辆车(根据车辆装载的box可以得到该车辆的路径).
            :param sub_sol_list: 一个以bonded warehouse开头的排列
            :return truck_list: 每个元素都是dictionary类型，包含一个truck的全部装载信息
        """
        truck_list = []
        route = utils.trans_sol_2_route(sub_sol_list, self.platformCode_list)
        packed_bin_list = self._pack_by_platform(route)  # 待实现
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
                truck["maxLoad"] = packed_bin.max_weight # （Han:最大装载重量）
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
        return truck_list

    def _gen_box_list(self, spu_ids=None):  
        raw_box_list = self.data["boxes"]
        box_map_list = []
        all_box_volume_sum = 0
        all_box_weight_sum = 0
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
            all_box_volume_sum += box_list[-1].volume
            all_box_weight_sum += box_list[-1].weight
        return box_list, all_box_volume_sum, all_box_weight_sum

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

    def _gen_output_file(self, truck_list, file_name=None, output_path=None):
        res = {"estimateCode": self.data["estimateCode"], "solutionArray": []}
        res["solutionArray"].append(truck_list)
        if output_path is not None:
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            output_file = os.path.join(output_path, file_name)
            with open(
                    output_file, 'w', encoding='utf-8', errors='ignore') as f:
                json.dump(res, f, ensure_ascii=False, indent=4)  # 将解保存为json文件


def main(argv):
    input_dir = argv[1]  # get input from terminal
    output_dir = argv[2]
    # input_dir = "E:\\VSCodeSpace\\HuaweiCompetitionSpace\\mycodes\\3L-SDVRP\\data\\inputs"
    # output_dir = "E:\\VSCodeSpace\\HuaweiCompetitionSpace\\mycodes\\3L-SDVRP\\data\\outputs"
    time_record = {}
    t1 = time.process_time()
    for file_name in os.listdir(input_dir):
        print("The order ", file_name, " is processing...")
        t2 = time.process_time()
        input_path = os.path.join(input_dir, file_name)
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            message_str = f.read()
        order = Order(message_str)
        route = list(range(1, order.platform_num+1))
        # truck_code = "R110"  # "CT03" "CT10"  #"R110"
        pack_obj = Pack(order)
        truck_list = pack_obj.decode_sol_SDVRP(route)
        pack_obj._gen_output_file(truck_list, file_name, output_dir)
        t3 = time.process_time()
        time_record[file_name] = t3 - t2
        print("The order ", file_name, " is done.", "Processing time: ", t3-t2)
        output_file = os.path.join(output_dir, "TimeRecords.json")
        with open(
                output_file, 'w', encoding='utf-8', errors='ignore') as f:
            json.dump(time_record, f, ensure_ascii=False, indent=4)
    t4 = time.process_time()
    total_time = t4 - t1
    time_record["TotalTime"] = total_time
    output_file = os.path.join(output_dir, "TimeRecords.json")
    with open(
            output_file, 'w', encoding='utf-8', errors='ignore') as f:
        json.dump(time_record, f, ensure_ascii=False, indent=4)  # 将时间保存为json文件
    print("Total time: ", total_time)


if __name__ == "__main__":
    main(sys.argv)
