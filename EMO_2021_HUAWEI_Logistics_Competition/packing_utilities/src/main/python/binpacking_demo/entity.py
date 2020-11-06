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


class Space:
    def __init__(self,
                 lx=0,
                 ly=0,
                 lz=0,
                 min_coord=(0, 0, 0),
                 max_coord=None,
                 trans_space=None,
                 hold_surface=None):
        """
        :param lx=0: x轴方向长度
        :param ly=0: y轴方向长度
        :param lz=0: z轴方向长度
        :param min_coord=(0, 0, 0): 空间的参考点坐标，前面左下角点坐标值
        :param max_coord=None: 空间的最大坐标点
        :param trans_space=None: 可以被转移的空间
        :param hold_surface=None: 空间的支撑平面
        """
        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.min_coord = list(min_coord)
        if max_coord:
            self.max_coord = max_coord
        else:
            self.max_coord = [lx + min_coord[0],
                              ly + min_coord[1],
                              lz + min_coord[2]]
        self.trans_space = trans_space
        self.vol = self.lx * self.ly * self.lz
        if hold_surface:
            self.hold_surface = hold_surface
        else:
            self.hold_surface = []

    @classmethod
    def by_length(cls,
                  lx,
                  ly,
                  lz,
                  min_coord=(0, 0, 0),
                  trans_space=None,
                  hold_surface=None):
        """
        docstring here
            :param lx: 空间x方向长度
            :param ly: 空间y方向长度
            :param lz: 空间z方向长度
            :param min_coord：空间最小坐标
            :param trans_space: 空间中可以被用来转换的空间
        """
        max_coord = [lx + min_coord[0], ly + min_coord[1], lz + min_coord[2]]
        return cls(lx, ly, lz, min_coord, max_coord, trans_space, hold_surface)

    @classmethod
    def by_coordinate(cls, min_coord, max_coord, hold_surface=None):
        """
        根据空间的最小坐标和最大坐标生成空间
            :param min_coord: 区域的最小坐标
            :param max_coord: 区域的最大坐标
            :param hold_surface: 支撑平面
        """
        size = [
            max_coord[0] - min_coord[0],
            max_coord[1] - min_coord[1],
            max_coord[2] - min_coord[2],
        ]
        return cls(
            *size,
            min_coord=min_coord,
            max_coord=max_coord,
            hold_surface=hold_surface)

    def set_trans_space(self, trans_space):
        """
        设置可以被转移的空间

        :param trans_space:可以被转移的空间
        :return:
        """
        self.trans_space = trans_space

    def __eq__(self, other):
        return self.lx == other.lx and self.ly == other.ly and \
               self.lz == other.lz and self.min_coord == other.min_coord


class SimpleBlock:
    def __init__(self,
                 nx,
                 ny,
                 nz,
                 item_size,
                 weight=0,
                 amount=0,
                 hold_surface=None,
                 bottom_surface=None,
                 packed_box_list=None,
                 platform=None):
        """
        docstring here
            :param nx: x方向箱子个数
            :param ny: y方向箱子个数
            :param nz: z方向箱子个数
            :param item_size: 组成块的物品的类型
            :param weight=0: 块的总重量
            :param amount=0: 块的总金额
            :param hold_surface=None: 块的支撑平面
            :param bottom_surface=None: 块的底部平面
            :param packed_box_list=None: 组成该块的箱子集合
            :param platform: 块所在的提货点
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.box_num = nx * ny * nz
        self.item_size = item_size
        self.weight = weight
        self.amount = amount
        self.hold_surface = hold_surface
        self.bottom_surface = bottom_surface
        self.packed_box_list = packed_box_list
        self.vol = item_size[0] * item_size[1] * item_size[2] * nx * ny * nz
        box_lx, box_ly, box_lz = item_size
        self.lx = box_lx * nx
        self.ly = box_ly * ny
        self.lz = box_lz * nz
        self.platform = platform


class Box:
    def __init__(self,
                 box_type,
                 length,
                 width,
                 height,
                 weight,
                 all_directions,
                 platform,
                 max_layer,
                 max_weight,
                 order=-1,
                 amount=0,
                 box_id=-1):
        """
            箱子父类
            :param box_type: 箱子的种类
            :param length: 箱子的长
            :param width: 箱子的宽
            :param height: 箱子的高
            :param weight: 箱子的重量
            :param all_directions: 箱子所有可能的选装方向
            :param platform: 箱子所在平台
            :param max_layer: 箱子的最大堆叠层数（底面积相同时）
            :param max_weight: 箱子的最大承载重量（底面积不同时）
            :param amount=-1: 箱子的金额
            :param order=-1: 箱子的摆放顺序
            :param box_id=-1: 箱子的唯一标识id
        """
        self.box_type = box_type
        self.length = length
        self.width = width
        self.height = height
        self.all_directions = all_directions
        self.weight = weight
        self.platform = platform
        self.max_layer = max_layer
        self.max_weight = max_weight
        self.amount = amount
        self.order = order
        self.box_id = box_id


class AlgorithmBox(Box):
    def __init__(self,
                 box_type,
                 length,
                 width,
                 height,
                 weight,
                 all_directions,
                 box_num,
                 platform,
                 max_layer,
                 max_weight,
                 is_cylinder=False,
                 order=-1,
                 amount=0,
                 box_id=-1,
                 direction=-1):
        """
            算法的输入箱子类
            :param box_type: 箱子的种类
            :param length: 箱子的长
            :param width: 箱子的宽
            :param height: 箱子的高
            :param weight: 箱子的重量
            :param all_directions: 箱子所有可能的选装方向
            :param box_num: 箱子的数量
            :param platform: 箱子所在平台
            :param max_layer: 箱子的最大堆叠层数（底面积相同时）
            :param max_weight: 箱子的最大承载重量（底面积不同时）
            :param amount=-1: 箱子的金额
            :param order=-1: 箱子的摆放顺序
            :param box_id=-1: 箱子的唯一标识id
            :param direction=-1: 箱子的方向（不一定实际使用）
        """
        super(AlgorithmBox, self).__init__(
            box_type, length, width, height, weight, all_directions, platform,
            max_layer, max_weight, order, amount, box_id)
        self.is_cylinder = is_cylinder
        self.box_num = box_num
        self.direction = direction

    @classmethod
    def copy_algorithm_box(cls, box, box_num):
        return cls(box.box_type, box.length, box.width, box.height, box.weight,
                   box.all_directions, box_num, box.platform, box.max_layer,
                   box.max_weight, box.is_cylinder, box.order, box.amount,
                   box.box_id, box.direction)


class PackedBox(Box):
    def __init__(self, x, y, z, lx, ly, lz, box, order, direction,
                 cylinder_state):
        """
            算法的输出箱子类
            :param x: x坐标
            :param y: y坐标
            :param z: z坐标
            :param lx: x方向长度
            :param ly: y方向长度
            :param lz: z方向长度
            :param box: 一个输入箱子类的实例
            :param order: 箱子的摆放顺序
            :param direction: 箱子的方向
            :param cylinder_state: 0为立方体，1-3为对应圆柱体状态
        """
        super(PackedBox, self).__init__(
            box.box_type, box.length, box.width, box.height, box.weight,
            box.all_directions, box.platform, box.max_layer, box.max_weight,
            order, box.amount, box.box_id)
        self.x = x
        self.y = y
        self.z = z
        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.direction = direction
        self.cylinder_state = cylinder_state

    def change_ordinate(self, packed_bin):
        # 转换坐标系中心为集装箱中心
        self.x -= packed_bin.length / 2
        self.y -= packed_bin.width / 2
        self.z -= packed_bin.height / 2
        # 转换箱子坐标为箱子中心
        self.x += self.lx / 2
        self.y += self.ly / 2
        self.z += self.lz / 2
        # 转换箱子坐标轴
        self.x, self.y, self.z = self.y, self.z, self.x
        # 转换箱子方向
        if self.direction == 0:
            self.direction = 100
        elif self.direction == 1:
            self.direction = 200


class CoordinateItem:
    def __init__(self, box, coord, direction=None):
        self.box = box
        self.min_coord = coord
        if type(box) == AlgorithmBox:
            size = utils.choose_box_direction_len(box.length, box.width,
                                                  box.height, direction)
        else:
            size = [box.lx, box.ly, box.lz]
        self.max_coord = [
            coord[0] + size[0], coord[1] + size[1], coord[2] + size[2]
        ]


class Bin:
    def __init__(self,
                 bin_type,
                 length,
                 width,
                 height,
                 volume,
                 door_height=None,
                 max_weight=-1,
                 max_amount=-1,
                 weight=0,
                 truck_type_id=None,
                 truck_type_code=None):
        """
            算法输入卡车类
            :param bin_type: 容器类型
            :param length: 容器长
            :param width: 容器宽
            :param height: 容器高
            :param door_height: 门高
            :param volume: 容器体积
            :param max_weight: 容器的最大承载重量
            :param max_amount=-1: 容器装载货物的最大金额
            :param weight=0: 容器重量
        """
        self.bin_type = bin_type
        self.length = length
        self.width = width
        self.height = height
        self.volume = volume
        if door_height is None:
            self.door_height = height
        else:
            self.door_height = door_height
        self.max_weight = max_weight
        self.max_amount = max_amount
        self.weight = weight
        self.truck_type_id = truck_type_id
        self.truck_type_code = truck_type_code


class PackedBin(Bin):
    def __init__(self,
                 bin_type,
                 length,
                 width,
                 height,
                 volume,
                 space_obj=None,
                 order=0,
                 max_weight=-1,
                 max_amount=-1,
                 load_weight=0,
                 load_amount=0,
                 load_volume=0,
                 weight=0,
                 packed_box_list=None,
                 box_list=None,
                 ratio=0,
                 door_height=None,
                 truck_type_id=None,
                 truck_type_code=None):
        """
            算法输出卡车类
            :param bin_type: 容器类型
            :param length: 容器长
            :param width: 容器宽
            :param height: 容器高
            :param volume: 容器体积
            :param space_obj: 使用的空间算法类
            :param order: 箱子中目前放置的箱子序号
            :param max_weight: 容器的最大承载重量
            :param max_amount=-1: 容器装载货物的最大金额
            :param load_weight=0: 容器装载货物重量
            :param load_amount=0: 容器装载货物金额
            :param load_volume=0: 容器装载货物体积
            :param weight=0: 容器重量
            :param packed_box_list=None: 容器装载的箱子集合，存储的是输入箱子类
            :param box_list=None: 容器装载的箱子集合，存储的是输出箱子类
            :param ratio=0: 卡车的实装率
        """
        super(PackedBin, self).__init__(
            bin_type,
            length,
            width,
            height,
            volume,
            door_height,
            max_weight,
            max_amount,
            weight,
            truck_type_id,
            truck_type_code
        )
        self.space_obj = space_obj
        self.order = order
        self.load_weight = load_weight
        self.load_amount = load_amount
        self.load_volume = load_volume
        if packed_box_list is None:
            self.packed_box_list = []
        else:
            self.packed_box_list = packed_box_list
        if box_list is None:
            self.box_list = []
        else:
            self.box_list = box_list
        self.ratio = ratio
        self.surplus_length = self._gen_surplus_length()

    def _gen_surplus_length(self):
        max_x = 0
        if self.packed_box_list:
            for packed_box in self.packed_box_list:
                max_x = max(packed_box.x + packed_box.lx, max_x)
        return self.length - max_x

    @classmethod
    def create_by_bin(cls, bin_obj):
        return cls(
            bin_obj.bin_type,
            bin_obj.length,
            bin_obj.width,
            bin_obj.height,
            bin_obj.volume,
            max_weight=bin_obj.max_weight,
            max_amount=bin_obj.max_amount,
            weight=bin_obj.weight,
            door_height=bin_obj.door_height,
            truck_type_id=bin_obj.truck_type_id,
            truck_type_code=bin_obj.truck_type_code
        )


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
        self.max_coord = max_coord
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
