#!/usr/bin/env python
# coding=UTF-8

import copy
from entity import Space, Area
import general_utils as utils
import constrains


def is_overlap(min_coord1, max_coord1, min_coord2, max_coord2):
    """
        判断两个空间是否重合，若重合返回True，反之返回False
        :param min_coord1: 第一个区域的最小坐标
        :param max_coord1: 第一个区域的最大坐标
        :param min_coord2: 第二个区域的最小坐标
        :param max_coord2: 第二个区域的最大坐标
    """
    length = min(len(min_coord1), len(min_coord2))
    for i in range(length):
        if min_coord1[i] >= max_coord2[i] or max_coord1[i] <= min_coord2[i]:
            return False
    return True

class SimpleSpace:
    def __init__(self, space_list):
        self.space_list = space_list

    def transfer_space(self, space):
        # begin: han add
        for extnsn_s in self.space_list:
            if extnsn_s.min_coord[2] == space.min_coord[2] and \
                extnsn_s.min_coord[0] <= space.max_coord[0] and \
                extnsn_s.max_coord[1] >= space.max_coord[1] and\
                extnsn_s.min_coord[1] <= space.min_coord[1]:
                space.max_coord[0] = extnsn_s.max_coord[0]
                space.lx = space.max_coord[0] - space.min_coord[0]
                extnsn_s.max_coord[1] = space.min_coord[1]
                extnsn_s.ly = extnsn_s.max_coord[1] - extnsn_s.min_coord[1]
                if space.lx > 0 and space.ly > 0:
                    self.space_list.append(space)
                if extnsn_s.lx == 0 or extnsn_s.ly == 0:
                    self.space_list.remove(extnsn_s)
                return None
            if extnsn_s.min_coord[2] == space.min_coord[2] and \
                extnsn_s.min_coord[0] <= space.min_coord[0] and \
                extnsn_s.max_coord[0] >= space.max_coord[0] and \
                extnsn_s.min_coord[1] <= space.max_coord[1]:
                space.max_coord[1] = extnsn_s.max_coord[1]
                space.ly = space.max_coord[1] - space.min_coord[1]
                extnsn_s.max_coord[0] = space.min_coord[0]
                extnsn_s.lx = extnsn_s.max_coord[0] - extnsn_s.min_coord[0]
                if space.lx > 0 and space.ly > 0:
                    self.space_list.append(space)
                if extnsn_s.lx == 0 or extnsn_s.ly == 0:
                    self.space_list.remove(extnsn_s)
                return None
        # end: han add

    def update_space(self, packed_item, space):
        """
            装载新物品后更新空间集合

            :param packed_item: 被装载的物品
            :param space: 装载的空间
        """
        block = packed_item.box
        container_list = []
        mx = space.lx - block.lx
        my = space.ly - block.ly
        coord_x_space = copy.copy(space.min_coord)
        coord_x_space[0] += block.lx
        coord_y_space = copy.copy(space.min_coord)
        coord_y_space[1] += block.ly
        coord_z_space = copy.copy(space.min_coord)
        coord_z_space[2] += block.lz
        coord_trans_space = copy.copy(space.min_coord)
        coord_trans_space[0] += block.lx
        coord_trans_space[1] += block.ly
        space_trans = Space.by_length(mx, my, space.lz, coord_trans_space)
        if mx > my:
            container_x = Space.by_length(
                mx,
                space.ly,
                space.lz,
                coord_x_space,
                space_trans,
                hold_surface=space.hold_surface)
            container_y = Space.by_length(
                block.lx,
                my,
                space.lz,
                coord_y_space,
                hold_surface=space.hold_surface)
            if my > 0:
                container_list.append(container_y)
            if mx > 0:
                container_list.append(container_x)
        else:
            container_x = Space(
                mx,
                block.ly,
                space.lz,
                coord_x_space,
                hold_surface=space.hold_surface)
            container_y = Space(
                space.lx,
                my,
                space.lz,
                coord_y_space,
                trans_space=space_trans,
                hold_surface=space.hold_surface)
            if mx > 0:
                container_list.append(container_x)
            if my > 0:
                container_list.append(container_y)
        if space.lz - block.lz > 0:
            container_list.append(
                Space.by_length(
                    block.lx,
                    block.ly,
                    space.lz - block.lz,
                    coord_z_space,
                    hold_surface=block.hold_surface))
        self.space_list.extend(container_list)
        return self.space_list

    def get_new_space(self, packed_item, space, packed_box_list=None, bin_width=None):
        """
            han add
            装载新物品后产生新空间
            :param packed_item: 被装载的物品
            :param space: 装载的空间
        """
        block = packed_item.box
        refer_boxes_left = []
        if packed_box_list and space.min_coord[2] == 0:
            # 用于coord_x_space向左扩展 (即最小坐标点平行于y轴向左延伸)
            refer_boxes_left = [b for b in packed_box_list if b.x<=(space.min_coord[0]+block.lx) and (b.x+b.lx)>(space.min_coord[0]+block.lx) and (b.y+b.ly)<=space.min_coord[1]]
            refer_boxes_left.sort(key=lambda b: (b.y+b.ly), reverse=True)
            # 用于coord_x_space向右扩展 (即最大坐标点平行于y轴向右延伸)
            refer_boxes_right = [b for b in packed_box_list if b.x<=(space.min_coord[0]+block.lx) and (b.x+b.lx)>(space.min_coord[0]+block.lx) and b.y>=(space.min_coord[1]+block.ly)]
            refer_boxes_right.sort(key=lambda b: b.y, reverse=False)
        if packed_box_list and space.min_coord[2] == 0:
            container_list = []
            mx = space.lx - block.lx
            my = space.ly - block.ly
            coord_x_space = copy.copy(space.min_coord)
            coord_x_space[0] += block.lx
            if refer_boxes_left:
                coord_x_space[1] = refer_boxes_left[0].y + refer_boxes_left[0].ly
                # 检查x方向上(前方)有没有阻挡
                refer_boxes_front = [b for b in packed_box_list if b.x>(space.min_coord[0]+block.lx) and (b.y+b.ly)>coord_x_space[1]]
                if refer_boxes_front:
                    coord_x_space[1] = space.min_coord[1]
            else:
                coord_x_space[1] = 0
            if refer_boxes_right:
                width_x_space = refer_boxes_right[0].y - coord_x_space[1]
            else:
                if bin_width:
                    width_x_space = bin_width - coord_x_space[1]
                else:
                    width_x_space = space.ly + (space.min_coord[1] - coord_x_space[1])
            coord_y_space = copy.copy(space.min_coord)
            coord_y_space[1] += block.ly
            coord_z_space = copy.copy(space.min_coord)
            coord_z_space[2] += block.lz
            coord_trans_space = copy.copy(space.min_coord)
            coord_trans_space[0] += block.lx
            coord_trans_space[1] += block.ly
            space_trans = Space.by_length(mx, my, space.lz, coord_trans_space)
            container_x = Space.by_length(
                mx,
                width_x_space,
                space.lz,
                coord_x_space,
                space_trans,
                hold_surface=space.hold_surface)
            container_x.gen_by_which_box_order = packed_box_list[-1].order
            container_y = Space.by_length(
                space.lx,
                my,
                space.lz,
                coord_y_space,
                hold_surface=space.hold_surface)
            container_y.gen_by_which_box_order = packed_box_list[-1].order
            if my > 0:
                container_list.append(container_y)
            if mx > 0:
                container_list.append(container_x)
            if space.lz - block.lz > 0:
                container_list.append(
                    Space.by_length(
                        block.lx,
                        block.ly,
                        space.lz - block.lz,
                        coord_z_space,
                        hold_surface=block.hold_surface))
        # elif packed_box_list and space.min_coord[2] == 0 and not refer_boxes_left:
        #     container_list = []
        #     mx = space.lx - block.lx
        #     my = space.ly - block.ly
        #     coord_x_space = copy.copy(space.min_coord)
        #     coord_x_space[0] += block.lx
        #     coord_x_space[1] = 0
        #     coord_y_space = copy.copy(space.min_coord)
        #     coord_y_space[1] += block.ly
        #     coord_z_space = copy.copy(space.min_coord)
        #     coord_z_space[2] += block.lz
        #     coord_trans_space = copy.copy(space.min_coord)
        #     coord_trans_space[0] += block.lx
        #     coord_trans_space[1] += block.ly
        #     space_trans = Space.by_length(mx, my, space.lz, coord_trans_space)
        #     container_x = Space.by_length(
        #         mx,
        #         space.max_coord[1] - coord_x_space[1],
        #         space.lz,
        #         coord_x_space,
        #         space_trans,
        #         hold_surface=space.hold_surface)
        #     container_x.gen_by_which_box_order = packed_box_list[-1].order
        #     container_y = Space.by_length(
        #         space.lx,
        #         my,
        #         space.lz,
        #         coord_y_space,
        #         hold_surface=space.hold_surface)
        #     container_y.gen_by_which_box_order = packed_box_list[-1].order
        #     if my > 0:
        #         container_list.append(container_y)
        #     if mx > 0:
        #         container_list.append(container_x)
        #     if space.lz - block.lz > 0:
        #         container_list.append(
        #             Space.by_length(
        #                 block.lx,
        #                 block.ly,
        #                 space.lz - block.lz,
        #                 coord_z_space,
        #                 hold_surface=block.hold_surface))
        else:
            container_list = []
            mx = space.lx - block.lx
            my = space.ly - block.ly
            coord_x_space = copy.copy(space.min_coord)
            coord_x_space[0] += block.lx
            coord_y_space = copy.copy(space.min_coord)
            coord_y_space[1] += block.ly
            coord_z_space = copy.copy(space.min_coord)
            coord_z_space[2] += block.lz
            coord_trans_space = copy.copy(space.min_coord)
            coord_trans_space[0] += block.lx
            coord_trans_space[1] += block.ly
            space_trans = Space.by_length(mx, my, space.lz, coord_trans_space)
            container_x = Space.by_length(
                mx,
                space.ly,
                space.lz,
                coord_x_space,
                space_trans,
                hold_surface=space.hold_surface)
            container_y = Space.by_length(
                space.lx,
                my,
                space.lz,
                coord_y_space,
                hold_surface=space.hold_surface)
            if my > 0:
                container_list.append(container_y)
            if mx > 0:
                container_list.append(container_x)
            if space.lz - block.lz > 0:
                container_list.append(
                    Space.by_length(
                        block.lx,
                        block.ly,
                        space.lz - block.lz,
                        coord_z_space,
                        hold_surface=block.hold_surface))
        self.space_list.extend(container_list)
        return self.space_list

    def get_new_epspace(self, new_packed_bin):
        """
            han add (从mypack.py移植过来的)
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

    def update_epspace_by_box_2(self, packed_box_list, epspace_list):
        """
            给改进后的华为装箱算法用的(packing_algorithm3.py, packing_algorithm4.py)
            根据box的位置和尺寸更新ep space的尺寸、hold_surface等数据
            :param packed_box: PackedBox类
            :param space: 某个extreme point所能利用的空间, Space类
        """
        revised_space_list = []
        # wasted_space_list = []
        space_list = copy.deepcopy(epspace_list)
        for space in space_list:
            # if space.min_coord[2] != 0:
            #     revised_space_list.append(space)
            #     continue
            # if space.min_coord[0]+space.lx != space.max_coord[0] or \
            #    space.min_coord[1]+space.ly != space.max_coord[1] or \
            #    space.min_coord[2]+space.lz != space.max_coord[2]:
            #     raise Exception("Space max coord needs to be revised.")
            # # 选出可以为该空间提供支撑平面的box更新其hold_surface
            # support_box_list = [box for box in packed_box_list \
            #                     if (box.z+box.lz) == space.min_coord[2] and \
            #                         is_overlap(space.min_coord, space.max_coord, \
            #                                    [box.x, box.y], [box.x+box.lx, box.y+box.ly])]
            # if len(support_box_list) == 1:
            #     hold_surface_list = get_hold_surface(space, support_box_list)
            #     if hold_surface_list != "Float":
            #         space.hold_surface.extend(hold_surface_list)
            # 选出与该空间重合的box更新空间大小
            selc_box_list = [box for box in packed_box_list \
                            if is_overlap(space.min_coord, space.max_coord, \
                                        [box.x, box.y, box.z],
                                        [box.x+box.lx, box.y+box.ly, box.z+box.lz])]
            for box in selc_box_list:
                # 获得box和space的尺寸和最小最大坐标
                box_min_coord = [box.x, box.y, box.z]
                # box_lx = box.lx
                # box_ly = box.ly
                # box_lz = box.lz
                # box_max_coord = [box_min_coord[0]+box_lx, \
                #                  box_min_coord[1]+box_ly, \
                #                  box_min_coord[2]+box_lz]
                space_min_coord = space.min_coord
                # space_lx = space.lx
                # space_ly = space.ly
                # space_lz = space.lz
                # space_max_coord = [space_min_coord[0]+space_lx, \
                #                    space_min_coord[1]+space_ly, \
                #                    space_min_coord[2]+space_lz]
                # if space_max_coord[0] != space.max_coord[0] or \
                #    space_max_coord[1] != space.max_coord[1] or \
                #    space_max_coord[2] != space.max_coord[2]:
                #     raise Exception("Space max coord needs to be revised.")
                # 依据重叠情况修正space的尺寸和hold_surface
                if is_overlap(space.min_coord, space.max_coord, \
                            [box.x, box.y, box.z],
                            [box.x+box.lx, box.y+box.ly, box.z+box.lz]):
                    # x方向
                    if box_min_coord[1] <= space_min_coord[1] and box_min_coord[2] <= space_min_coord[2]:
                        # 最小x,z坐标不变，最小y坐标变为该box的最最大y坐标
                        space.min_coord[1] = box.y + box.ly
                        # 更新y方向上space的宽度
                        space.ly = space.max_coord[1] - space.min_coord[1]
                    # y方向
                    if box_min_coord[1] > space_min_coord[1]:
                        # if box_min_coord[2] <= space_min_coord[2]:  # 修正y方向的长度
                        if space.ly > (box_min_coord[1] - space_min_coord[1]):
                            space.ly = box_min_coord[1] - space_min_coord[1]
                            # 更新space.max_coord
                            space.max_coord[1] = space.min_coord[1] + space.ly
                            # space.max_coord = [space.min_coord[0] + space.lx,
                            #                 space.min_coord[1] + space.ly,
                            #                 space.min_coord[2] + space.lz]
                            # # 修正该空间的hold_surface
                            # space.hold_surface = revise_hold_surface(space)
                        # if box_min_coord[2] > space_min_coord[2]:  # 可以修正z或y方向的长度
                        #     space1_ly = box_min_coord[1] - space_min_coord[1]
                        #     space2_lz = box_min_coord[2] - space_min_coord[2]
                        #     if space.ly > space1_ly and space.lz > space2_lz:
                        #         # 若修改y方向的长度有更大体积，则修改y方向
                        #         if space.lx*space1_ly*space.lz >= space.lx*space.ly*space2_lz:
                        #             space.ly = space1_ly
                        #             # 更新space.max_coord
                        #             space.max_coord[1] = space.min_coord[1] + space.ly
                        #             # 更新space.hold_surface
                        #             # space.hold_surface = revise_hold_surface(space=space)
                        #         # 否则，修改z方向的长度
                        #         else:
                        #             space.lz = space2_lz
                        #             # 更新space.max_coord. z方向上的改变不影响hold_surface,故无需修正支撑平面
                        #             space.max_coord[2] = space.min_coord[2] + space.lz
                    # z方向
                    # if ((box_min_coord[1] <= space_min_coord[1]) or (box_min_coord[1] <= space_min_coord[1])) and box_min_coord[2] > space_min_coord[2]:
                    #     if space.lz > box_min_coord[2] - space_min_coord[2]:
                    #         space.lz = box_min_coord[2] - space_min_coord[2]
                    #         # 更新space.max_coord. z方向上的改变不影响hold_surface,故无需修正支撑平面
                    #         space.max_coord[2] = space.min_coord[2] + space.lz
            # 若支撑平面列表为空且最小纵坐标不为0，则该空间为悬浮空间，舍弃
            # if not space.hold_surface and space.min_coord[2] != 0:
            #     space.valid_space = "Float"
            # if space.valid_space != "Float":
            #     if space.valid_space:
            #         revised_space_list.append(space)
            #     else:
            #         wasted_space_list.append(space)
            if space.valid_space:
                revised_space_list.append(space)
        self.space_list.clear()
        self.space_list.extend(revised_space_list)
        return revised_space_list
