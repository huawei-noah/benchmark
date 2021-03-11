#!/usr/bin/env python
# coding=UTF-8

import copy
from entity import Space, Area,SimpleBlock
import general_utils as utils
import constrains


class SimpleSpace:
    def __init__(self, space_list):
        self.space_list = space_list

    def transfer_space(self, space):
        if space.trans_space:
            next_space = self.space_list.pop()
            if space.trans_space.min_coord[0] == next_space.min_coord[0]:
                next_space.ly += space.trans_space.ly
                next_space.max_coord[
                    1] = next_space.min_coord[1] + next_space.ly
            else:
                next_space.lx += space.trans_space.lx
                next_space.max_coord[
                    0] = next_space.min_coord[0] + next_space.lx
            self.space_list.append(next_space)

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
        """
        if mx > my:
            container_x = Space(
                mx,
                #block.ly,
                space.ly,
                space.lz,
                coord_x_space,
                trans_space=space_trans,
                hold_surface=space.hold_surface)
            container_y = Space(
                #space.lx,
                block.lx,
                my,
                space.lz,
                coord_y_space,
                hold_surface=space.hold_surface)
            container_list.append(container_x)
            container_list.append(container_y)
        
        else:"""
        if my > mx or my > 9*block.ly:
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
                #trans_space=space_trans,
                hold_surface=space.hold_surface)
        else: 
            container_x = Space(
                mx,
                space.ly,
                space.lz,
                coord_x_space,
                hold_surface=space.hold_surface)
            container_y = Space(
                block.lx,
                my,
                space.lz,
                coord_y_space,
                #trans_space=space_trans,
                hold_surface=space.hold_surface)
        container_list.append(container_x)
        container_list.append(container_y)
        if isinstance(block, SimpleBlock):
            container_list.append(
                Space.by_length(
                    block.lx,
                    block.ly,
                    space.lz - block.lz,
                    coord_z_space,
                    hold_surface=block.hold_surface))
        else:
            container_list.append(
                Space.by_length(
                    block.minlx,
                    block.ly,
                    space.lz - block.lz,
                    coord_z_space,
                    hold_surface=block.hold_surface))
        self.space_list.extend(container_list)
        return self.space_list