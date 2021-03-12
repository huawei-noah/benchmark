import general_utils as utils
import config


def can_in_space(box, bin_obj, space, box_direction):
    """
        判断box是否能装到space中
        :param box: AlgorithmBox类
        :param bin_obj: PackedBin类
        :param space: ExtremePointSpace类
        :param box_direction: 0 or 1
    """
    # 根据方向确定box在x y z轴上的长度
    lx, ly, lz = utils.choose_box_direction_len(
        box.length, box.width, box.height, box_direction)
    if lx + space.min_coord[0] > bin_obj.length - config.distance_to_door:
        if lz + space.min_coord[2] > bin_obj.door_height:
            return False
    box_size = [lx, ly, lz]
    space_size = [space.lx, space.ly, space.lz]
    for i in range(len(box_size)):
        if box_size[i] > space_size[i]:
            return False
    return True


def can_hold_box(box, space, direction):
    """
        判断space的hold_surface是否满足supporting constraints
        cf. Huawei: constraints.py/can_hold_box()
        :param box: AlgorithmBox class
        :param space: Space class
        :param direction: 0 or 1
    """
    # 如果支撑平面为空则该空间为从地面起始的空间
    support_area = 0
    if not space.hold_surface:
        return True
    lx, ly, _ = utils.choose_box_direction_len(box.length, box.width,
                                                box.height, direction)
    min_coord = [space.min_coord[0], space.min_coord[1]]  # box的底面最小坐标
    max_coord = [min_coord[0] + lx, min_coord[1] + ly]  # box的底面最大坐标
    for area in space.hold_surface:  # space.hold_surface是list,包含Area类型的元素
        if area.max_coord[0] != area.min_coord[0] + area.lx or area.max_coord[1] != area.min_coord[1] + area.ly:
            area.max_coord[0] = area.min_coord[0] + area.lx
            area.max_coord[1] = area.min_coord[1] + area.ly
        # 底面一样
        if area.min_coord == min_coord and area.max_coord == max_coord:
            return True
        if not utils.is_overlap(min_coord, max_coord, area.min_coord, area.max_coord):
            # 若box与空间的支撑平面没有重合区域，那么检查下一个area
            continue
        # 计算重合区域面积
        support_area += overlap_area(
            min_coord, max_coord, area.min_coord, area.max_coord)
    box_bottom_area = lx * ly
    if support_area / box_bottom_area >= 1:
        return True
    else:
        return False


def overlap_area(min_coord1, max_coord1, min_coord2, max_coord2):
    """
        计算支撑面积，这里默认是有重合区域
        :param min_coord1: box的backleftlow角的x,y坐标
        :param max_coord1: box的最大x,y坐标
        :param min_coord2: 空间支撑平面的最小x,y坐标
        :param max_coord2: 空间支撑平面的最大x,y坐标
    """
    x_coord = [min_coord1[0], max_coord1[0], min_coord2[0], max_coord2[0]]
    y_coord = [min_coord1[1], max_coord1[1], min_coord2[1], max_coord2[1]]
    x_coord.sort()  # 升序排列
    y_coord.sort()
    overlap_area = (x_coord[2] - x_coord[1]) * (y_coord[2] - y_coord[1])
    return overlap_area


def can_in_bin(bin_obj, weight):
    """
        当bin有最大载重量限制，判断箱子能否装入
        :param bin_obj: PackedBin类
        :param weight: box的重量
    """
    if weight + bin_obj.load_weight > bin_obj.max_weight:
        return False
    return True
