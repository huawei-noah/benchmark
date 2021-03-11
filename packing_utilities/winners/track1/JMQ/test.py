import os
import sys
from pack import Pack
import json
import random
import copy
import statistics
from itertools import permutations

def swap_positions(list, pos1, pos2):
     
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list

def reverse_position(list, pos1, pos2):
    list[pos1:pos2+1] = reversed(list[pos1:pos2+1]) 
    return list

def slide_position(list, pos1, pos2):
    tmp = list[pos1]
    list[pos1:pos2] = list[pos1+1:pos2+1]
    list[pos2] = tmp
    return list

def process_must_be_first(platform):
    platformCode = [i["platformCode"] for i in platform]
    platMustbefirst = [i["mustFirst"] for i in platform]
    for i, value in enumerate(platMustbefirst):
        if value and i != 0:
            swap_positions(platformCode, 0, i)
            return True, platformCode

    return False, platformCode

def process_must_be_first_r(route, data):
    platform = data["algorithmBaseParamDto"]["platformDtoList"]
    must_be_first_platform = [i["platformCode"] for i in platform if i["mustFirst"]]
    if must_be_first_platform and must_be_first_platform in route:
        idx = route.index(must_be_first_platform)
        if idx != 0:
            plat = route.pop(idx)
            route.insert(0, plat)

    return route


def generate_multip_routes(flag, platformCode,n=100, data=None):
    routes = [copy.deepcopy(platformCode)]

    if flag:
        index = list(range(1,len(platformCode)))
    else:
        index = list(range(0,len(platformCode)))

    for _ in range(n-1):
        rs = sorted(random.sample(index, 2))
        p = random.random()
        if p < 0.3:
            platformCode = swap_positions(platformCode, rs[0], rs[1])
        elif p < 0.6:
            platformCode = reverse_position(platformCode, rs[0], rs[1])
        else:
            platformCode = slide_position(platformCode, rs[0], rs[1])

        routes.append(copy.deepcopy(platformCode))
    '''
    for route in routes:
        firstds = distanceMap["start_point+" + route[0]] + distanceMap[route[len(route)-1] + "+" + "end_point"]
        for i in range(1,len(route)):
            firstds += distanceMap[route[i-1] + "+" + route[i]]
        routes_len.append(firstds)
    '''
    return routes

def change_last_bin_truck(input_str, last_packed_bin, trucks):
    newsolution = []
    spu_ids = [i.box_id for i in last_packed_bin.packed_box_list]
    platform_list = [i.platform for i in last_packed_bin.packed_box_list]
    platform = sorted(set(platform_list),key=platform_list.index)
    for i in range(1, len(trucks)):
        truck = trucks[i]["truckTypeCode"]
        pack_last = Pack(input_str, spu_ids,  truck_code=truck,  route=[platform])
        pack_bin_last = []
        try:
            pack_bin_last = pack_last.run()
        except:
            pass

        if len(pack_bin_last) == 1:
            newsolution = pack_bin_last[0]
        else:
            break
        
    if newsolution:
        last_packed_bin = newsolution
        return newsolution

    return None

def get_best_soltion_ind(routes_length, ratio_list):
    '''
    最佳个体的选择策略
    '''
    max_length = max(routes_length)
    routes_length_min_max = [i/max_length for i in routes_length]
    ratio_list_kong = [1 - i for i in ratio_list]
    max_ratio = max(ratio_list_kong)
    ratio_list_kong = [i/max_ratio for i in ratio_list_kong]
    standard = []
    for i in range(len(ratio_list_kong)):
        standard.append(ratio_list_kong[i] + routes_length_min_max[i])
    idx = standard.index(min(standard))

    return idx, routes_length[idx], ratio_list[idx]


def gen_res(data, packed_bin_list):
    res = {"estimateCode": data["estimateCode"], "solutionArray": []}
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
            truck["maxLoad"] = packed_bin.max_weight
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

def output_file(res, output_path, data):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_file = os.path.join(output_path, data['estimateCode'])
    with open(output_file, 'w', encoding='utf-8', errors='ignore') as f:
            json.dump(res, f, ensure_ascii=False)


def compute_route_length(packed_bin_list, data):
    route_len = 0    
    for packed_bin in packed_bin_list:
        platform_list = [i.platform for i in packed_bin.packed_box_list]
        platform = sorted(set(platform_list),key=platform_list.index) 
        route_len += get_dist_for_route(platform, data)
        '''
        firstds = distanceMap["start_point+" + platform[0]] + distanceMap[platform[len(platform)-1] + "+" + "end_point"]
        for i in range(1,len(platform)):
            firstds += distanceMap[platform[i-1] + "+" + platform[i]]
        route_len += firstds
        '''
    return route_len

def initilize_path(flag, platformCode, data):
    '''
    '''
    distanceMap = data["algorithmBaseParamDto"]["distanceMap"]
    if flag:
        first_point = [platformCode[0]]
    else:
        sub_dist_map = {k:v for k, v in distanceMap.items() 
        if k.split('+')[0]=="start_point"}
        min_item = min(sub_dist_map, key=sub_dist_map.get)
        mind = min_item.split('+')
        first_point = [mind[1]]

    while True:
        last_item = first_point[-1]
        sub_dist_map = {k:v for k, v in distanceMap.items() 
        if k.split('+')[0]==last_item 
        and k.split('+')[1] not in first_point
        and k.split('+')[1] != "end_point"}
        if sub_dist_map:
            min_item = min(sub_dist_map, key=sub_dist_map.get)
            first_point.append(min_item.split('+')[1])
        else:
            break
    
    return first_point


def get_dist_for_route(route, data):
    distance_map = data["algorithmBaseParamDto"]["distanceMap"]
    firstds = distance_map["start_point+" + route[0]] + distance_map[route[-1] + "+" + "end_point"]
    for i in range(1,len(route)):
        firstds += distance_map[route[i-1] + "+" + route[i]]

    return firstds

def get_each_dist_for_truck(platform, data):
    res = {}
    routes = permutations(platform)
    for route in routes:
        res[route] = get_dist_for_route(route, data)


    return res

def change_truck_route(input_str, packed_bin, data):
    spu_ids = [i.box_id for i in packed_bin.packed_box_list]
    platform_list = [i.platform for i in packed_bin.packed_box_list]
    platform = sorted(set(platform_list),key=platform_list.index)
    if len(platform) < 2:
        return packed_bin
    else:
        res = get_each_dist_for_truck(platform, data)
        order_res = sorted(res.items(), key=lambda x: x[1])
        for route, _ in order_res:
            if list(route) == platform:
                return packed_bin
            else:
                pack_last = Pack(input_str, spu_ids,  truck_code=packed_bin.truck_type_code,  route=[list(route)])
                new_pack_bin = pack_last.run()
                if len(new_pack_bin) == 1:
                    return new_pack_bin[0]

    return packed_bin


def process_one(input_path = u'C:/work/bb/huawei_content/dataset/dataset/E1594871546666',
            output_path = u"C:/work/bb\huawei_content/program/test001"):
    
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        n = 1
        input_str = f.read()
        data = json.loads(input_str)    
        platform = data["algorithmBaseParamDto"]["platformDtoList"]
        flag, platform_code = process_must_be_first(platform)
        platform_code = initilize_path(flag, platform_code, data)
        routes = generate_multip_routes(flag, platform_code, n, data)
        trucks = data["algorithmBaseParamDto"]["truckTypeDtoList"]
        trucks.sort(key=lambda x: x["length"]*x["width"]*x["height"], reverse=True)
        spu_ids_all = [i["spuBoxId"] for i in data["boxes"]]      
        pack = Pack(input_str, spu_ids_all,  truck_code=trucks[0]["truckTypeCode"],  route=[routes[0]])
        packed_bin_list = pack.run()

        res = gen_res(data, packed_bin_list)
        output_file(res, output_path, data)

def process_one_method2(input_path = u'C:/work/bb/huawei_content/dataset/dataset/E1594871546666',
            output_path = u"C:/work/bb\huawei_content/program/test001"):
    input_str = f.read()
    data = json.loads(input_str)    
    platform = data["algorithmBaseParamDto"]["platformDtoList"]
    flag, platform_code = process_must_be_first(platform)    
    trucks = data["algorithmBaseParamDto"]["truckTypeDtoList"]
    trucks.sort(key=lambda x: x["length"]*x["width"]*x["height"], reverse=True)
    spu_ids_all = [i["spuBoxId"] for i in data["boxes"]]
    routes = [[i] for i in platform_code]
    pack = Pack(input_str, spu_ids_all,  truck_code=trucks[0]["truckTypeCode"],  route=routes)
    packed_bin_list = pack.run()
    new_packed_bin_list = []
    spus = []
    for i in range(len(packed_bin_list)):
        if packed_bin_list[i].ratio < 0.8:
            spus.extend(packed_bin_list[i].box_list)
        else:
            new_packed_bin_list.append(packed_bin_list[i])
    spus_ids = [i.box_id for i in spus]
    platform_code_dct = {}
    for box in spus:
        if box.platform in platform_code_dct.keys():
            platform_code_dct[box.platform] += box.length * box.width * box.height
        else:
            platform_code_dct[box.platform] = box.length * box.width * box.height

    platform_code_dct_ord = sorted(platform_code_dct.items(), key=lambda item:item[1], reverse=True)
    route = [i[0] for i in platform_code_dct_ord]
    platform_code = process_must_be_first_r(route, data) 
    pack_new = Pack(input_str, spus_ids,  truck_code=trucks[0]["truckTypeCode"],  route=[platform_code])
    packed_new_bin_list = pack_new.run()
    new_packed_bin_list.extend(packed_new_bin_list)

    for i in range(len(new_packed_bin_list)):
        #if packed_bin_list[i].ratio > 0.5:
         #    continue
        packed_bin = new_packed_bin_list.pop(i)
        lb = change_last_bin_truck(input_str, packed_bin, trucks)
        if lb:
            new_packed_bin_list.insert(i,lb)
        else:
            new_packed_bin_list.insert(i,packed_bin) 

            # change route
        for i in range(len(new_packed_bin_list)):
                #if packed_bin_list[i].ratio > 0.5:
                #    continue
            packed_bin = new_packed_bin_list.pop(i)
            lb = change_truck_route(input_str, packed_bin, data)
            if lb:
                new_packed_bin_list.insert(i,lb)
            else:
                new_packed_bin_list.insert(i,packed_bin) 

    packed_box_ratio = statistics.mean(
                [packed_bin.ratio for packed_bin in new_packed_bin_list])
    route_length = compute_route_length(new_packed_bin_list, data)

    print("file_name = " + str(data["estimateCode"]) + " dist = " + str(route_length) + " ratio =" + str(packed_box_ratio))

    res = gen_res(data, new_packed_bin_list)
    output_file(res, output_path, data)
  

 

def main(argv):
    input_dir = argv[1]
    output_dir = argv[2]
    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            process_one(input_path, output_dir)



if __name__ == "__main__":
    #main(sys.argv)
    
    input_dir = u"data/inputs"
    output_dir= u"data/output"

    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
           process_one(input_path, output_dir)
           break
    
