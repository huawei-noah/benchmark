"""
    对于三维装箱问题，如果只考虑箱子的总体积和重量是否超过车辆承载能力(而不考虑其他装箱约束),
    则Routing.evaluate_sol_set()中应调用 truck_list = self.get_a_decodedsol(sol_list),
    这样实际上整个问题处理的是capacitated split delivery VRP; 
    也用到了Order类中的对装载模式的划分

    如果考虑全部的装箱约束,
    则Routing.evaluate_sol_set()中应调用 truck_list = pack_obj.decode_sol_SDVRP(sol_list),
    这才是真正的3L-SDVRP问题.
    而在这里，实际上没有用到Order类中的装载模式1C-FLP (full load pattern)和1C-SP (split pattern)
"""
import json
import os
import copy
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from order import Order

# figure_name = ''


class Routing:
    def __init__(self, order, pack_obj=None, which_pack_obj=1, useFLP=None) -> None:
        """
            :param order: Order类对象
            :param pack_obj: Pack类对象
            :param which_pack_obj: 1:huawei packing方法; 2: 自己的packing方法
            :param useFLP: True: use Full Load Pattern;
                False: not use FLP, and platforms' SP(segment pattern) will include platforms' all boxes;
                Setting useFLP makes sense only when pack_obj=None.
        """
        super().__init__()
        self.evaluated_num = 0
        self.truck_listdict = []
        self.order = order
        self.platform_num = self.order.platform_num
        self.allplatform_listdict = self.order.allplatform_listdict
        if pack_obj:
            self.bin = pack_obj.bin
        else:
            self.bin = self.order.bin  # 体积最大的
        self.distance_2dMatrix = self.order.distanceMap_2dMatrix
        self.pack_patterns_dictlistdictlistAlgorithmBox =\
            self.order.pack_pattern_each_platform_dictlistdictlistAlgorithmBox
        self.res = {"estimateCode": self.order.data["estimateCode"], "solutionArray": []}  # 存储整个order的解决方案
        # self.init_sol_list = self.gen_initial_sol()  # 1~N的一个排列
        self.max_split_costs = float('inf')
        # self.res["solutionArray"].append(
        #     self.get_a_decodedsol(self.init_sol_list))  # 这只是个示范，不要把init_sol_list加上
        # self.f1_list, self.f2_list = self.get_objective_values_of_all_sols()
        self.evaluated_sol_listlist = []  # 存储已经被评价过的解
        self.localsearched_sol_listlist = []  # 存储已经被局部搜索过的解
        # self.Max_split_costs_list = self.get_Max_split_costs()
        # self.nps = len(self.Max_split_costs_list)  # 算法中的nps (number of partial search)
        self.niter = 1  # 算法中的niter, 每个partial search中迭代的次数
        self.history_indvd_listlist = []  # 可用于存储出现过的个体
        self.file_name = self.order.data["estimateCode"]  # 最终保存的解的文件与输入文件名相同
        self.experiments_results = {}  # 存储单次实验结果
        self.experiments_results["evaluation_num"] = []
        # self.full_pop_listtuple = list(itertools.permutations(list(range(1, self.platform_num+1))))
        self.original_list = list(range(1, self.platform_num + 1))
        if pack_obj:
            self.pack_obj = pack_obj
        else:
            self.pack_obj = None
        self.sol_2_trucklist_Dict = {}
        self.which_pack_obj = which_pack_obj
        self.re_pack = False
        self.useFLP = useFLP

    def local_search(self, sol, eval_num=None):
        """ 对某个解进行局部搜索 """
        s_best = sol  # [1,2,3,4,5,6,7,8,9]
        nbh_size = 0
        n = self.platform_num
        for ips in range(1):#range(self.nps):
            self.max_split_costs = float('inf') #self.Max_split_costs_list[ips]
            for inbh in range(1):#range(2)
                if inbh == 0:
                    nbh_size = int(n/2)  # max(n/4, 3)
                else:
                    nbh_size = max(n, 3)
                nondominated_sol_listlist, nondominated_sol_f1f2_listlist =\
                  self.partial_search(s_best, nbh_size, eval_num)
        return nondominated_sol_listlist, nondominated_sol_f1f2_listlist

    def partial_search(self, s_best, nbh_size, eval_num=None):
        s_curr = s_best
        nbh_sample = []
        for iter in range(self.niter):
            nbh_aft_swap_listlist = self.swap_operator(s_curr, nbh_size)
            nbh_aft_2opt_listlist = self.get_nbh_sol_set_by_2opt(s_curr)
            nbh_aft_3opt_listlist = self.get_nbh_sol_set_by_3opt(s_curr)
            nbh = nbh_aft_swap_listlist + nbh_aft_2opt_listlist + nbh_aft_3opt_listlist
            if eval_num is not None and len(nbh) > int(eval_num):
                nbh_sample_listtuple = random.sample(nbh, int(eval_num))  # 控制评价次数
                for nb_tuple in nbh_sample_listtuple:
                    nbh_sample.append(list(nb_tuple))
            else:
                nbh_sample = nbh
            if len(nbh_sample) == 0:
                raise Exception("nbh_sample is empty.")
            nondominated_solset_listlist, nondominated_solset_f1f2_listlist =\
              self.get_nondominated_sol_set(nbh_sample, nbh=nbh)
        return nondominated_solset_listlist, nondominated_solset_f1f2_listlist

    def individual_global_search(self, init_sol, eval_num):
        """ 不严格控制评价次数,即允许最后一代结束后总评价次数大于eval_num """
        t1 = time.process_time()
        # 用于存储搜索过程中的所有非支配解集, 以便选择hv值最大的进行输出
        # [i]表示索引为i的非支配解集,是二维list; [i][j]表示对应的一个解，是一个一维list
        paretoSolSet_3dlist = []
        # 用于绘图.每个元素是一个list,存储每一代的pareto最优解集两个目标函数值,
        # pareto集中的每个元素也是一个list,存储f1f2值
        paretoSolSet_f1f2_3dlist = []
        terminate_indication = 0
        hv_list = []
        eval_num_list = []
        many_nondomi_solset_listlist = []
        many_nondomi_solset_f1f2_listlist = []
        while init_sol in self.localsearched_sol_listlist:
            # 重新采样一个初始解
            # init_sol_listtuple = random.sample(self.full_pop_listtuple, 1)
            # init_sol = list(init_sol_listtuple[0])
            indi_list = self.sample_pop(1)
            init_sol = indi_list[0]
        # 对初始解执行local_search(),得到一个非支配解集
        nondomi_solset_listlist, nondomi_solset_f1f2_listlist =\
          self.local_search(init_sol)
        self.localsearched_sol_listlist.append(init_sol)
        hv = self.get_HV(nondomi_solset_listlist, nondomi_solset_f1f2_listlist)
        hv_list.append(hv)
        eval_num_list.append(self.evaluated_num)
      # Strt:local_search()中不对输入的解进行评价，所以需要单独评价一下初始解,并加入集合
      # 因为evaluate_sol_set()只接受二维list参数，所以要对单个的init_sol做转换
        init_sol_listlist, init_f1f2_listlist = self.evaluate_sol_set([init_sol])
        if len(init_sol_listlist) > 0:  # init_sol没被评价过
            nondomi_solset_listlist.append(init_sol_listlist[0])
            nondomi_solset_f1f2_listlist.append(init_f1f2_listlist[0])
            paretoSolSet_f1f2_3dlist.append(nondomi_solset_f1f2_listlist[:-1])
            paretoSolSet_3dlist.append(nondomi_solset_listlist[:-1])  # 用于最后输出
            self.evaluated_num += 1
        else:
            paretoSolSet_f1f2_3dlist.append(nondomi_solset_f1f2_listlist)
            paretoSolSet_3dlist.append(nondomi_solset_listlist)  # 用于最后输出
        many_nondomi_solset_listlist.extend(nondomi_solset_listlist)
        many_nondomi_solset_f1f2_listlist.extend(nondomi_solset_f1f2_listlist)
        if len(nondomi_solset_listlist) == 1:
            raise Exception("wrong.")
      # End
        while self.evaluated_num < eval_num and terminate_indication < 3:
            # 对非支配解集中的每个解进行local_search(),并将所有非支配解集合并
            for sol_list in nondomi_solset_listlist:
                # 只对没有进行过局部搜索的解进行局部搜索
                if sol_list not in self.localsearched_sol_listlist:
                    nondomi_sol_subset, nondomi_sol_f1f2_subset =\
                        self.local_search(sol_list)
                    # 将该解加入到局部搜索过的个体记录中
                    self.localsearched_sol_listlist.append(sol_list)
                    many_nondomi_solset_listlist.extend(nondomi_sol_subset)
                    many_nondomi_solset_f1f2_listlist.extend(nondomi_sol_f1f2_subset)
            # 从合并后的解集中再找出一个非支配解集
            nondomi_solset_listlist, nondomi_solset_f1f2_listlist =\
                self.get_nondominated_sol_set(
                    many_nondomi_solset_listlist, many_nondomi_solset_f1f2_listlist)
            many_nondomi_solset_listlist = copy.deepcopy(nondomi_solset_listlist)
            many_nondomi_solset_f1f2_listlist = copy.deepcopy(nondomi_solset_f1f2_listlist)
            hv = self.get_HV(nondomi_solset_listlist, nondomi_solset_f1f2_listlist)
            if len(hv_list) > 0 and hv == hv_list[-1]:
                terminate_indication += 1
            eval_num_list.append(self.evaluated_num)
            hv_list.append(hv)  # 可删去
            if len(nondomi_solset_listlist) != 0:
                paretoSolSet_3dlist.append(nondomi_solset_listlist)  # 用于最后输出
                paretoSolSet_f1f2_3dlist.append(nondomi_solset_f1f2_listlist)  # 用于绘图
            else:
                raise Exception("nondomi_solset_listlist is empty.")
        new_hv_list = self.calculate_HVs(paretoSolSet_3dlist, paretoSolSet_f1f2_3dlist)
        best_hv = max(new_hv_list)
        best_nondomi_set_index = new_hv_list.index(best_hv)
        best_nondomi_set_listlist = paretoSolSet_3dlist[best_nondomi_set_index]
        best_nondomi_set_f1f2_listlist = paretoSolSet_f1f2_3dlist[best_nondomi_set_index]
        t2 = time.process_time()
        print("Evaluation Number: ", eval_num_list, flush=True)
        print("HV: ", hv_list, flush=True)
        print("New HV: ", new_hv_list, flush=True)
        # self.get_POF(paretoSolSet_f1f2_3dlist, new_hv_list, eval_num_list)
        # 存储实验结果
        self.experiments_results["best_hv"] = best_hv
        self.experiments_results["best_nondomi_solset"] = best_nondomi_set_listlist
        self.experiments_results["hv_list"] = new_hv_list
        self.experiments_results["paretoSolSet_f1f2_3dlist"] = paretoSolSet_f1f2_3dlist
        self.experiments_results["evaluation_num"].append(eval_num_list)
        self.experiments_results["cpu_time"] = t2 - t1
        return best_nondomi_set_listlist, best_nondomi_set_f1f2_listlist

    def individual_global_search_limit_by_eval_num(self, init_sol, eval_num, gen_num):
        """
        通过把评价次数分配给每一次迭代，严格控制评价次数,不允许最后一代结束后总评价次数大于eval_num;
        每一代结束后的累计评价次数self.evaluated_num不要超过截至该代每代所分配的评价次数之和.
        """
        remaining_eval_num = 0
        eval_num_this_gen = 0
        # 用于存储搜索过程中的所有非支配解集, 以便选择hv值最大的进行输出
        # [i]表示索引为i的非支配解集,是二维list; [i][j]表示对应的一个解，是一个一维list
        paretoSolSet_3dlist = []
        # 用于绘图.每个元素是一个list,存储每一代的pareto最优解集,
        # pareto集中的每个元素也是一个list,存储f1f2值
        paretoSolSet_f1f2_3dlist = []
        terminate_indication = 0
        hv_list = []
        eval_num_list = []
        many_nondomi_solset_listlist = []
        many_nondomi_solset_f1f2_listlist = []
        while init_sol in self.localsearched_sol_listlist:
            # 重新采样一个初始解
            # init_sol_listtuple = random.sample(self.full_pop_listtuple, 1)
            # init_sol = list(init_sol_listtuple[0])
            indi_list = self.sample_pop(1)
            init_sol = indi_list[0]
        # 第0代分配的评价次数
        eval_num_init_gen = int(eval_num/(gen_num+1))
        # 对初始解执行local_search(),得到一个非支配解集
        nondomi_solset_listlist, nondomi_solset_f1f2_listlist =\
          self.local_search(init_sol, eval_num_init_gen)
        self.localsearched_sol_listlist.append(init_sol)
        # hv = self.get_HV(nondomi_solset_listlist, nondomi_solset_f1f2_listlist)
        # hv_list.append(hv)
        eval_num_list.append(self.evaluated_num)
      # Strt:local_search()中不对输入的解进行评价，所以需要单独评价一下初始解,并加入集合
      # 因为evaluate_sol_set()只接受二维list参数，所以要对单个的init_sol做转换
        init_sol_listlist, init_f1f2_listlist = self.evaluate_sol_set([init_sol])
        if len(init_sol_listlist) > 0:  # init_sol没被评价过
            nondomi_solset_listlist.append(init_sol_listlist[0])
            nondomi_solset_f1f2_listlist.append(init_f1f2_listlist[0])
            paretoSolSet_f1f2_3dlist.append(nondomi_solset_f1f2_listlist[:-1])
            paretoSolSet_3dlist.append(nondomi_solset_listlist[:-1])  # 用于最后输出
            self.evaluated_num += 1
        else:
            paretoSolSet_f1f2_3dlist.append(nondomi_solset_f1f2_listlist)
            paretoSolSet_3dlist.append(nondomi_solset_listlist)  # 用于最后输出
        many_nondomi_solset_listlist.extend(nondomi_solset_listlist)
        many_nondomi_solset_f1f2_listlist.extend(nondomi_solset_f1f2_listlist)
        remaining_eval_num = eval_num - self.evaluated_num  # 更新评价次数
      # End
        while self.evaluated_num < eval_num and terminate_indication < 3:
            # 确定该代的最大评价次数
            if gen_num > 0:
                eval_num_this_gen = remaining_eval_num / gen_num
            else:
                eval_num_this_gen = remaining_eval_num
            indi_num = len(nondomi_solset_listlist)
            # 对非支配解集中的每个解进行local_search(),并将所有非支配解集合并
            for sol_list in nondomi_solset_listlist:
                eval_num_this_indi_nbh =\
                    eval_num_this_gen / indi_num  # 该个体的邻域所分配到的最大评价次数
                # 只对没有进行过局部搜索的解进行局部搜索
                if eval_num_this_indi_nbh > 1 and (
                   sol_list not in self.localsearched_sol_listlist):
                    nondomi_sol_subset, nondomi_sol_f1f2_subset =\
                        self.local_search(sol_list, eval_num_this_indi_nbh)
                    # 将该解加入到局部搜索过的个体记录中
                    self.localsearched_sol_listlist.append(sol_list)
                    many_nondomi_solset_listlist.extend(nondomi_sol_subset)
                    many_nondomi_solset_f1f2_listlist.extend(nondomi_sol_f1f2_subset)
            # 更新剩余的评价次数
            remaining_eval_num = eval_num - self.evaluated_num
            gen_num -= 1
            # 从合并后的解集中再找出一个非支配解集
            nondomi_solset_listlist, nondomi_solset_f1f2_listlist =\
                self.get_nondominated_sol_set(
                    many_nondomi_solset_listlist, many_nondomi_solset_f1f2_listlist)
            many_nondomi_solset_listlist = copy.deepcopy(nondomi_solset_listlist)
            many_nondomi_solset_f1f2_listlist = copy.deepcopy(nondomi_solset_f1f2_listlist)
            # hv = self.get_HV(nondomi_solset_listlist, nondomi_solset_f1f2_listlist)
            # if len(hv_list) > 0 and hv == hv_list[-1]:
            if self.evaluated_num == eval_num_list[-1]:
                terminate_indication += 1
            eval_num_list.append(self.evaluated_num)
            # print("Evaluation Number: ", eval_num_list, flush=True)
            # hv_list.append(hv)
            if len(nondomi_solset_listlist) != 0:
                paretoSolSet_3dlist.append(nondomi_solset_listlist)  # 用于最后输出
                paretoSolSet_f1f2_3dlist.append(nondomi_solset_f1f2_listlist)  # 用于绘图
            else:
                raise Exception("nondomi_solset_listlist is empty.")
        new_hv_list = self.calculate_HVs(paretoSolSet_3dlist, paretoSolSet_f1f2_3dlist)
        best_hv = max(new_hv_list)
        best_nondomi_set_index = new_hv_list.index(best_hv)
        best_nondomi_set_listlist = paretoSolSet_3dlist[best_nondomi_set_index]
        best_nondomi_set_f1f2_listlist = paretoSolSet_f1f2_3dlist[best_nondomi_set_index]
        # print("Evaluation Number: ", eval_num_list, flush=True)
        # print("HV: ", hv_list, flush=True)
        # print("New HV: ", new_hv_list, flush=True)
        # self.get_POF(paretoSolSet_f1f2_3dlist, new_hv_list, eval_num_list)
        # 存储实验结果
        # self.experiments_results["best_hv"] = best_hv
        # self.experiments_results["best_nondomi_solset"] = best_nondomi_set_listlist
        # self.experiments_results["hv_list"] = new_hv_list
        # self.experiments_results["paretoSolSet_f1f2_3dlist"] = paretoSolSet_f1f2_3dlist
        # self.experiments_results["evaluation_num"].append(eval_num_list)
        # self.experiments_results["cpu_time"] = t2 - t1
        return best_nondomi_set_listlist, best_nondomi_set_f1f2_listlist, best_hv

    def population_global_search(self, eval_num):
        """
        与两个individual_global_search相比，
        该搜索方法的第一步不是从单个解的邻域找pareto optimal set并往下搜索，
        而是从解空间中采样多个解，从邻域中找pareto optimal set,
        将多个帕雷托最优解集合并，找出新的帕雷托最优解集(实际上相当于将采样的多个解的邻域合并，
        从中找出一个帕雷托最优解集)，再对解集中的每个解往下搜索。
        """
        gen_num = 3
        t1 = time.process_time()
        # 用于存储搜索过程中的所有非支配解集, 以便选择hv值最大的进行输出
        # [i]表示索引为i的非支配解集,是二维list; [i][j]表示对应的一个解，是一个一维list
        paretoSolSet_3dlist = []
        # 用于绘图.每个元素是一个list,存储每一代的pareto最优解集,
        # pareto集中的每个元素也是一个list,存储f1f2值
        paretoSolSet_f1f2_3dlist = []
        terminate_indication = 0
        hv_list = []
        eval_num_list = []
        many_nondomi_solset_listlist = []
        many_nondomi_solset_f1f2_listlist = []
        # 得到初始种群
        # full_pop_listtuple = self.full_pop_listtuple
        # if len(full_pop_listtuple) > 10:
        #     init_pop_listtuple = random.sample(full_pop_listtuple, 10)
        # else:
        #     init_pop_listtuple = full_pop_listtuple
        init_pop_listtuple = self.sample_pop(10)
        # 第0代分配的评价次数
        eval_num_init_gen = int(eval_num/(gen_num+1))
        # 第0代每个个体分配的评价次数
        eval_num_evey_init_indi = int(eval_num_init_gen/len(init_pop_listtuple))
        # 对初始种群中的每个个体,进行local_search(),得到非支配解集
        for indi_sol_tuple in init_pop_listtuple:
            indi_sol = list(indi_sol_tuple)
            nondomi_solset_listlist, nondomi_solset_f1f2_listlist =\
              self.local_search(indi_sol, eval_num_evey_init_indi)
            self.localsearched_sol_listlist.append(indi_sol)
            indi_sol_listlist, indi_f1f2_listlist = self.evaluate_sol_set([indi_sol])
            if len(indi_sol_listlist) > 0:
                self.evaluated_num += 1
                nondomi_solset_listlist.append(indi_sol_listlist[0])
                nondomi_solset_f1f2_listlist.append(indi_f1f2_listlist[0])
                many_nondomi_solset_listlist.extend(nondomi_solset_listlist[:-1])
                many_nondomi_solset_f1f2_listlist.extend(nondomi_solset_f1f2_listlist[:-1])
            else:  # indi_sol已经被评价过
                many_nondomi_solset_listlist.extend(nondomi_solset_listlist)
                many_nondomi_solset_f1f2_listlist.extend(nondomi_solset_f1f2_listlist)
        # 对合并后的解集，从中再找出一个非支配解集
        nondomi_solset_listlist, nondomi_solset_f1f2_listlist =\
            self.get_nondominated_sol_set(
                many_nondomi_solset_listlist, many_nondomi_solset_f1f2_listlist)
        many_nondomi_solset_listlist = copy.deepcopy(nondomi_solset_listlist)
        many_nondomi_solset_f1f2_listlist = copy.deepcopy(nondomi_solset_f1f2_listlist)
        hv = self.get_HV(nondomi_solset_listlist, nondomi_solset_f1f2_listlist)
        hv_list.append(hv)
        eval_num_list.append(self.evaluated_num)
        paretoSolSet_3dlist.append(nondomi_solset_listlist)  # 用于最后输出
        paretoSolSet_f1f2_3dlist.append(nondomi_solset_f1f2_listlist)  # 用于绘图
        remaining_eval_num = eval_num - self.evaluated_num  # 更新评价次数
        # 对非支配解集中的解再进行局部搜索
        while self.evaluated_num < eval_num and terminate_indication < 3:
            # 确定该代的最大评价次数
            if gen_num > 0:
                eval_num_this_gen = remaining_eval_num / gen_num
            else:
                eval_num_this_gen = remaining_eval_num
            indi_num = len(nondomi_solset_listlist)
            # 对非支配解集中的每个解进行local_search(),并将所有非支配解集合并
            for sol_list in nondomi_solset_listlist:
                eval_num_this_indi_nbh =\
                    eval_num_this_gen / indi_num  # 该个体的邻域所分配到的最大评价次数
                # 只对没有进行过局部搜索的解进行局部搜索
                if eval_num_this_indi_nbh > 1 and (
                   sol_list not in self.localsearched_sol_listlist):
                    nondomi_sol_subset, nondomi_sol_f1f2_subset =\
                        self.local_search(sol_list, eval_num_this_indi_nbh)
                    # 将该解加入到局部搜索过的个体记录中
                    self.localsearched_sol_listlist.append(sol_list)
                    many_nondomi_solset_listlist.extend(nondomi_sol_subset)
                    many_nondomi_solset_f1f2_listlist.extend(nondomi_sol_f1f2_subset)
            # 更新剩余的评价次数
            remaining_eval_num = eval_num - self.evaluated_num
            gen_num -= 1
            # 从合并后的解集中再找出一个非支配解集
            nondomi_solset_listlist, nondomi_solset_f1f2_listlist =\
                self.get_nondominated_sol_set(
                    many_nondomi_solset_listlist, many_nondomi_solset_f1f2_listlist)

            """ # 从many_nondomi_solset_listlist中采样10个新个体加入到nondomi_solset_listlist
            disturbance_indis_listtuple = random.sample(many_nondomi_solset_listlist, 10) """

            many_nondomi_solset_listlist = copy.deepcopy(nondomi_solset_listlist)
            many_nondomi_solset_f1f2_listlist = copy.deepcopy(nondomi_solset_f1f2_listlist)
            hv = self.get_HV(nondomi_solset_listlist, nondomi_solset_f1f2_listlist)
            if len(hv_list) > 0 and hv == hv_list[-1]:
                terminate_indication += 1
            eval_num_list.append(self.evaluated_num)
            hv_list.append(hv)
            if len(nondomi_solset_listlist) != 0:
                paretoSolSet_3dlist.append(nondomi_solset_listlist)  # 用于最后输出
                paretoSolSet_f1f2_3dlist.append(nondomi_solset_f1f2_listlist)  # 用于绘图
            else:
                raise Exception("nondomi_solset_listlist is empty.")

            """ # 从many_nondomi_solset_listlist中采样10个新个体加入到nondomi_solset_listlist
            for distb_indi_tuple in disturbance_indis_listtuple:
                distb_indi_list = list(distb_indi_tuple)
                if distb_indi_list not in nondomi_solset_listlist:
                    nondomi_solset_listlist.append(distb_indi_list) """

        new_hv_list = self.calculate_HVs(paretoSolSet_3dlist, paretoSolSet_f1f2_3dlist)
        best_hv = max(new_hv_list)
        best_nondomi_set_index = new_hv_list.index(best_hv)
        best_nondomi_set_listlist = paretoSolSet_3dlist[best_nondomi_set_index]
        best_nondomi_set_f1f2_listlist = paretoSolSet_f1f2_3dlist[best_nondomi_set_index]
        t2 = time.process_time()
        print("Evaluation Number: ", eval_num_list, flush=True)
        print("HV: ", hv_list, flush=True)
        print("New HV: ", new_hv_list, flush=True)
        print("Best hv:", best_hv, flush=True)
        # self.get_POF(paretoSolSet_f1f2_3dlist, new_hv_list, eval_num_list)
        # 存储实验结果
        self.experiments_results["best_hv"] = best_hv
        self.experiments_results["best_nondomi_solset"] = best_nondomi_set_listlist
        self.experiments_results["hv_list"] = new_hv_list
        self.experiments_results["paretoSolSet_f1f2_3dlist"] = paretoSolSet_f1f2_3dlist
        self.experiments_results["evaluation_num"].append(eval_num_list)
        self.experiments_results["cpu_time"] = t2 - t1
        return best_nondomi_set_listlist, best_nondomi_set_f1f2_listlist

    def population_global_search2(self, eval_num, sample_num):
        # t1 = time.process_time()
        gen_num = 3
        # 用于绘图.每个元素是一个list,存储每一代的pareto最优解集,
        # pareto集中的每个元素也是一个list,存储f1f2值
        paretoSolSet_f1f2_3dlist = []
        paretoSolSet_3dlist = []
        hv_list = []
        many_nondomi_solset_listlist = []
        many_nondomi_solset_f1f2_listlist = []
        # 得到初始种群
        # if len(self.full_pop_listtuple) > sample_num:
        #     init_pop_listtuple = random.sample(self.full_pop_listtuple, sample_num)
        # else:
        #     init_pop_listtuple = self.full_pop_listtuple
        init_pop_listtuple = self.sample_pop(sample_num)
        # 对初始种群中的每个个体,进行local_search(),得到非支配解集
        indi_index = 0
        # global figure_name
        # init_figure_name = copy.deepcopy(figure_name)
        for indi_sol_tuple in init_pop_listtuple:
            # figure_name = init_figure_name + 'indi' + str(indi_index) + '.png'
            indi_index += 1
            # print('--------The', indi_index, 'th individual', '---------------------', flush=True)
            self.evaluated_num = 0
            indi_sol = list(indi_sol_tuple)
            #nondomi_sol, nondomi_f1f2 = self.individual_global_search(indi_sol, eval_num/10)
            nondomi_sol, nondomi_f1f2, hv =\
                self.individual_global_search_limit_by_eval_num(indi_sol, eval_num/sample_num, gen_num=gen_num)
            if (nondomi_sol) != 0:
                paretoSolSet_3dlist.append(nondomi_sol)
                paretoSolSet_f1f2_3dlist.append(nondomi_f1f2)
            else:
                raise Exception("nondomi_sol is empty.")
            # hv_list.append(hv)
            many_nondomi_solset_listlist.extend(nondomi_sol)
            many_nondomi_solset_f1f2_listlist.extend(nondomi_f1f2)
        # 对合并解集再找出一个非支配解集
        nondomi_solset_listlist, nondomi_solset_f1f2_listlist =\
            self.get_nondominated_sol_set(
                many_nondomi_solset_listlist, many_nondomi_solset_f1f2_listlist)
        # hv = self.get_HV(nondomi_solset_listlist, nondomi_solset_f1f2_listlist)
        # hv_list.append(hv)
        if len(nondomi_solset_listlist) != 0:
            paretoSolSet_3dlist.append(nondomi_solset_listlist)  # 用于最后输出
            paretoSolSet_f1f2_3dlist.append(nondomi_solset_f1f2_listlist)  # 用于绘图
        else:
            raise Exception("nondomi_solset_listlist is empty.")
        new_hv_list = self.calculate_HVs(paretoSolSet_3dlist, paretoSolSet_f1f2_3dlist)
        best_hv = max(new_hv_list)
        best_nondomi_set_index = new_hv_list.index(best_hv)
        best_nondomi_set_listlist = paretoSolSet_3dlist[best_nondomi_set_index]
        # t2 = time.process_time()
        # print("Best nondominated set:", best_nondomi_set_listlist, flush=True)
        # print("hv:", hv_list, flush=True)
        # print("New hv:", new_hv_list, flush=True)
        # print("Best hv:", best_hv, flush=True)
        # print("end.", flush=True)
        # 存储实验结果
        # self.experiments_results["best_hv"] = best_hv
        # self.experiments_results["best_nondomi_solset"] = best_nondomi_set_listlist
        # self.experiments_results["hv_list"] = new_hv_list
        # self.experiments_results["paretoSolSet_f1f2_3dlist"] = paretoSolSet_f1f2_3dlist
        # self.experiments_results["cpu_time"] = t2 - t1
        # self.get_POF(paretoSolSet_f1f2_3dlist)
        # self.get_POF([many_nondomi_solset_f1f2_listlist])
        return best_nondomi_set_listlist

    def gen_initial_sol(self):
        """ 从start point开始,找距离自己最近的没有参加排列的pickup point作为下一个 """
        init_sol_list = []  # 1~N的一个排列
        platform_flag_dict = {}
        for platform_dict in self.allplatform_listdict:
            platformCode = platform_dict["platformCode"]
            platform_flag_dict[platformCode] = 1
        i = 0
        dist_of_this_point_list = self.distance_2dMatrix[0].tolist()
        while len(init_sol_list) < self.platform_num:
            min_dist = min(dist_of_this_point_list)
            min_dist_index_of_list =\
                dist_of_this_point_list.index(min_dist)  # 在dist_of_this_point_list中的索引
            min_dist_index_array =\
                np.where(self.distance_2dMatrix[i] == min_dist)  # array类型,在原matrix中的索引
            if (int(min_dist_index_array[0]) in init_sol_list) or (int(min_dist_index_array[0]) == (self.platform_num + 1)):
                del dist_of_this_point_list[min_dist_index_of_list]
                continue
            else:
                init_sol_list.append(int(min_dist_index_array[0]))
                i = int(min_dist_index_array[0])
                dist_of_this_point_list = self.distance_2dMatrix[i].tolist()
        return init_sol_list

    def get_a_decodedsol(self, sol_list, useFLP=None):
        """ 
            既包含direct routes,也包含SDVRP routes
            :param sol_list: 一个platform索引(在输入文件中的索引，代码是从1开始)的permutation,
                        e.g., [1,2,3,4,5]表示input文件中(即self.data)["algorithmBaseParamDto"]["platformDtoList"]中
                        第1,2,3,4,5个platform, 而第1个platform的platformCode可能是platform06(而不是platform01)
            :param useFLP: True: use Full Load Pattern; False: not use FLP, platforms' SP(segment pattern) will include platforms' all boxes
        """
        truck_list = []
        if useFLP:
            truck_list_direct_trip = self.direct_route()
            truck_list_SDVRP = self.decode_sol_SDVRP(sol_list, useFLP)
            for truck in truck_list_direct_trip:
                truck_list.append(truck)
            for truck in truck_list_SDVRP:
                truck_list.append(truck)
        else:
            truck_list_SDVRP = self.decode_sol_SDVRP(sol_list, useFLP)
            for truck in truck_list_SDVRP:
                truck_list.append(truck)
        return truck_list

    def direct_route(self):
        direct_route_truck_listdict = []  # 存储该order中所有的direct route
        for i in range(self.platform_num):
            platformCode = self.allplatform_listdict[i]["platformCode"]
            FLP_of_this_platform_dict =\
                self.pack_patterns_dictlistdictlistAlgorithmBox[platformCode][0]
            FLP_num_of_this_platform = FLP_of_this_platform_dict["1C-FLPNum"]
            for j in range(FLP_num_of_this_platform):
                truck = {}  # 每一辆truck都包含了一条route,所装载的boxes以及truck自己的数据
                truck["truckTypeId"] = self.bin.truck_type_id
                truck["truckTypeCode"] = self.bin.truck_type_code
                truck["piece"] = FLP_of_this_platform_dict["boxes_num"][j]  # 一共装载了多少boxes,platformCode定位某个点,0定位1C-FLP装载模式
                truck["volume"] = FLP_of_this_platform_dict["total_boxes_volume"][j]  # Total volume (mm3) of boxes packed in this truck.
                truck["weight"] = FLP_of_this_platform_dict["total_boxes_weight"][j]  # Total weight (kg) of the boxes packed in this truck.
                truck["innerLength"] = self.bin.length  # Truck length (mm). Same as the input file.
                truck["innerWidth"] = self.bin.width  # Truck width
                truck["innerHeight"] = self.bin.height  # Truck height
                truck["maxLoad"] = self.bin.max_weight  # Carrying capacity of the truck (kg).
                truck["platformArray"] = [platformCode]  # direct route. Huawei文件pack.py的_gen_res()中提供的方法可参考
                spu_list = []
                for algorithm_box in FLP_of_this_platform_dict["1C-FLPs"][j]:  # 当前没有考虑box的具体装载情况,后面应该还要改成packed_box,参考Huawei文件中pack.py中_gen_res()
                    spu = {}
                    spu["spuId"] = algorithm_box.box_id
                    spu["direction"] = 100  # 100 or 200,先随便填一个.这个在AlgorithmBox类中没有,存在于PackedBox中
                    spu["x"] = 0  # 同spu["direction"]
                    spu["y"] = 0
                    spu["z"] = 0
                    spu["order"] = 0  # 同spu["direction"]. Order of the box being packed.
                    spu["length"] = algorithm_box.length
                    spu["width"] = algorithm_box.width
                    spu["height"] = algorithm_box.height
                    spu["weight"] = algorithm_box.weight
                    spu["platformCode"] = algorithm_box.platform
                    spu_list.append(spu)
                # spu_list.sort(key=lambda box: box['order'])  # 按‘order’字段进行排序,当前暂不需要
                truck["spuArray"] = spu_list  # 存储装载的boxes
                direct_route_truck_listdict.append(truck)
        return direct_route_truck_listdict

    def decode_sol_SDVRP(self, sol_list, useFLP=None):
        """
        我们将路径分成了direct route和split delivery route,
        该方法解码split delivery route
        注意将解码后的解添加进self.res["solutionArray"]时是哪一级list
        :param sol_list: 一个platform索引(在输入文件中的索引，代码是从1开始)的permutation,
                        e.g., [1,2,3,4,5]表示input文件中(即self.data)["algorithmBaseParamDto"]["platformDtoList"]中
                        第1,2,3,4,5个platform, 而第1个platform的platformCode可能是platform06(而不是platform01)
        :param useFLP: True: use Full Load Pattern; False: not use FLP, platforms' SP(segment pattern) will include platforms' all boxes
        """
        truck_list = []
        sub_sol_listlist = self.get_sub_sol_according2_mustfirst_points(sol_list)
        for sub_sol_list in sub_sol_listlist:
            sub_truck_list = self.decode_sub_sol(sub_sol_list, useFLP)
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
        sub_sol_listlist = []
        end_pointer = len(sol_list)
        platform_listdict = self.allplatform_listdict
        for platform_index in reversed(sol_list):
            if platform_listdict[int(platform_index) - 1]["mustFirst"]:
                start_pointer = sol_list.index(platform_index)
                sub_sol_list = sol_list[start_pointer:end_pointer]
                sub_sol_listlist.append(sub_sol_list)
                end_pointer = start_pointer
        if platform_listdict[sol_list[0]-1]["mustFirst"] is not True:
            sub_sol_listlist.append(sol_list[0:end_pointer])
        return sub_sol_listlist

    def decode_sub_sol(self, sub_sol_list, useFLP=None):
        """
            :param useFLP: True: use Full Load Pattern; False: not use FLP, platforms' SP(segment pattern) will include platforms' all boxes
        """
        total_boxes_listAlgorithmBox, total_boxes_num = self.get_total_boxes(sub_sol_list, useFLP)
        boxes_pointer = 0  # 记录下次应该装载的box的索引
        loaded_boxes_num = 0  # 记录实际已经装载的boxes数量
        truck_list = []
        max_split_costs = self.max_split_costs  # 先临时定一个来测试
        while boxes_pointer < total_boxes_num:
            truck = {}
            truck["truckTypeId"] = self.bin.truck_type_id
            truck["truckTypeCode"] = self.bin.truck_type_code
            truck["piece"] = 0  # 一共装载了多少boxes,platformCode定位某个点,0定位1C-FLP装载模式
            truck["volume"] = 0  # Total volume (mm3) of boxes packed in this truck.
            truck["weight"] = 0  # Total weight (kg) of the boxes packed in this truck.
            truck["innerLength"] = self.bin.length  # Truck length (mm). Same as the input file.
            truck["innerWidth"] = self.bin.width  # Truck width
            truck["innerHeight"] = self.bin.height  # Truck height
            truck["maxLoad"] = self.bin.max_weight  # Carrying capacity of the truck (kg).
            truck["platformArray"] = []
            truck["spuArray"] = []
            truck_volume_load = 0  # 装载的boxes的体积
            truck_weight_load = 0  # 装载的boxes的重量
            spu_list = []
            platform_list = []
            split_indication = float('inf')
            for i in range(boxes_pointer, total_boxes_num):  # 将boxes一个一个地装载
                algo_box = total_boxes_listAlgorithmBox[i]
              # split_indication>1表示当前truck的剩余容量可以容纳下个点的所有货物;
              # split_indication=1表示当前truck的剩余不足以容纳下个点的所有货物，
              # 但允许下个点的(SP模式的)货物进行split，将一部分放到该truck中.
                platformCode = algo_box.platform
                if i != boxes_pointer and platformCode != spu_list[-1]["platformCode"]:
                    prev_platformCode = spu_list[-1]["platformCode"]
                    split_indication = self.get_split_indication(platformCode, prev_platformCode,truck_volume_load, truck_weight_load, max_split_costs)
              # 更新truck装载的volume和weight
                truck_volume_load += algo_box.volume
                truck_weight_load += algo_box.weight
              # 当truck可容纳下个点所有货物或者下个点允许split,且未超过装载能力则装载
                if (split_indication >= 1 and\
                  truck_volume_load < self.bin.volume and\
                  truck_weight_load < self.bin.max_weight):
                  # 将该box载入该truck
                    spu = {}
                    spu["spuId"] = algo_box.box_id
                    spu["direction"] = 100  # 100 or 200,先随便填一个.这个在AlgorithmBox类中没有,存在于PackedBox中
                    spu["x"] = 0  # 同spu["direction"]
                    spu["y"] = 0
                    spu["z"] = 0
                    spu["order"] = i  # 同spu["direction"]. Order of the box being packed.
                    spu["length"] = algo_box.length
                    spu["width"] = algo_box.width
                    spu["height"] = algo_box.height
                    spu["weight"] = algo_box.weight
                    spu["platformCode"] = algo_box.platform
                    spu_list.append(spu)
                  # 更新路径platform_list
                    if spu["platformCode"] not in platform_list:
                        platform_list.append(spu["platformCode"])
                    loaded_boxes_num += 1
                else:  # 超过装载能力则进行下一辆车
                    truck["spuArray"] = spu_list
                    truck["platformArray"] = platform_list
                    truck["volume"] = truck_volume_load - algo_box.volume
                    truck["weight"] = truck_weight_load - algo_box.weight
                    truck["piece"] = i - boxes_pointer
                    if truck["piece"] != len(spu_list):
                        raise Exception("Wrong.\
                          The vaule of truck[\"piece\"] should equal to\
                            the value of (i - boxes_pointer).")
                    truck_list.append(truck)
                    boxes_pointer = i
                    break  # 跳出for循环，执行后续while循环语句
            if loaded_boxes_num == total_boxes_num:
                truck["spuArray"] = spu_list
                truck["platformArray"] = platform_list
                truck["volume"] = truck_volume_load
                truck["weight"] = truck_weight_load
                truck["piece"] = len(spu_list)
                truck_list.append(truck)
                boxes_pointer = loaded_boxes_num
        return truck_list

    def get_total_boxes(self, sub_sol_list, useFLP=None):
        """
            :param useFLP: True: use Full Load Pattern; False: not use FLP, platforms' SP(segment pattern) will include platforms' all boxes
        """
        remaining_boxes_list = []
        remaining_boxes_num = 0
        for platform_index in sub_sol_list:
            platformCode = self.allplatform_listdict[int(platform_index) - 1]["platformCode"]
            if useFLP:
                algoBoxes_list =\
                    self.pack_patterns_dictlistdictlistAlgorithmBox[platformCode][1]["1C-SP"]
                # 对boxes按底面积和体积降序排列
                algoBoxes_list.sort(key=lambda box: (box.length*box.width, box.height, box.width, box.length), reverse=True)
            else:
                algoBoxes_list = \
                    self.order.boxes_by_platform_dictlistAlgorithmBox[platformCode]
                # 对boxes按底面积和体积降序排列
                algoBoxes_list.sort(key=lambda box: (box.length*box.width, box.height, box.width, box.length), reverse=True)
            for algoBox in algoBoxes_list:
                remaining_boxes_list.append(algoBox)
            remaining_boxes_num += len(algoBoxes_list)
        return remaining_boxes_list, remaining_boxes_num

    def get_split_indication(
        self, platformCode, prev_platformCode,\
        truck_volume_load, truck_weight_load, max_split_costs=float('inf')):
        """
        当truck容量不足以装下下一个点的所有1C-SP货物时，根据max_split_costs确定split_indication值.
        依照此函数返回的split_indication来决定是否对下个点进行split，
        若split_indication = 1, 则卡车访问下个点; =0, 则不访问下个点
        """
        split_indication = float('inf')
        pack_patterns = self.pack_patterns_dictlistdictlistAlgorithmBox
        dist_map = self.distance_2dMatrix
        N = self.platform_num
        truck_remaining_v_capacity = self.bin.volume - truck_volume_load
        truck_remaining_w_capacity = self.bin.max_weight - truck_weight_load
        boxes_total_v = pack_patterns[platformCode][1]["total_boxes_volume"]
        boxes_total_w = pack_patterns[platformCode][1]["total_boxes_weight"]
        if truck_remaining_v_capacity < boxes_total_v or\
          truck_remaining_w_capacity < boxes_total_w:
            prev_platformCode_index = next(index for (index, d) in\
              enumerate(self.allplatform_listdict) if d["platformCode"] == prev_platformCode)
            platformCode_index = next(index for (index, d) in\
              enumerate(self.allplatform_listdict) if d["platformCode"] == platformCode)
            d_cprev_c = dist_map[prev_platformCode_index+1][platformCode_index+1]
            d_c_endpoint = dist_map[platformCode_index+1][N+1]
            d_cprev_endpoint = dist_map[prev_platformCode_index+1][N+1]
            if (d_cprev_c + d_c_endpoint) / d_cprev_endpoint <= max_split_costs:
                split_indication = 1
            else:
                split_indication = 0
        return split_indication

    def get_objective_values_of_all_sols(self):
        sols_listlist = self.res["solutionArray"]
        f1_all_sols_list = []
        f2_all_sols_list = []
        for truck_list in sols_listlist:
            f1, f2 = self.get_f1f2_values_one_sol(truck_list)
            f1_all_sols_list.append(f1)
            f2_all_sols_list.append(f2)
        return f1_all_sols_list, f2_all_sols_list

    def get_f1f2_values_one_sol(self, truck_list):
        truck_num = len(truck_list)
        loading_rate_sum = 0
        route_len_sum = 0
        for truck_dict in truck_list:
          # 计算f1
            v_rate = truck_dict["volume"] /\
                (truck_dict["innerLength"]*truck_dict["innerWidth"]*truck_dict["innerHeight"])
            w_rate = truck_dict["weight"] / (truck_dict["maxLoad"])
            loading_rate = max(v_rate, w_rate)
            loading_rate_sum += loading_rate
          # 计算f2
            route_len = 0
            start_platform_index = 0
            end_platform_index = 0
            platform_list = self.allplatform_listdict
            for platformCode in truck_dict["platformArray"]:
                platform_index = next(index for (index, d)\
                  in enumerate(platform_list) if d["platformCode"] == platformCode)
                end_platform_index = platform_index + 1
                route_len += self.distance_2dMatrix[start_platform_index][end_platform_index]
                start_platform_index = end_platform_index
            route_len_sum += route_len
        f1 = 1 - loading_rate_sum / truck_num
        f2 = route_len_sum
        return f1, f2

    def swap_operator(self, encoded_sol_list, nbh_size):
        nbh_aft_swap_listlist = []
        sol_len = len(encoded_sol_list)
        for i in range(sol_len):
            encoded_sol_copy = copy.deepcopy(encoded_sol_list)
            swap_pointer = int((i + nbh_size - 1) % sol_len)
            encoded_sol_copy[i], encoded_sol_copy[swap_pointer] =\
              encoded_sol_copy[swap_pointer], encoded_sol_copy[i]
            if encoded_sol_copy not in nbh_aft_swap_listlist:
                nbh_aft_swap_listlist.append(encoded_sol_copy)
        return nbh_aft_swap_listlist

    def get_nbh_sol_set_by_2opt(self, encoded_sol_list):
        # e.g., [3,1,4,5,6,9,8,2,7] → [0,3,1,4,5,6,9,8,2,7,10]
        extend_sol = [0] + encoded_sol_list + [len(encoded_sol_list) + 1]
        edges_num = len(extend_sol) - 1
        nbh_sol_listlist = []
        for t1 in range(edges_num - 2):
            for t3 in range(t1+2, edges_num):
                nb_sol = self.two_opt_operator(extend_sol, t1, t3)
                nbh_sol_listlist.append(nb_sol)
        return nbh_sol_listlist

    def two_opt_operator(self, encoded_sol_list, t1, t3):
        """
        该操作不会改变encoded_sol_list[0]和encoded_sol_lsit[-1]的位置
        t1,t3是索引,用于指示encoded_sol_list(下称ESL)中的2条breaking-edges;
        t2=t1+1; t4=t3+1;
        t3 > t2 = t1+1; 
        将ELS[t2:t4]反转得到一个邻域个体.
        """
        t2 = t1 + 1
        t4 = t3 + 1
        segment1 = encoded_sol_list[:t2]
        segment2 = encoded_sol_list[t2:t4]
        segment3 = encoded_sol_list[t4:]
        nb_sol = segment1 + list(reversed(segment2)) + segment3
        # 将首尾的start point和delivery point去掉
        return nb_sol[1:-1]

    def get_nbh_sol_set_by_3opt(self, encoded_sol_list):
        # e.g., [3,1,4,5,6,9,8,2,7] → [0,3,1,4,5,6,9,8,2,7,10]
        extend_sol = [0] + encoded_sol_list + [len(encoded_sol_list) + 1]
        edges_num = len(extend_sol) - 1
        nbh_sol_listlist = []
        for t1 in range(edges_num - 4):
            for t3 in range(t1+2, edges_num - 2):
                for t5 in range(t3+2, edges_num):
                    sub_nbh_sol_list =\
                      self.three_opt_operator(extend_sol, t1, t3, t5)
                    nbh_sol_listlist += sub_nbh_sol_list
        return nbh_sol_listlist

    def three_opt_operator(self, encoded_sol_list, t1, t3, t5):
        """
        该操作不会改变encoded_sol_list[0]和encoded_sol_lsit[-1]的位置
        t1,t3,t5是索引,用于指示encoded_sol_list(下称ESL)中的3条breaking-edges;
        t2=t1+1; t4=t3+1; t6=t5+1;
        t3 > t2 = t1+1; t5 > t4 = t3+1;
        break 3条已存在的edges: (ESL[t1], ESL[t2]), (ESL[t3],ESL[t4]), (ESL[t5],ESL[t6]);
        再建立3条reconnecting-edges;
        3-opt语境下，共有8个cases for reconnection, 只有4个cases,
        其中所有reconnecting-dege are all new edges,我们只考虑这种情况;
        其他4种情况包含一个与原来一模一样的reconnection，以及3种实际上是2-opt的情况。
        """
        nbh_sol_listlist = []
        t2 = t1 + 1
        t4 = t3 + 1
        t6 = t5 + 1
        segment1 = encoded_sol_list[0:t2]
        segment2 = encoded_sol_list[t2:t4]
        segment3 = encoded_sol_list[t4:t6]
        segment4 = encoded_sol_list[t6:]
        nb_sol1 = segment1 + list(reversed(segment2)) + list(reversed(segment3)) + segment4
        nb_sol2 = segment1 + segment3 + list(reversed(segment2)) + segment4
        nb_sol3 = segment1 + segment3 + segment2 + segment4
        nb_sol4 = segment1 + list(reversed(segment3)) + segment2 + segment4
        # 将首尾的start point和delivery point去掉
        nbh_sol_listlist.append(nb_sol1[1:-1])
        nbh_sol_listlist.append(nb_sol2[1:-1])
        nbh_sol_listlist.append(nb_sol3[1:-1])
        nbh_sol_listlist.append(nb_sol4[1:-1])
        return nbh_sol_listlist

    def get_Max_split_costs(self):
        Max_split_costs = []
        ric_list = []  # ric: relative insertions costs
        dist_map = self.distance_2dMatrix
        platform_num = self.platform_num
        for i in range(1, platform_num):
            for j in range(i+1, platform_num+1):
                ric_ij = (dist_map[i][j] + dist_map[j][platform_num+1]) /\
                  dist_map[i][platform_num+1]
                ric_ji = (dist_map[j][i] + dist_map[i][platform_num+1])/\
                  dist_map[j][platform_num+1]
                ric_list.append(ric_ij)
                ric_list.append(ric_ji)
        ric_min = min(ric_list)
        ric_max = max(ric_list)
        msc = ric_min  # max split cost
        while msc < ric_max:
            Max_split_costs.append(msc)
            msc *= 2
        Max_split_costs.append(ric_max)
        return Max_split_costs

    def get_nondominated_sol_set(self, sol_listlist, sol_f1f2_listlist=None, nbh=None):
        nondominated_solset_listlist = []
        nondominated_solset_f1f2_listlist = []
        if sol_f1f2_listlist is None:
            new_sol_listlist, f1f2_listlist = self.evaluate_sol_set(sol_listlist, nbh=nbh)
            sol_num = len(new_sol_listlist)
        else:
            f1f2_listlist = sol_f1f2_listlist
            sol_num = len(f1f2_listlist)
            new_sol_listlist = sol_listlist
        # nondominated_indication[i][j]=1表示sol_listlist[i] is dominated by sol_listlist[j];
        # nondominated_indication[i][j]=0表示sol_listlist[i] isn't dominated by sol_listlist[j];
        nondominated_indication = np.zeros((sol_num, sol_num))
        for i in range(sol_num-1):
            f1_i = f1f2_listlist[i][0]
            f2_i = f1f2_listlist[i][1]
            for j in range(sol_num):
                f1_j = f1f2_listlist[j][0]
                f2_j = f1f2_listlist[j][1]
                if f1_j <= f1_i and f2_j <= f2_i and (f1_j < f1_i or f2_j < f2_i):
                    # i is dominated by j
                    nondominated_indication[i][j] = 1
                elif f1_i <= f1_j and f2_i <= f2_j and (f1_i < f1_j or f2_i < f2_j):
                    # j is dominated by i
                    nondominated_indication[j][i] = 1
        nondominated_indication_sum_by_line = np.sum(nondominated_indication, axis=1)
        for i in range(sol_num):
            if nondominated_indication_sum_by_line[i] == 0:
                nondominated_solset_listlist.append(new_sol_listlist[i])
                nondominated_solset_f1f2_listlist.append(f1f2_listlist[i])
        if len(nondominated_solset_listlist) == 0:
            raise Exception("nondominated_solset_listlist is empty.")
        return nondominated_solset_listlist, nondominated_solset_f1f2_listlist

    def evaluate_sol_set(self, sol_listlist, nbh=None):
        new_sol_listlist = []
        f1f2_listlist = []  # 每个元素是一个list,存储一个sol的f1和f2的值
        for sol_list in sol_listlist:
            # 只对未评价过的个体进行评价;self.re_pack:用自己的packing方法进行重新装载时,都是已经评价过的个体
            if sol_list not in self.evaluated_sol_listlist or self.re_pack:
                if self.pack_obj:
                    if self.which_pack_obj == 1:
                        # 先从sol_list得到以bonded warehouse开头的子路径
                        sub_sol_listlist = self.get_sub_sol_according2_mustfirst_points(sol_list)
                        # sub_sol_listlist中存的是索引，转化为以platformCode表示的list
                        route_list = self.sols_2_routes(sub_sol_listlist)
                        # 讲子路径的list传给pack_obj
                        self.pack_obj.routes = route_list
                        res, truck_listdict = self.pack_obj.run()
                        self.sol_2_trucklist_Dict[''.join([str(x) for x in sol_list])] = truck_listdict
                    elif self.which_pack_obj == 2:
                        truck_listdict = self.pack_obj.decode_sol_SDVRP(sol_list)
                        self.sol_2_trucklist_Dict[''.join([str(x) for x in sol_list])] = truck_listdict
                        self.pack_obj.bin = self.pack_obj.origin_bin
                else:
                    truck_listdict = self.get_a_decodedsol(sol_list, useFLP=self.useFLP)
                f1, f2 = self.get_f1f2_values_one_sol(truck_listdict)
                f1f2_listlist.append([f1, f2])
                new_sol_listlist.append(sol_list)
                self.evaluated_sol_listlist.append(sol_list)  # 更新list
            elif nbh is not None:  # 如果个体被评价过了，那么从邻域种再采样来补充
                sample_num = 0
                sample_sol_list = list(random.sample(nbh, 1)[0])
                while(sample_sol_list in self.evaluated_sol_listlist and sample_num < 100):
                    sample_sol_list = list(random.sample(nbh, 1)[0])
                    sample_num += 1
                if sample_sol_list not in self.evaluated_sol_listlist:
                    if self.pack_obj:
                        if self.which_pack_obj == 1: 
                            # 先从sol_list得到以bonded warehouse开头的子路径
                            sub_sol_listlist = self.get_sub_sol_according2_mustfirst_points(sample_sol_list)
                            # sub_sol_listlist中存的是索引，转化为以platformCode表示的list
                            route_list = self.sols_2_routes(sub_sol_listlist)
                            # 讲子路径的list传给pack_obj
                            self.pack_obj.routes = route_list
                            res, truck_listdict = self.pack_obj.run()
                            self.sol_2_trucklist_Dict[''.join([str(x) for x in sol_list])] = truck_listdict
                        elif self.which_pack_obj == 2:
                            truck_listdict = self.pack_obj.decode_sol_SDVRP(sample_sol_list)
                            self.sol_2_trucklist_Dict[''.join([str(x) for x in sol_list])] = truck_listdict
                            self.pack_obj.bin = self.pack_obj.origin_bin
                    else:
                        truck_listdict = self.get_a_decodedsol(sample_sol_list, useFLP=self.useFLP)
                    f1, f2 = self.get_f1f2_values_one_sol(truck_listdict)
                    f1f2_listlist.append([f1, f2])
                    new_sol_listlist.append(sol_list)
                    self.evaluated_sol_listlist.append(sample_sol_list)  # 更新list
        self.evaluated_num += len(new_sol_listlist)
        #print("evaluated_num: ", self.evaluated_num, flush=True)
        return new_sol_listlist, f1f2_listlist

    def get_HV(self, nondominated_solset_listlist, nondominated_solset_f1f2_listlist, min_f1=None, max_f1=None, min_f2=None, max_f2=None):
        """ 计算一个Pareto optimal set的hyper volume值 """
        HV = 0
        nomalized_f1f2_listlist = []
        sol_num = len(nondominated_solset_listlist)
        if min_f1 is None:
            if sol_num == 1:
                return -1
            f1_list = [i[0] for i in nondominated_solset_f1f2_listlist]
            f2_list = [i[1] for i in nondominated_solset_f1f2_listlist]
            f1_max = max(f1_list)
            f1_min = min(f1_list)
            f2_max = max(f2_list)
            f2_min = min(f2_list)
        else:
            f1_max = max_f1
            f1_min = min_f1
            f2_max = max_f2
            f2_min = min_f2
        # begin:先将各f1和f2值标准化
        f1_gap = f1_max - f1_min
        f2_gap = f2_max - f2_min
        if f1_gap == 0 or f2_gap == 0:
            return 1.5
        for f1f2_list in nondominated_solset_f1f2_listlist:
            f1 = f1f2_list[0]
            f2 = f1f2_list[1]
            nomal_f1 = (f1 - f1_min) / f1_gap
            nomal_f2 = (f2 - f2_min) / f2_gap
            nomalized_f1f2_listlist.append([nomal_f1, nomal_f2])
      # end:标准化
        # 按f1值从小到大排序
        nomalized_f1f2_listlist.sort(key=lambda f1f2_list: f1f2_list[0])
        HV += (1.2-nomalized_f1f2_listlist[0][0]) * (1.2-nomalized_f1f2_listlist[0][1])
        for i in range(1, sol_num):
            HV += (1.2-nomalized_f1f2_listlist[i][0]) *\
              (nomalized_f1f2_listlist[i-1][1]-nomalized_f1f2_listlist[i][1])
        return HV

    def calculate_HVs(self, nondomi_sols_3dlist, nondomi_sols_f1f2_3dlist):
        min_f1, max_f1, min_f2, max_f2 = self.get_minmax_f1f2(nondomi_sols_f1f2_3dlist)
        hv_list = []
        for i in range(len(nondomi_sols_3dlist)):
            if len(nondomi_sols_3dlist[i]) == 0:
                hv_list.append(0)
                continue
            hv = self.get_HV(nondomi_sols_3dlist[i], nondomi_sols_f1f2_3dlist[i], min_f1=min_f1, max_f1=max_f1, min_f2=min_f2, max_f2=max_f2)
            hv_list.append(hv)
        return hv_list

    def get_minmax_f1f2(self, nondomi_sols_f1f2_3dlist):
        """ 从产生的所有非支配解集中，寻找最大和最小的f1 f2值，以便我们的对hv的计算更准确 """
        f1_list = []
        f2_list = []
        for nondomi_f1f2_2dlist in nondomi_sols_f1f2_3dlist:
            sub_f1_list = [f1f2_list[0] for f1f2_list in nondomi_f1f2_2dlist]
            sub_f2_list = [f1f2_list[1] for f1f2_list in nondomi_f1f2_2dlist]
            f1_list.extend(sub_f1_list)
            f2_list.extend(sub_f2_list)
        min_f1 = min(f1_list)
        max_f1 = max(f1_list)
        min_f2 = min(f2_list)
        max_f2 = max(f2_list)
        # minmax_f1 = [min_f1, max_f1]
        # minmax_f2 = [min_f2, max_f2]
        return min_f1, max_f1, min_f2, max_f2

    def get_POF(self, paretoSolSet_f1f2_3dlist, hv_list=None, eval_num_list=None):
        """
        绘制每一代的Pareto optimal front (POF);
        :paretoSolSet_f1f2_3dlist: 第一维[i]表示每一代；第二维[i][j]表示每一代的POF上的每一个点；
        第三维[i][j][0]/[i][j][1]表示f1/f2值.
        """
        f1_min, f1_max, f2_min, f2_max = self.get_minmax_f1f2(paretoSolSet_f1f2_3dlist)
        f1_gap = f1_max - f1_min
        f2_gap = f2_max - f2_min
        plt.figure(figsize=(10,5),dpi=200)  #plt.figure(figsize=(50,25))
        gen_num = len(paretoSolSet_f1f2_3dlist)
        if gen_num > 3:
            for i in range(3): # 画3条线
                nomalized_f1f2_listlist = []
                if i == 2:
                    paretoSolSet_f1f2_2dlist = paretoSolSet_f1f2_3dlist[-1]
                    marker = 'r*:'
                    label = 'Iteration ' + str(gen_num)
                elif i == 1:
                    paretoSolSet_f1f2_2dlist = paretoSolSet_f1f2_3dlist[2]
                    marker = 'gs--'
                    label = 'Iteration ' + str(2)
                else:
                    paretoSolSet_f1f2_2dlist = paretoSolSet_f1f2_3dlist[0]
                    marker = 'bo-'
                    label = 'Iteration ' + str(0)
                paretoSolSet_f1f2_2dlist.sort(key=lambda f1f2_list: f1f2_list[0])
                f1_list = [f1f2_list[0] for f1f2_list in paretoSolSet_f1f2_2dlist]
                f2_list = [f1f2_list[1]/100000 for f1f2_list in paretoSolSet_f1f2_2dlist]
                plt.subplot(121)
                plt.xlabel('f1')
                plt.ylabel('f2 (e+05)')
                plt.plot(f1_list, f2_list, marker, label=label)
                plt.legend()  # 显示图例
        elif gen_num == 1:
            paretoSolSet_f1f2_2dlist = paretoSolSet_f1f2_3dlist[-1]
            marker = 'r*:'
            label = 'POF'
            paretoSolSet_f1f2_2dlist.sort(key=lambda f1f2_list: f1f2_list[0])
            f1_list = [f1f2_list[0] for f1f2_list in paretoSolSet_f1f2_2dlist]
            f2_list = [f1f2_list[1]/100000 for f1f2_list in paretoSolSet_f1f2_2dlist]
            plt.subplot(121)
            plt.plot(f1_list, f2_list, marker, label=label)
            plt.legend()  # 显示图例
        else:
            raise Exception("The number of generations is not enougth.")

            """
          # 标准化
            for f1f2_list in paretoSolSet_f1f2_2dlist:
                f1 = f1f2_list[0]
                f2 = f1f2_list[1]
                nomal_f1 = (f1 - f1_min) / f1_gap
                nomal_f2 = (f2 - f2_min) / f2_gap
                nomalized_f1f2_listlist.append([nomal_f1, nomal_f2])
          # end:标准化
            # 按f1值从小到大排序
            nomalized_f1f2_listlist.sort(key=lambda f1f2_list: f1f2_list[0])
            nomal_f1_list = [f1f2_list[0] for f1f2_list in nomalized_f1f2_listlist]
            nomal_f2_list = [f1f2_list[1] for f1f2_list in nomalized_f1f2_listlist]
            plt.subplot(122)
            plt.plot(nomal_f1_list, nomal_f2_list, marker, label=label)
            """
        # 画hv曲线
        if hv_list is not None:
            plt.subplot(122)
            plt.plot(eval_num_list, hv_list, 'ro-', label='Hyper volume (HV)')
        plt.xlabel('# evaluation')
        plt.ylabel('HV')
        plt.legend()  # 显示图例
        plt.savefig(figure_name)
        # plt.show()

    def save_sols(self, output_path):
        """ 将解保存到输出文件 """
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        output_file = os.path.join(output_path, self.file_name)
        with open(
                output_file, 'w', encoding='utf-8', errors='ignore') as f:
            json.dump(self.res, f, ensure_ascii=False, indent=4)  # 将解保存为json文件，无后缀

    def save_experiments_results(self, output_path):
        """ 将实验结果保存到输出文件 """
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        output_file = os.path.join(output_path, self.file_name + "_results.json")
        with open(
                output_file, 'w', encoding='utf-8', errors='ignore') as f:
            json.dump(self.experiments_results, f, ensure_ascii=False, indent=4)  # 将解保存为json文件

    def get_random_init_sol(self, num):
        init_pop = self.sample_pop(num)
        return init_pop

    def sample_pop(self, num):
        """
        生成num数量的个体
        :param num: 要生成的个体数
        :return indi_list: 包含num个元素，每个元素都是一个platform索引的排列
        """
        indi_list = []
        for i in range(num):
            indi = np.random.choice(self.original_list, size=self.platform_num, replace=False)
            indi_list.append(list(indi))
        return indi_list

    def sols_2_routes(self, sol_2dList):
        """
        将sol_2dList中以platform索引表示的路径转化为以platformCode表示的route;
        :param sol_2dList: 每个元素都是一条以bonded warehouse开头的路径(路径中以platform在输入文件中的索引(这里是从1开始而不是从0开始表示第一个platform)表示该platform)
        """
        routes = []
        for sol in sol_2dList:
            route = []
            for index in sol:
                platformCode = self.allplatform_listdict[index-1]["platformCode"]
                route.append(platformCode)
            routes.append(route)
        return routes

    def select_customer_pairs(self):
        """ 现在还用不着这个 """
        return
