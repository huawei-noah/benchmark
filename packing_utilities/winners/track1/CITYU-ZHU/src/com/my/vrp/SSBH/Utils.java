package com.my.vrp.SSBH;


import com.my.vrp.Box;
import com.my.vrp.Carriage;
import com.my.vrp.Node;
import com.my.vrp.Route;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;

public class Utils {
    public static String[] get_two_box_sizes(Box box){
        //长宽高来储存
        String string_size_1 = Double.toString(box.getLength()) + "+" + Double.toString(box.getWidth()) + "+" + Double.toString(box.getHeight());
        //宽长高来储存
        String string_size_2 = Double.toString(box.getWidth()) + "+" + Double.toString(box.getLength()) + "+" + Double.toString(box.getHeight());
        String[] size_to_return = {string_size_1,string_size_2};
        return size_to_return;
    }
    public static Double[] from_stringsizes_to_doubles(String sizes){
        String[] string_sizes = sizes.split("\\+");
        Double[] double_to_return = {Double.valueOf(string_sizes[0]),
        		Double.valueOf(string_sizes[1]),
        		Double.valueOf(string_sizes[2])};
        return double_to_return;
    }
    public static void block_table_sorting(ArrayList<SimpleBlock> block_table){
        block_table.sort(new Comparator<SimpleBlock>(){
            public int compare(SimpleBlock b1, SimpleBlock b2) {
                // 按照宽乘高,高宽长来排序
                if (b1.getWidth()* b1.getHeight() > b2.getWidth()* b2.getHeight()) return -1;
                else if(b1.getWidth()* b1.getHeight() < b2.getWidth()* b2.getHeight()) return 1;

                if(b1.getHeight() > b2.getHeight()) return -1;
                else if(b1.getHeight() < b2.getHeight()) return 1;

                if(b1.getWidth() > b2.getWidth()) return -1;
                else if(b1.getWidth() < b2.getWidth()) return 1;

                if(b1.getLength() > b2.getLength()) return -1;
                else if(b1.getLength() < b2.getLength()) return 1;
                else return 0;
            }
        }
        );
    }
    public static String from_doublesize_to_string(Double[] size){
        String string_to_return;
        string_to_return = size[0].toString() + "+" + size[1].toString()+ "+" + size[2].toString();
        return string_to_return;
    }
    public static boolean can_in_space(Double[] block_size, Double[] space_size){
        for(int idx=0; idx<block_size.length; idx++){
            if(block_size[idx] > space_size[idx])
                return false;
        }
        return true;
    }
    public static boolean can_get_hold_block(SimpleBlock now_block,
                                             HashMap<String, ArrayList<Integer>> box_size_map,
                                             ArrayList<Box> box_list,
                                             Carriage c,
                                             Space space_using){
        ArrayList<Integer> residual_box_idx = box_size_map.get(now_block.getStringsize());
        //是否有足够的箱子可以构成块
        if(residual_box_idx.size() >= now_block.getBox_num()){
            if(Utils.can_hold_block(now_block, space_using)){
                return true;
            }
        }
        return false;
    }
    public static boolean can_hold_block(SimpleBlock now_block, Space space_using){
        //首先更新block的支撑平面
        now_block.setHold_surface(space_using);
        if(space_using.getHold_surface()==null){
            return true;
        }
        else {
            Double[] item_size = now_block.getItemSizes();
            for(int i = 0; i<now_block.getNz();i++){
                for(int j=0;j<now_block.getNx();j++){
                    if(item_size[0]*(i+1) <= space_using.getHold_surface().getLength() && item_size[1]*(j+1) <= space_using.getHold_surface().getLength()){
                        continue;
                    }
                    if(item_size[0]*(i+1) > space_using.getHold_surface().getLength() && item_size[1]*(j+1) <= space_using.getHold_surface().getLength()){
                        Double out_length = item_size[0]*(i+1) - space_using.getHold_surface().getLength();
                        if(out_length <= 0.2 * item_size[0]){
                            continue;
                        }else {
                            return false;
                        }
                    }
                    if(item_size[0]*(i+1) <= space_using.getHold_surface().getLength() && item_size[1]*(j+1) > space_using.getHold_surface().getLength()){
                        Double out_width = item_size[1]*(j+1) - space_using.getHold_surface().getWidth();
                        if(out_width <= 0.2 * item_size[1]){
                            continue;
                        }else {
                            return false;
                        }
                    }
                    if(item_size[0]*(i+1) > space_using.getHold_surface().getLength() && item_size[1]*(j+1) > space_using.getHold_surface().getLength()){
                        Double out_area = 0.;
                        out_area += (item_size[1]*(j+1) - space_using.getHold_surface().getWidth())*item_size[0];
                        out_area += (item_size[0]*(i+1) - space_using.getHold_surface().getLength())*item_size[1];
                        out_area -= (item_size[0]*(i+1) - space_using.getHold_surface().getLength())*(item_size[1]*(j+1) - space_using.getHold_surface().getWidth());
                        if(out_area > 0.2*item_size[0]*item_size[1]){
                            return false;
                        }
                    }
                }
            }
            return true;
        }
    }
    public static ArrayList<Integer> assign_block_in_space(SimpleBlock now_block,
                                                           Space space_using,
                                                           ArrayList<Box> boxingSequence,
                                                           HashMap<String, ArrayList<Integer>> box_size_map){
        ArrayList<Integer> packed_boxes_idx = new ArrayList<>();
        Double[] baseCoord = space_using.getCoord();
        String stringsize = now_block.getStringsize();
        Double[] itemsize = now_block.getItemSizes();
        ArrayList<Integer> all_avil_box_idx = box_size_map.get(stringsize);
        Integer box_putting = 0;
        Double[] box_size = new Double[3];
        box_size[0] = boxingSequence.get(all_avil_box_idx.get(box_putting)).getLength();
        box_size[1] = boxingSequence.get(all_avil_box_idx.get(box_putting)).getWidth();
        box_size[2] = boxingSequence.get(all_avil_box_idx.get(box_putting)).getHeight();

        for(int numberY=1; numberY <= now_block.getNy(); numberY++){
            for(int numberX=1; numberX <= now_block.getNx(); numberX++){
                for(int numberZ=1;numberZ <= now_block.getNz(); numberZ++){
                    packed_boxes_idx.add(all_avil_box_idx.get(box_putting));
                    Box this_box = boxingSequence.get(all_avil_box_idx.get(box_putting));
                    if(this_box.getLength()!=itemsize[0])
                        this_box.setDirection(200);
                    this_box.setZCoor(baseCoord[0] + (numberZ - 1)*itemsize[0]);
                    this_box.setXCoor(baseCoord[1] + (numberX - 1)*itemsize[1]);
                    this_box.setYCoor(baseCoord[2] + (numberY - 1)*itemsize[2]);
                    box_putting += 1;
                }
            }
        }
        return packed_boxes_idx;
    }
    public static void update_map(HashMap<String, ArrayList<Integer>> box_size_map,
                                  ArrayList<Box> boxingSequence,
                                  HashMap<String, Integer> avail,
                                  SimpleBlock block_use,
                                  ArrayList<Integer> packed_box_ids){
        String itemsize = block_use.getStringsize();
        Double[] double_itemsize = block_use.getItemSizes();
        String another_itemsize = double_itemsize[1].toString()+"+"+double_itemsize[0].toString()+"+"+double_itemsize[2].toString();
        for(int i=0; i< packed_box_ids.size(); i++){
            Integer idx_to_delete = packed_box_ids.get(i);
            ArrayList<Integer> temp = box_size_map.get(itemsize);
            temp.remove(idx_to_delete);
            temp = box_size_map.get(another_itemsize);
            temp.remove(idx_to_delete);
        }
        avail.put(itemsize, box_size_map.get(itemsize).size());
        avail.put(another_itemsize, box_size_map.get(another_itemsize).size());
    }
    public static ArrayList<Route> ConcatRoute(ArrayList<Route> total_routes, Problem CurrentProblem){
        Route result_route = new Route((int)(1000+Math.random()*8999));
        Carriage want_truck = new Carriage(total_routes.get(0).getCarriage());
        want_truck.setLength(0.0);
        result_route.setCarriage(want_truck);
        LinkedList<Node> nodes = new LinkedList<Node>();
        nodes.add(new Node(CurrentProblem.depot_start));
        for(int client_idx = 0; client_idx < total_routes.size();client_idx++){
            nodes.add(new Node(CurrentProblem.clients.get(client_idx)));

            for(int boxi = 0; boxi < total_routes.get(client_idx).getBoxes().size(); boxi++){
                Box now_box = total_routes.get(client_idx).getBoxes().get(boxi);
                now_box.setZCoor(now_box.getZCoor() + result_route.getCarriage().getLength());
                result_route.getBoxes().add(now_box);
            }
            result_route.getCarriage().setLength(result_route.getCarriage().getLength()+total_routes.get(client_idx).getCarriage().getLength());
        }
        nodes.add(new Node(CurrentProblem.depot_start));
        result_route.setNodes(nodes);
        ArrayList<Route> return_routes = new ArrayList<Route>();
        return_routes.add(result_route);
        return return_routes;
    }
}
