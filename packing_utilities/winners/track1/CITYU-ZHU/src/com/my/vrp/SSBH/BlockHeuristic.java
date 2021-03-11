package com.my.vrp.SSBH;

import com.my.vrp.Box;
import com.my.vrp.Carriage;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

public class BlockHeuristic {
    private Carriage c;
    private ArrayList<Box> box_list;
    public HashMap<String, ArrayList<Integer>> box_size_map;
    public BlockHeuristic(Carriage c, ArrayList<Box> box_list) {
        this.c = c;
        this.box_list = box_list;
        //将box_list按箱子的类型进行分类。box_size_map，长宽高，宽长高这两种箱子的类型进行分类。
        //“长+宽+高”对应的box_list中箱子的idx
        this.box_size_map = this.gen_box_size_map();
    }
    
    public ArrayList<SimpleBlock> generate_rectangle_block(){
        ArrayList<SimpleBlock> block_table = new ArrayList<>();
        for(Map.Entry<String, ArrayList<Integer>> entry : this.box_size_map.entrySet()){
            String string_size = entry.getKey();
            Double[] sizes = Utils.from_stringsizes_to_doubles(string_size);//长宽高，或宽长高。
            ArrayList<Integer> indexes = entry.getValue();
            int num_of_box = indexes.size();
            //以长宽高储存，在java里面，对应的坐标是z,x,y所以将对应的长度设为lz, lx, ly
            Double lz, lx, ly;
            lz = sizes[0];
            lx = sizes[1];
            ly = sizes[2];
            for(int ny=1;ny<=num_of_box && ny <= (int)(this.c.getHeight() / ly);ny++){
                for(int nz=1; nz<=(int)(num_of_box/ny) && nz <= (int)(this.c.getLength() / lz);nz++){
                    for(int nx=1; nx<=(int)(num_of_box/ny/nz) && nx <=(int)(this.c.getWidth()/ lx);nx++){
                        block_table.add(new SimpleBlock(nz, nx, ny, sizes));
                    }
                }
            }
        }
        return block_table;
    }
    public HashMap<String, ArrayList<Integer>> gen_box_size_map(){
        HashMap<String, ArrayList<Integer>> box_size_map = new HashMap<>();
        for(int box_idx=0;box_idx<this.box_list.size();box_idx++){
            //以长宽高来储存
            //two_sizes[0]:长宽高，two_sizes[1]:宽长高。
            String[] two_sizes = Utils.get_two_box_sizes(this.box_list.get(box_idx));
            for(String s: two_sizes){
                box_size_map.putIfAbsent(s, new ArrayList<Integer>());//如果没有，则加进去。
                box_size_map.get(s).add(box_idx);//为这种类型的箱子增加当前的箱子idx
            }
        }
        return box_size_map;
    }
    public SimpleBlock gen_avail_block(ArrayList<SimpleBlock> block_list,
                                       Space space_using,
                                       HashMap<String, Integer> avail){
        for(int block_idx=0; block_idx<block_list.size();block_idx++){
            SimpleBlock now_block = block_list.get(block_idx);
            if (now_block.getLength() + space_using.getZCoor() > this.c.getLength()
                    && now_block.getHeight() + space_using.getYCoor() >  this.c.getHeight()){
                continue;
            }
            Double[] sizes = now_block.getSize();
            Double[] item_size = now_block.getItemSizes();
            String string_size = Utils.from_doublesize_to_string(item_size);
            Integer box_num_of_blcok = now_block.getBox_num();
            //检查1.是否有足够数量的箱子来构成块
            //2.块是否能放在当前空间
            //3.是否满足支撑约束
            if(avail.get(string_size) >= box_num_of_blcok){
                if(Utils.can_in_space(sizes, space_using.getSize())){
                    if(Utils.can_get_hold_block(now_block, this.box_size_map, this.box_list, this.c, space_using))
                        return now_block;
                }
            }
        }
        return null;
    }
}
