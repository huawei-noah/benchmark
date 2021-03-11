package com.my.vrp.SSBH;

import com.my.vrp.Box;
import com.my.vrp.Carriage;
import com.my.vrp.Node;

import java.util.*;

public class Problem {
    public int CLIENT_NUM;
    public Map<Integer, String> PlatformIDCodeMap = new HashMap<>();
    public Map<String, Integer> PlatformCodeIDMap = new HashMap<>();
    public int TRUCKTYPE_NUM;
    public ArrayList<Carriage> BASIC_TRUCKS = new ArrayList<>();
    public Map<String, Double> distanceMap = new HashMap<>();
    public Node depot_start = new Node();
    public ArrayList<Node> clients = new ArrayList<>();
    Node depot_end = new Node();

    public static ArrayList<Double>[] bin_packing(ArrayList<Box>[] boxes_to_load,
                                                  ArrayList<Carriage> load_truck,
                                                  String method){
        ArrayList<Double>[] lengthList = new ArrayList[boxes_to_load.length];
        for(int i=0;i<boxes_to_load.length;i++){
            lengthList[i] = new ArrayList<Double>();
            for(int j=0;j<load_truck.size();j++){
                Double length_get = dblf_long(boxes_to_load[i], load_truck.get(j));
            }
        }
        return lengthList;
    }
    public static Double dblf_long(ArrayList<Box> boxes, Carriage truck){
        double width = truck.getWidth();
        double height = truck.getHeight();
        ArrayList<Box> thisBoxes = new ArrayList<Box>();//按顺序保存该箱子。
        ArrayList<Box> thissortedBox = new ArrayList<Box>();
        ArrayList<Double> horizontal_levels = new ArrayList<Double>(); //后面观察用处

        horizontal_levels.add(0.0);
        thissortedBox = new ArrayList<Box>();//清空已经存放的boxes
        thisBoxes = new ArrayList<Box>();//按顺序保存该箱子。
        Iterator<Box> iteratorBox;
        for(int boxi=0;boxi<boxes.size();boxi++){
            Box curr_box = boxes.get(boxi); //现在要装载的箱子

            //第一步先求3DCorners=========================================================
            ArrayList<Box> Corners3D = new ArrayList<Box>();//如果已经存放的箱子是0，则原点。

            for(int k=0;k<horizontal_levels.size();k++) {
                ArrayList<Box> I_k = new ArrayList<Box>();
                iteratorBox = thissortedBox.iterator();
                while (iteratorBox.hasNext()) {
                    Box currBox = iteratorBox.next();
                    if (currBox.getZCoor() + currBox.getLength() > horizontal_levels.get(k)) {
                        I_k.add(new Box(currBox));
                    }
                }
                //求2DCorners==========================================================begin
                if (I_k.size() < 1) {
                    //如果这个平面之上没有box,添加原点。
                    Box corner = new Box();
                    corner.setXCoor(0.0);
                    corner.setYCoor(0.0);
                    corner.setZCoor(horizontal_levels.get(k));
                    corner.setPlatformid(k);//记录哪个level
                    Corners3D.add(corner);
                }
                else {
                    //Phase 1: identify the extreme items e_1,...,e_m
                    ArrayList<Integer> e = new ArrayList<Integer>();
                    double bar_x = 0.0;//注意I_k是根据y,x排序的。
                    for (int i = 0; i < I_k.size(); i++) {
                        if (I_k.get(i).getXCoor() + I_k.get(i).getWidth() > bar_x) {
                            e.add(i);
                            bar_x = I_k.get(i).getXCoor() + I_k.get(i).getWidth();//
                        }
                    }
                    //Phase 2: determine the corner points
                    double XCoor = 0.0;
                    double YCoor = I_k.get(e.get(0)).getYCoor()+I_k.get(e.get(0)).getHeight();
                    if(XCoor+curr_box.getWidth()<=width&&YCoor+curr_box.getHeight()<=height) {
//							Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
                        Box corner = new Box();
                        corner.setXCoor(XCoor);corner.setYCoor(YCoor);corner.setZCoor(horizontal_levels.get(k));corner.setPlatformid(k);//记录哪个level
                        Corners3D.add(corner);
                    }
                    /**
                     * 是否添加？
                     */
                    if(I_k.get(e.get(0)).getXCoor()>0.0) {
                        XCoor = I_k.get(e.get(0)).getXCoor();
                        YCoor = I_k.get(e.get(0)).getYCoor()+I_k.get(e.get(0)).getHeight();
                        if(XCoor+curr_box.getWidth()<=width&&YCoor+curr_box.getHeight()<=height) {
//							Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
                            Box corner = new Box();
                            corner.setXCoor(XCoor);corner.setYCoor(YCoor);corner.setZCoor(horizontal_levels.get(k));corner.setPlatformid(k);//记录哪个level
                            Corners3D.add(corner);
                        }
                    }
                    for(int j=1;j<e.size();j++) {
                        XCoor = I_k.get(e.get(j-1)).getXCoor()+I_k.get(e.get(j-1)).getWidth();
                        YCoor = I_k.get(e.get(j)).getYCoor()+I_k.get(e.get(j)).getHeight();
                        if(XCoor+curr_box.getWidth()<=width&&YCoor+curr_box.getHeight()<=height) {
//								Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
                            Box corner = new Box();
                            corner.setXCoor(XCoor);corner.setYCoor(YCoor);corner.setZCoor(horizontal_levels.get(k));corner.setPlatformid(k);//记录哪个level
                            Corners3D.add(corner);
                        }
                        if(I_k.get(e.get(j)).getXCoor()>XCoor) {
                            XCoor = I_k.get(e.get(j)).getXCoor();
                            YCoor = I_k.get(e.get(j)).getYCoor()+I_k.get(e.get(j)).getHeight();
                            if(XCoor+curr_box.getWidth()<=width&&YCoor+curr_box.getHeight()<=height) {
//								Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
                                Box corner = new Box();
                                corner.setXCoor(XCoor);corner.setYCoor(YCoor);corner.setZCoor(horizontal_levels.get(k));corner.setPlatformid(k);//记录哪个level
                                Corners3D.add(corner);
                            }
                        }
                    }
                    XCoor = I_k.get(e.get(e.size()-1)).getXCoor()+I_k.get(e.get(e.size()-1)).getWidth();
                    YCoor = 0.0;
                    if(XCoor+curr_box.getWidth()<=width&&YCoor+curr_box.getHeight()<=height) {
//							Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
                        Box corner = new Box();
                        corner.setXCoor(XCoor);corner.setYCoor(YCoor);corner.setZCoor(horizontal_levels.get(k));corner.setPlatformid(k);//记录哪个level
                        Corners3D.add(corner);
                    }
                }
            }

                iteratorBox = Corners3D.iterator();
                while(iteratorBox.hasNext()){
                    Box curr_position = iteratorBox.next();
                    if(curr_position.getXCoor()+curr_box.getWidth()<=width&&curr_position.getYCoor()+curr_box.getHeight()<=height){
                        //判断这个位置能不能站稳
                        //当前箱子的坐标： boxingSequence.x,y,z
                        //当前箱子的底部高度：boxingSequence.y，如果为0的话，就可以了
                        //遍历所有的已经放了的箱子，看是否支撑现在的箱子。（暴力了点）
                        boolean support = false;
                        if(curr_position.getYCoor()==0) {
                            support = true;
                        }
                        else{
                            //计算该箱子的底部面积。
                            double bottomArea = curr_box.getWidth()*curr_box.getLength();
                            double curr_y = curr_position.getYCoor();//+boxingSequence.get(i).getHeight();
                            double crossArea = 0;
                            //计算所有已放箱子的顶部与该箱子的底部交叉面积
                            for (int boxii=0;boxii<thissortedBox.size();boxii++) {
                                //如果这个箱子的顶部与boxingSequence.get(i)的底部在同一水平上
                                Box existBox = thissortedBox.get(boxii);

                                if(Math.abs(existBox.getYCoor()+existBox.getHeight()-curr_y)<=1.5) {
                                    double xc=curr_position.getXCoor(),zc=curr_position.getZCoor(),xe=existBox.getXCoor(),ze=existBox.getZCoor();
                                    double wc=curr_box.getWidth(),lc=curr_box.getLength(),we=existBox.getWidth(),le=existBox.getLength();

                                    if(!((xc+wc<xe)||(xe+we<xc)||(zc+lc<ze)||(ze+le<zc))) {//如果有交叉，则计算交叉面积。
                                        double [] XCoor = {xc,xc+wc,xe,xe+we};
                                        double [] ZCoor = {zc,zc+lc,ze,ze+le};
                                        //sort xc,xc+wc,xe,xe+we
                                        Arrays.sort(XCoor);
                                        Arrays.sort(ZCoor);
                                        //sort zc,zc+lc,ze,ze+le
                                        crossArea = crossArea + Math.abs(XCoor[2]-XCoor[1])*Math.abs(ZCoor[2]-ZCoor[1]);
                                        if(crossArea>=0.8*bottomArea) {support=true;break;}//如果支撑面积大于80%，则不用继续判断了。
                                    }
                                }
                            }

                        }
                        if(support) {//当前箱子可以加入到这辆车中。
                            Box loadBox = new Box(curr_box);
                            loadBox.setXCoor(curr_position.getXCoor());
                            loadBox.setYCoor(curr_position.getYCoor());
                            loadBox.setZCoor(curr_position.getZCoor());

                            //将这个箱子插入到sortedBox里面，按Y-X从大到小进行排序。
                            int idx=0;
                            for(idx=0;idx<thissortedBox.size();idx++) {//按y,x,z来排序。
                                Box thisbox = thissortedBox.get(idx);
                                //如果在一个水平面上，则对比X
                                if(Math.abs(thisbox.getYCoor()+thisbox.getHeight()-loadBox.getYCoor()-loadBox.getHeight())<1.5) {
                                    if(thisbox.getXCoor()+thisbox.getWidth()<loadBox.getXCoor()+loadBox.getWidth()) {break;}
                                }else if(thisbox.getYCoor()+thisbox.getHeight()<loadBox.getYCoor()+loadBox.getHeight()){//如果不在一个水平面上，则对比Y
                                    break;
                                }
                            }
                            thissortedBox.add(idx, new Box(loadBox));//sort sortedBox by y,x coordinate
                            //新增水平面。
                            double curr_level = loadBox.getZCoor()+loadBox.getLength();
                            boolean addFlag=true;
                            for(idx=curr_position.getPlatformid();idx<horizontal_levels.size();idx++) {
                                if(Math.abs(horizontal_levels.get(idx)-curr_level)<1.5) {//两个level相差多远就不加入来了。
                                    addFlag=false;break;
                                }else if(horizontal_levels.get(idx)>curr_level) {
                                    break;
                                }
                            }
                            if(addFlag) horizontal_levels.add(idx, curr_level);
                            thisBoxes.add(loadBox);
                            break;
                        }
                    }
                }
        }
        return null;
    }
}
