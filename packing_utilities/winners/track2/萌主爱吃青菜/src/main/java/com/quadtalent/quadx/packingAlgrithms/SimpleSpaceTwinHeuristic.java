package com.quadtalent.quadx.packingAlgrithms;

import com.quadtalent.quadx.SpaceStorageVehicle;
import com.quadtalent.quadx.Vehicle;
import com.quadtalent.quadx.inputEntity.Box;
import com.quadtalent.quadx.inputEntity.Truck;

import java.util.ArrayList;
import java.util.List;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/2/3
 * @mail zhangyan.zy@quadtalent.com
 */
public class SimpleSpaceTwinHeuristic {

    public static List<SpaceStorageVehicle> pack(Truck truck, List<Box> boxes, SpaceStorageVehicle lastVehicle){
        List<SpaceStorageVehicle> vehicles = new ArrayList<>();
        List<Box> boxListCopy = new ArrayList<>(boxes);
        boolean lastVehicleCanBeUsed = false;
        if (lastVehicle!=null){
            lastVehicleCanBeUsed = true;
        }
        while (boxListCopy.size()>0){
//            TODO:这里需要一个检测，每个箱子装空车一定能装入，while循环才能不出问题
            if (lastVehicleCanBeUsed){
                List<Box> leftBoxes = lastVehicle.insertBlock(boxListCopy,true);
                vehicles.add(lastVehicle);
                lastVehicleCanBeUsed = false;
                boxListCopy = leftBoxes;
            }
            else{
                SpaceStorageVehicle tmpVehicle = new SpaceStorageVehicle("tmpVehicle",truck);
                List<Box> leftBoxes = tmpVehicle.insertBlock(boxListCopy,false);
                vehicles.add(tmpVehicle);
                boxListCopy = leftBoxes;
            }
        }
        return vehicles;

    }

}
