package com.quadtalent.quadx.packingAlgrithms;

import com.quadtalent.quadx.Layer;
import com.quadtalent.quadx.Vehicle;
import com.quadtalent.quadx.inputEntity.Box;
import com.quadtalent.quadx.inputEntity.Truck;
import com.quadtalent.quadx.manager.LayerManager;
import com.quadtalent.quadx.manager.PlatformLayerManager;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/1/29
 * @mail zhangyan.zy@quadtalent.com
 */
public class LayerHeuristic {
    public static List<Vehicle> pack(Truck truck, List<Box> boxes, List<Layer> layers, Vehicle lastVehicle, Map<String,Box> boxMap){
        List<Vehicle> vehicles = new ArrayList<>();
        List<Box> boxListCopy = new ArrayList<>(boxes);

        boolean lastVehicleCanBeUsed = false;
        if (lastVehicle!=null){
            lastVehicleCanBeUsed = true;
        }
        while (boxListCopy.size()>0 || (layers != null && layers.size()>0)){
//            TODO:这里需要一个检测，每个箱子装空车一定能装入，while循环才能不出问题
            if (lastVehicleCanBeUsed){
                List<Box> leftBoxes = lastVehicle.insertLayer(boxListCopy,layers,boxMap);
                vehicles.add(lastVehicle);
                lastVehicleCanBeUsed = false;
                boxListCopy = leftBoxes;
            }
            else{
                Vehicle tmpVehicle = new Vehicle("tmpVehicle",truck);
                List<Box> leftBoxes = tmpVehicle.insertLayer(boxListCopy,layers,boxMap);
                vehicles.add(tmpVehicle);
                boxListCopy = leftBoxes;
            }
        }
        return vehicles;

    }
}
