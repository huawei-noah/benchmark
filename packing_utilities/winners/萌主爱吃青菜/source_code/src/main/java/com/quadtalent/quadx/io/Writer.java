package com.quadtalent.quadx.io;

import com.quadtalent.quadx.Utils;
import com.quadtalent.quadx.Vehicle;
import com.quadtalent.quadx.algrithmDataStructure.BoxSpace;
import com.quadtalent.quadx.inputEntity.Box;
import com.quadtalent.quadx.inputEntity.RawInput;
import com.quadtalent.quadx.inputEntity.Truck;
import com.quadtalent.quadx.outputEntity.LoadTruck;
import com.quadtalent.quadx.outputEntity.Output;
import com.quadtalent.quadx.outputEntity.PackedBox;

import java.io.File;
import java.util.*;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/1/27
 * @mail zhangyan.zy@quadtalent.com
 */
public class Writer {
    public static void toJsonFile(RawInput rawInput,
                                  List<List<Vehicle>> solution,String path){
//"E:\\code\\emo-huawei\\saic-motor\\output\\test_th001.json"
        List<List<LoadTruck>> solutionArray = new ArrayList<>();
        for (int solutionNumber=0;solutionNumber<solution.size();solutionNumber++){
            List<Vehicle> vehicleList = solution.get(solutionNumber);
            List<LoadTruck> truckArray = new ArrayList<>();
            for (Vehicle vehicle:vehicleList){
                List<PackedBox> packedBoxes = new ArrayList<>();
                List<String> platformNameList = new ArrayList<>();
                Truck tmpTruck = vehicle.getTruck();
                for (int i=0;i<vehicle.getUsedRects().size();i++){
                    Box tmpBox = vehicle.getInsertedBins().get(i);
                    BoxSpace tmpBoxSpace = vehicle.getUsedRects().get(i);
                    tmpBoxSpace.changeCoordinate(tmpTruck);
                    PackedBox packedBox = new PackedBox(tmpBox.getSpuBoxId(),tmpBox.getPlatformCode(),tmpBoxSpace.getDirection(),tmpBoxSpace.x,tmpBoxSpace.y,tmpBoxSpace.z,
                            tmpBoxSpace.getInnerOrder(),tmpBox.getLength(),tmpBox.getWidth(),tmpBox.getHeight(),tmpBox.getWeight());
                    packedBoxes.add(packedBox);
                    if (!platformNameList.contains(tmpBox.getPlatformCode())){
                        platformNameList.add(tmpBox.getPlatformCode());
                    }
                }
                LoadTruck tmpLoadTruck = new LoadTruck(tmpTruck.getTruckTypeId(),tmpTruck.getTruckTypeCode(),vehicle.getUsedRects().size(),
                        vehicle.getVolume(),tmpTruck.getMaxLoad()-vehicle.getWeight(),tmpTruck.getLength(),tmpTruck.getWidth(),tmpTruck.getHeight(),tmpTruck.getMaxLoad(),
                        platformNameList,packedBoxes);
                truckArray.add(tmpLoadTruck);
            }
            solutionArray.add(truckArray);
        }

        Output output = new Output(rawInput.getEstimateCode(),solutionArray);
        JacksonUtils.toFile(new File(path),output);
    }
}
