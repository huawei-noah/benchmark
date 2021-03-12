package com.quadtalent.quadx.evaluation;

import com.quadtalent.quadx.Objective;
import com.quadtalent.quadx.SpaceStorageVehicle;
import com.quadtalent.quadx.Utils;
import com.quadtalent.quadx.algrithmDataStructure.PlatformPair;
import com.quadtalent.quadx.inputEntity.Box;
import com.quadtalent.quadx.inputEntity.Platform;
import com.quadtalent.quadx.inputEntity.RawInput;

import java.util.*;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/2/22
 * @mail zhangyan.zy@quadtalent.com
 */
public class Evaluation {
    public static List<Float> getLoadingRates(List<SpaceStorageVehicle> vehicles){
        List<Float> loadingRateList = new ArrayList<>();
        float loadingRate;
        for (SpaceStorageVehicle vehicle:vehicles){
            loadingRate = Math.max(vehicle.getAreaRatio(),(vehicle.getTruck().getMaxLoad()-vehicle.getWeight())/vehicle.getTruck().getMaxLoad());
            loadingRateList.add(loadingRate);
        }
        return loadingRateList;
    }

    public static float getTotalDistance(RawInput rawInput,List<SpaceStorageVehicle> vehicles){
        Map<PlatformPair,Double> distMap = Utils.getDistMap(rawInput.getEnv().getDistanceMap());
        float totalDistance = 0;
        for (SpaceStorageVehicle vehicle:vehicles){
            List<String> platformNameList = vehicle.getPlatformList();
            for (int i=0;i<platformNameList.size();i++){
                if (i==0){
                    PlatformPair platformPair = new PlatformPair("start_point",platformNameList.get(i));
                    totalDistance += distMap.get(platformPair);
                }
                if (i==platformNameList.size() - 1){
                    PlatformPair platformPair = new PlatformPair(platformNameList.get(i),"end_point");
                    totalDistance += distMap.get(platformPair);
                    break;
                }
                PlatformPair platformPair = new PlatformPair(platformNameList.get(i),platformNameList.get(i+1));
                totalDistance += distMap.get(platformPair);
            }
        }
        return totalDistance;
    }

    public static void test(RawInput rawInput,List<SpaceStorageVehicle> vehicles){
        List<Float> loadingRateList = getLoadingRates(vehicles);
        System.out.println(loadingRateList);
        float totalDistance = getTotalDistance(rawInput,vehicles);
        System.out.println(totalDistance);
    }

    public static Objective calculate(RawInput rawInput,List<SpaceStorageVehicle> vehicles){
        List<Float> loadingRateList = getLoadingRates(vehicles);
        float totalLoading = (float) 0.0;
        for (int i=0;i<loadingRateList.size();i++){
            totalLoading += loadingRateList.get(i);
        }
        float f1 = 1- totalLoading/loadingRateList.size();
        float totalDistance = getTotalDistance(rawInput,vehicles);
        return new Objective(f1,totalDistance);
    }


}
