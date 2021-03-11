package com.quadtalent.quadx;

import com.quadtalent.quadx.algrithmDataStructure.Route;
import com.quadtalent.quadx.algrithmDataStructure.Space;
import com.quadtalent.quadx.evaluation.Evaluation;
import com.quadtalent.quadx.inputEntity.Box;
import com.quadtalent.quadx.inputEntity.Platform;
import com.quadtalent.quadx.inputEntity.RawInput;
import com.quadtalent.quadx.inputEntity.Truck;
import com.quadtalent.quadx.manager.LayerManager;
import com.quadtalent.quadx.manager.PlatformLayerManager;
import com.quadtalent.quadx.manager.RouteManager;
import com.quadtalent.quadx.manager.TransferManager;
import com.quadtalent.quadx.packingAlgrithms.LayerHeuristic;
import com.quadtalent.quadx.packingAlgrithms.SimpleSpaceBlockHeuristic;
import com.quadtalent.quadx.packingAlgrithms.SimpleSpaceTwinHeuristic;

import java.time.Duration;
import java.time.Instant;
import java.util.*;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/1/27
 * @mail zhangyan.zy@quadtalent.com
 */
public class Problem {
    public static List<Vehicle> packByPlatform(List<Platform> platforms, List<Box> boxes, Truck truck){
        Map<String,List<Box>> platformBoxes = Utils.getPlatformBoxes(boxes);
        List<Route> routes = RouteManager.routeInitialize(platforms);
        Vehicle lastVehicle = null;
        List<Vehicle> allVehicles = new ArrayList<>();
        for (Route route:routes){
            for (Platform platform:route.getPlatforms()){
                List<Box> boxList = platformBoxes.get(platform.getPlatformCode());
                List<Vehicle> vehicles = SimpleSpaceBlockHeuristic.pack(truck,boxList,lastVehicle);
                lastVehicle = vehicles.remove(vehicles.size()-1);
                allVehicles.addAll(vehicles);
            }
            allVehicles.add(lastVehicle);
        }
        return allVehicles;
    }
    public static List<SpaceStorageVehicle> packByPlatformTest(List<Platform> platforms, List<Box> boxes, Truck truck){
        Map<String,List<Box>> platformBoxes = Utils.getPlatformBoxes(boxes);
        List<Route> routes = RouteManager.routeInitialize(platforms);
        SpaceStorageVehicle lastVehicle = null;
        List<SpaceStorageVehicle> allVehicles = new ArrayList<>();
        for (Route route:routes){
            for (Platform platform:route.getPlatforms()){
                List<Box> boxList = platformBoxes.get(platform.getPlatformCode());
                List<SpaceStorageVehicle> vehicles = SimpleSpaceTwinHeuristic.pack(truck,boxList,lastVehicle);
                lastVehicle = vehicles.remove(vehicles.size()-1);
                allVehicles.addAll(vehicles);
            }
            allVehicles.add(lastVehicle);
        }
        return allVehicles;
    }
    public static List<Vehicle> packByPlatformLayer(List<Platform> platforms, List<Box> boxes, Truck truck){
        Map<String,List<Box>> platformBoxes = Utils.getPlatformBoxes(boxes);
        Map<String,Box> boxMap = Utils.getBoxMap(boxes);
        PlatformLayerManager platformLayerManager = LayerManager.getPlatformLayers(boxes,truck, boxMap,0.75);
        List<Route> routes = RouteManager.routeInitialize(platforms);
        Vehicle lastVehicle = null;
        List<Vehicle> allVehicles = new ArrayList<>();
        for (Route route:routes){
            for (Platform platform:route.getPlatforms()){
                List<Box> boxList = platformBoxes.get(platform.getPlatformCode());
                boxList.removeAll(platformLayerManager.getUsedBins());
                List<Layer> layers = platformLayerManager.getPlatformLayers().get(platform.getPlatformCode());
                List<Vehicle> vehicles = LayerHeuristic.pack(truck,boxList,layers,lastVehicle,boxMap);
                lastVehicle = vehicles.remove(vehicles.size()-1);
                allVehicles.addAll(vehicles);
            }
            allVehicles.add(lastVehicle);
        }
        return allVehicles;
    }

//    public static List<SpaceStorageVehicle> packByTurbulence(List<Platform> platforms, List<Box> boxes, Truck truck){
//        Map<String,List<Box>> platformBoxes = Utils.getPlatformBoxes(boxes);
//        TransferManager transferManager = new TransferManager(platforms);
//        SpaceStorageVehicle lastVehicle = null;
//        String lastNode = null;
//        List<SpaceStorageVehicle> allVehicles = new ArrayList<>();
//        String nextNode = transferManager.getNextNode(null,null);
//        while (nextNode!=null){
//            List<Box> boxList = platformBoxes.get(nextNode);
//            if (lastVehicle!=null){
//                List<Box> leftBoxes = lastVehicle.insertBlock(boxList,true);
//                if (leftBoxes.size()>0){
//                    platformBoxes.put(nextNode,leftBoxes);
//                    allVehicles.add(lastVehicle);
//                    lastVehicle = null;
//                    nextNode = transferManager.getNextNode(null,null);
//                }
//                else{
//                    transferManager.visited.add(nextNode);
//                    lastNode = nextNode;
//                    nextNode = transferManager.getNextNode(lastVehicle,lastNode);
//                    if (nextNode==null){
//                        allVehicles.add(lastVehicle);
//                    }
//                }
//            }
//            else{
//                SpaceStorageVehicle tmpVehicle = new SpaceStorageVehicle("tmpVehicle",truck);
//                List<Box> leftBoxes = tmpVehicle.insertBlock(boxList,false);
//                if (leftBoxes.size()>0){
//                    platformBoxes.put(nextNode,leftBoxes);
//                    allVehicles.add(tmpVehicle);
//                    lastVehicle = null;
//                    nextNode = transferManager.getNextNode(null,null);
//                }
//                else{
//                    transferManager.visited.add(nextNode);
//                    lastVehicle = tmpVehicle;
//                    lastNode = nextNode;
//                    nextNode = transferManager.getNextNode(lastVehicle,lastNode);
//                    if (nextNode==null){
//                        allVehicles.add(lastVehicle);
//                    }
//                }
//            }
//        }
//        return allVehicles;
//    }
    public static SpaceStorageVehicle switchSmallerVehicle(SpaceStorageVehicle vehicle,List<Truck> truckList){
        Truck currentTruck = vehicle.getTruck();
        List<Truck> available = new ArrayList<>();
        float currentVolume = currentTruck.getLength() * currentTruck.getWidth() * currentTruck.getHeight();
        for (Truck truck:truckList){
            float tmpVolume = truck.getLength() * truck.getWidth() * truck.getHeight();
            if (tmpVolume < vehicle.getVolume() || truck.getMaxLoad() < vehicle.getTruck().getMaxLoad()-vehicle.getWeight()){
                continue;
            }
            if (tmpVolume < currentVolume){
                available.add(truck);
            }
        }
        Collections.sort(available,new Comparator<Truck>() {
            @Override
            public int compare(Truck o1,Truck o2){
                return (int) ((o1.getLength()*o1.getWidth()*o1.getHeight() - o2.getLength()*o2.getWidth()*o2.getHeight()) * 1000);
            }
        });
        List<String> route = vehicle.getPlatformList();
        for (Truck truck:available){
            Map<String,List<Box>> platformBoxes = Utils.getPlatformBoxes(vehicle.getInsertedBins());
            SpaceStorageVehicle tmpVehicle = new SpaceStorageVehicle("tmpVehicle",truck);
            boolean canUseMark = true;
            for (String node:route){
                List<Box> boxList = platformBoxes.get(node);
                List<Box> leftBoxes = tmpVehicle.insertBlock(boxList,true);
                if (leftBoxes.size() > 0){
                    canUseMark = false;
                    break;
                }
            }
            if (canUseMark){
                return tmpVehicle;
            }
        }

        return null;
    }

    public static List<List<SpaceStorageVehicle>> groupPackByTurbulence(RawInput rawInput,Parameter parameter){
        Map<String,List<Box>> platformBoxesStable = Utils.getPlatformBoxes(rawInput.getBoxes());
        TransferManager transferManager = new TransferManager(rawInput.getEnv().getPlatformDtoList(),rawInput.getEnv().getDistanceMap());
        Truck truck = Utils.getMaxLoadTruck(rawInput.getEnv().getTruckTypeDtoList());
        List<List<SpaceStorageVehicle>> solution = new ArrayList<>();
        int iteration = 0;
        Instant start = Instant.now();
        while (iteration < parameter.totalIterationTimes && Duration.between(start,Instant.now()).toMillis() < parameter.timeLimitation){
            if (iteration==parameter.treeStart){
                transferManager.updateConnectionWeightByTree();
            }
            Map<String,List<Box>> platformBoxes = new HashMap<>(platformBoxesStable);
            SpaceStorageVehicle lastVehicle = null;
            String lastNode;
            List<SpaceStorageVehicle> allVehicles = new ArrayList<>();
            String nextNode = transferManager.getNextNode(null,null);
            while (nextNode!=null){
                List<Box> boxList = platformBoxes.get(nextNode);
                if (lastVehicle!=null){
                    List<Box> leftBoxes = lastVehicle.insertBlock(boxList,true);
                    if (leftBoxes.size()>0){
                        platformBoxes.put(nextNode,leftBoxes);
                        SpaceStorageVehicle attemptVehicle = switchSmallerVehicle(lastVehicle,rawInput.getEnv().getTruckTypeDtoList());
                        if (attemptVehicle==null){
                            allVehicles.add(lastVehicle);
                        }
                        else{
                            allVehicles.add(attemptVehicle);
                        }
//                        allVehicles.add(lastVehicle);
                        lastVehicle = null;
                        nextNode = transferManager.getNextNode(null,null);
                    }
                    else{
                        transferManager.visited.add(nextNode);
                        lastNode = nextNode;
                        nextNode = transferManager.getNextNode(lastVehicle,lastNode);
                        if (nextNode==null){
                            SpaceStorageVehicle attemptVehicle = switchSmallerVehicle(lastVehicle,rawInput.getEnv().getTruckTypeDtoList());
                            if (attemptVehicle==null){
                                allVehicles.add(lastVehicle);
                            }
                            else{
                                allVehicles.add(attemptVehicle);
                            }
                        }
                    }
                }
                else{
                    SpaceStorageVehicle tmpVehicle = new SpaceStorageVehicle("tmpVehicle",truck);
                    List<Box> leftBoxes = tmpVehicle.insertBlock(boxList,false);
                    if (leftBoxes.size()>0){
                        platformBoxes.put(nextNode,leftBoxes);
                        SpaceStorageVehicle attemptVehicle = switchSmallerVehicle(tmpVehicle,rawInput.getEnv().getTruckTypeDtoList());
                        if (attemptVehicle==null){
                            allVehicles.add(tmpVehicle);
                        }
                        else{
                            allVehicles.add(attemptVehicle);
                        }
//                        allVehicles.add(tmpVehicle);
                        lastVehicle = null;
                        nextNode = transferManager.getNextNode(null,null);
                    }
                    else{
                        transferManager.visited.add(nextNode);
                        lastVehicle = tmpVehicle;
                        lastNode = nextNode;
                        nextNode = transferManager.getNextNode(lastVehicle,lastNode);
                        if (nextNode==null){
                            SpaceStorageVehicle attemptVehicle = switchSmallerVehicle(lastVehicle,rawInput.getEnv().getTruckTypeDtoList());
                            if (attemptVehicle==null){
                                allVehicles.add(lastVehicle);
                            }
                            else{
                                allVehicles.add(attemptVehicle);
                            }

                        }
                    }
                }
            }
            if (solution.size()==0){
                solution.add(allVehicles);

            }
            else{
                boolean dominateMark = false;
                for (int i=0;i<solution.size();i++){
                    List<SpaceStorageVehicle> currentResult = solution.get(i);
                    Objective currentObj = Evaluation.calculate(rawInput,currentResult);
                    Objective tmpObj = Evaluation.calculate(rawInput,allVehicles);
                    if (tmpObj.isDominated(currentObj)){
                        dominateMark = true;
                        break;
                    }
                    if (currentObj.isDominated(tmpObj)){
                        solution.remove(i);
                        i--;
                    }
                }
                if (!dominateMark){
                    solution.add(allVehicles);
                }
            }
            transferManager.clearVisited();
            iteration += 1;
        }
        return solution;
    }
}
