package com.quadtalent.quadx;

import com.fasterxml.jackson.databind.deser.impl.CreatorCandidate;
import com.quadtalent.quadx.algrithmDataStructure.Face;
import com.quadtalent.quadx.algrithmDataStructure.Line;
import com.quadtalent.quadx.algrithmDataStructure.PlatformPair;
import com.quadtalent.quadx.evaluation.Evaluation;
import com.quadtalent.quadx.inputEntity.Box;
import com.quadtalent.quadx.inputEntity.RawInput;
import com.quadtalent.quadx.inputEntity.Truck;
import com.quadtalent.quadx.io.JacksonUtils;
import com.quadtalent.quadx.io.Reader;
import com.quadtalent.quadx.io.Writer;
import com.quadtalent.quadx.io.Writer1;
import com.quadtalent.quadx.manager.LayerManager;
import com.quadtalent.quadx.manager.PlatformLayerManager;
import com.quadtalent.quadx.manager.RouteManager;
import com.quadtalent.quadx.manager.TransferManager;
import com.quadtalent.quadx.minimalSpanningTree.Adjacency;
import com.quadtalent.quadx.minimalSpanningTree.Edge;
import com.quadtalent.quadx.minimalSpanningTree.KruskalsMST;
import com.quadtalent.quadx.minimalSpanningTree.WeightedGraph;


import java.io.File;
import java.io.FileNotFoundException;
import java.time.Duration;
import java.time.Instant;
import java.util.*;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2020/11/25
 * @mail zhangyan.zy@quadtalent.com
 */
public class Main {
    public static void batchProcessing(String inputPath, String outputPath, Parameter parameter){
        File file = new File(inputPath);
        String[] fileList = file.list();
        for (int i = 0; i < fileList.length; i++) {
            if (fileList[i].contains(".")){
                continue;
            }
            RawInput rawInput = Reader.getRawInput(inputPath + "/" + fileList[i]);
//            Truck targetTruck = Utils.getMaxLoadTruck(rawInput.getEnv().getTruckTypeDtoList());
//            List<SpaceStorageVehicle> allVehicles1 = Problem.packByPlatformTest(rawInput.getEnv().getPlatformDtoList(),rawInput.getBoxes(),targetTruck);
//            List<SpaceStorageVehicle> allVehicles1 = Problem.packByTurbulence(rawInput.getEnv().getPlatformDtoList(),rawInput.getBoxes(),targetTruck);
            List<List<SpaceStorageVehicle>> solution = Problem.groupPackByTurbulence(rawInput,parameter);
//            solution.add(allVehicles1);
            Writer1.toJsonFile(rawInput,solution,outputPath + "/" + fileList[i]);
            System.out.println("hello: " + fileList[i]);
        }
    }
    public static void main(String[] args) {
        Instant start = Instant.now();
        String inputPath = args[0];
        String outputPath = args[1];
        int totalIterationTimes = 50000;
        int treeStart = 20000;
        long timeLimitation = 480 * 1000;//2 minutes for each instance
        Parameter parameter = new Parameter(totalIterationTimes,treeStart,timeLimitation);
//        String inputPath = "E:\\code\\emo-huawei\\saic-motor\\output\\dataset";
//        String outputPath = "E:\\code\\emo-huawei\\saic-motor\\output\\output_2_27v3";
        batchProcessing(inputPath,outputPath,parameter);
        System.out.println(Duration.between(start, Instant.now()).toMillis());

//        RawInput rawInput = Reader.getRawInput("E:\\code\\emo-huawei\\saic-motor\\output\\dataset\\E1595994586443");
////        Truck targetTruck = Utils.getMaxLoadTruck(rawInput.getEnv().getTruckTypeDtoList());
////        List<List<SpaceStorageVehicle>> solution = Problem.groupPackByTurbulence(rawInput);
//        TransferManager transferManager = new TransferManager(rawInput.getEnv().getPlatformDtoList(),rawInput.getEnv().getDistanceMap());
//        transferManager.updateConnectionWeightByTree();
//        System.out.println("ok");
    }
}
