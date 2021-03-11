package com.quadtalent.quadx.manager;

import com.quadtalent.quadx.SpaceStorageVehicle;
import com.quadtalent.quadx.Utils;
import com.quadtalent.quadx.algrithmDataStructure.PlatformPair;
import com.quadtalent.quadx.inputEntity.Platform;
import com.quadtalent.quadx.minimalSpanningTree.Adjacency;
import com.quadtalent.quadx.minimalSpanningTree.KruskalsMST;
import com.quadtalent.quadx.minimalSpanningTree.WeightedGraph;

import java.util.*;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/2/24
 * @mail zhangyan.zy@quadtalent.com
 */
public class TransferManager {
    public Map<PlatformPair,Float> transferMap;
    public Map<String,List<String>> adjacency;
    public Map<String,List<String>> priorityAdjacency;
    public Set<String> bondedWarehouses;
    public Set<String> normalNodes;
    public Set<String> visited;
    public Map<PlatformPair,Double> distMap;

    public TransferManager(List<Platform> platforms,Map<String,Double> distanceMap){
        Init(platforms,distanceMap);
    }

    private void Init(List<Platform> platforms,Map<String,Double> distanceMap)
    {
        distMap = Utils.getDistMap(distanceMap);
        bondedWarehouses = new HashSet<>();
        normalNodes = new HashSet<>();
        transferMap = new HashMap<>();
        adjacency = new HashMap<>();
        for (Platform platform:platforms){
            if (platform.isMustFirst()){
                bondedWarehouses.add(platform.getPlatformCode());
            }
            else{
                normalNodes.add(platform.getPlatformCode());
            }
        }
        for (String from:bondedWarehouses){
            for (String to: normalNodes){
                connect(from, to);
            }
        }
        for (String from:normalNodes){
            for (String to: normalNodes){
                if (from.equals(to)){
                    continue;
                }
                connect(from, to);
            }
        }
        List<String> tmpList = new ArrayList<>();
        for (String to: normalNodes){
            PlatformPair tmpPair = new PlatformPair("start_point",to);
            transferMap.put(tmpPair,(float) 1.0);
            tmpList.add(to);
        }
        adjacency.put("start_point",tmpList);

        tmpList = new ArrayList<>();
        for (String to: bondedWarehouses){
            PlatformPair tmpPair = new PlatformPair("priority",to);
            transferMap.put(tmpPair,(float) 1.0);
            tmpList.add(to);
        }
        adjacency.put("priority",tmpList);
        getStrongConnection(platforms);
        visited = new HashSet<>();
    }

    private Map<String,List<Adjacency>> getMST(List<Platform> platforms){
        Set<PlatformPair> pairSet = new HashSet<>();

        WeightedGraph graph = new WeightedGraph(platforms);
        for (PlatformPair platformPair:distMap.keySet()){
            String from = platformPair.getFrom();
            String to = platformPair.getTo();
            if (from.equals("end_point") || to.equals("end_point")){
                continue;
            }
            PlatformPair twinPair = new PlatformPair(to,from);
            if (pairSet.contains(platformPair)){
                continue;
            }
            pairSet.add(platformPair);
            pairSet.add(twinPair);
            graph.addEdge(from,to,distMap.get(platformPair));
        }
        KruskalsMST mst = new KruskalsMST(graph);
        Map<String,List<Adjacency>> adjacentMap = mst.getMST();
        return adjacentMap;
    }
    private void getStrongConnection(List<Platform> platforms){
        Map<String,List<Adjacency>> adjacentMap = getMST(platforms);
        priorityAdjacency = new HashMap<>();
        for (String from:adjacentMap.keySet()){
            double standard = 0.0;
            for (Adjacency adj:adjacentMap.get(from)){
                if (adj.getWeight()>standard){
                    standard = adj.getWeight();
                }
            }
            if (from.equals("start_point")){
                List<String> tmpList1 = adjacency.get("start_point");
                List<String> tmpList2 = adjacency.get("priority");
                List<String> targetList1 = new ArrayList<>();
                List<String> targetList2 = new ArrayList<>();
                for (String to:tmpList1){
                    PlatformPair platformPair = new PlatformPair("start_point",to);
                    double tmpDist = distMap.get(platformPair);
                    if (tmpDist <= standard){
                        targetList1.add(to);
                    }
                }
                priorityAdjacency.put("start_point",targetList1);
                for (String to:tmpList2){
                    PlatformPair platformPair = new PlatformPair("start_point",to);
                    double tmpDist = distMap.get(platformPair);
                    if (tmpDist <= standard){
                        targetList2.add(to);
                    }
                }
                priorityAdjacency.put("priority",targetList2);
            }
            else{
                List<String> tmpList = adjacency.get(from);
                List<String> targetList = new ArrayList<>();
                for (String to:tmpList){
                    PlatformPair platformPair = new PlatformPair(from,to);
                    double tmpDist = distMap.get(platformPair);
                    if (tmpDist <= standard){
                        targetList.add(to);
                    }
                }
                priorityAdjacency.put(from,targetList);
            }

        }
    }

    private void connect(String from, String to) {
        PlatformPair tmpPair = new PlatformPair(from,to);
        transferMap.put(tmpPair,(float) 1.0);
        if (!adjacency.containsKey(from)){
            List<String> tmpList = new ArrayList<>();
            adjacency.put(from,tmpList);
        }
        List<String> tmpList = adjacency.get(from);
        tmpList.add(to);
    }

    public void clearVisited(){
        visited.clear();
    }

    public String getNextNode(SpaceStorageVehicle vehicle,String lastNode){
        if (vehicle==null){
            Set<String> priority = new HashSet<>(bondedWarehouses);
            priority.removeAll(visited);
            if (priority.size()>0){
                return makeChoice("priority",priority);
            }
            Set<String> starts = new HashSet<>(normalNodes);
            starts.removeAll(visited);
            if (starts.size()>0){
                return makeChoice("start_point",starts);
            }
            return null;
        }
        else{
            Set<String> transfer = new HashSet<>(adjacency.get(lastNode));
            transfer.removeAll(visited);
            if (transfer.size()>0){
                return makeChoice(lastNode,transfer);
            }
            return null;
        }
    }

    private String makeChoice(String from,Set<String> possible){
        List<String> sequence = new ArrayList<>(possible);
        List<Float> weight = new ArrayList<>();
        List<Float> actualWeight = new ArrayList<>();
        float total = (float) 0.0;
        for (String node:sequence){
            PlatformPair platformPair = new PlatformPair(from,node);
            float tmpWeight = transferMap.get(platformPair);
            weight.add(tmpWeight);
            total += tmpWeight;
        }
        for (float tmpWeight:weight){
            actualWeight.add(tmpWeight/total);
        }
        int index = rouletteWheelSelection(actualWeight);
        return sequence.get(index);
    }

    private int rouletteWheelSelection(List<Float> actualWeight){
        Random random = new Random();
        float value = random.nextFloat();
        float probability = (float) 0.0;
        for (int i=0;i<actualWeight.size();i++){
            probability += actualWeight.get(i);
            if (probability > value){
                return i;
            }
        }
        return actualWeight.size()-1;
    }

    public void updateConnectionWeightByTree(){
        for (String from:priorityAdjacency.keySet()){
            for (String to:priorityAdjacency.get(from)){
                PlatformPair tmpPair = new PlatformPair(from,to);
                transferMap.put(tmpPair,(float)2.0);
            }
        }
    }
}
