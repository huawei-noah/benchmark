package com.quadtalent.quadx.minimalSpanningTree;

import com.quadtalent.quadx.inputEntity.Platform;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.*;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/2/2
 * @mail zhangyan.zy@quadtalent.com
 */

@Data
@NoArgsConstructor
public class WeightedGraph {
    Set<Edge> edges = new HashSet<>();
    private boolean Undirected = true;
    Map<Integer,String> decoding = new HashMap<>();
    Map<String,Integer> encoding = new HashMap<>();

    public WeightedGraph(List<Platform> platforms){
        encoding.put("start_point",0);
        decoding.put(0,"start_point");
        encoding.put("end_point",1);
        decoding.put(1,"end_point");
        for (int i=2;i<platforms.size()+2;i++){
            Platform item = platforms.get(i-2);
            encoding.put(item.getPlatformCode(),i);
            decoding.put(i,item.getPlatformCode());
        }
    }

    public void addEdge(String v1, String v2, double weight) {
        int from = encoding.get(v1);
        int to = encoding.get(v2);
        Edge edge = new Edge(from, to, weight);
        edges.add(edge);
    }

    public Map<String,List<Adjacency>> getAdjacent(List<Edge> edges){
        Map<String,List<Adjacency>> adjacentMap = new HashMap<>();
        for (Edge edge:edges){
            String from = decoding.get(edge.getFrom());
            String to = decoding.get(edge.getTo());
            if (!adjacentMap.containsKey(from)){
                List<Adjacency> tmpAdjacentList = new ArrayList<>();
                adjacentMap.put(from,tmpAdjacentList);
            }
            if (!adjacentMap.containsKey(to)){
                List<Adjacency> tmpAdjacentList = new ArrayList<>();
                adjacentMap.put(to,tmpAdjacentList);
            }
            List<Adjacency> tmpAdjacentList = adjacentMap.get(from);
            Adjacency element = new Adjacency(to,edge.getWeight());
            tmpAdjacentList.add(element);
            element = new Adjacency(from,edge.getWeight());
            tmpAdjacentList = adjacentMap.get(to);
            tmpAdjacentList.add(element);
        }
        return adjacentMap;
    }

}
