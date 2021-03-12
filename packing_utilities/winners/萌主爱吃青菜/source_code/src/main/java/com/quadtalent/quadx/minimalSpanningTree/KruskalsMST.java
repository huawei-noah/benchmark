package com.quadtalent.quadx.minimalSpanningTree;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/2/2
 * @mail zhangyan.zy@quadtalent.com
 */
public class KruskalsMST {
    private WeightedGraph graph;
    private UnionFind unionFind;
    public KruskalsMST(WeightedGraph graph) {
        this.graph = graph;
        this.unionFind=new UnionFind(graph.encoding.size());
    }

    public Map<String,List<Adjacency>> getMST() {
        List<Edge> list = new ArrayList<>();
        PriorityQueue<Edge> q = new PriorityQueue<>(graph.getEdges());
        while (!q.isEmpty()){
            Edge minEdge=q.remove(); // remove min Edge and check if both vertices of this edge is connected
            if(!unionFind.connected(minEdge.getFrom(), minEdge.getTo())){
                list.add(minEdge);
                unionFind.union(minEdge.getFrom(), minEdge.getTo()); // make both vertices one component
            }
        }
        Map<String,List<Adjacency>> adjacentMap = graph.getAdjacent(list);
        return adjacentMap;
    }
}
