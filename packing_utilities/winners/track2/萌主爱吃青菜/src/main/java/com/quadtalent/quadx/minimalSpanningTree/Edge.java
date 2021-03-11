package com.quadtalent.quadx.minimalSpanningTree;

import com.quadtalent.quadx.algrithmDataStructure.Size;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Objects;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/2/2
 * @mail zhangyan.zy@quadtalent.com
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Edge implements Comparable<Edge>{
    private int from;
    private int to;
    private double weight;

    @Override
    public int compareTo(Edge o) {
        return (int) (this.weight - o.weight);
    }
}
