package com.quadtalent.quadx.minimalSpanningTree;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/2/2
 * @mail zhangyan.zy@quadtalent.com
 */
public class UnionFind {
    int[] a;
    public UnionFind(int size) {
        a = new int[size + 1];
        for (int i = 0; i < a.length; i++) {
            a[i] = i;
        }

    }

    public void union(int i, int j) {
        int ip = find(i);
        int jp = find(j);
        if (ip < jp) {
            a[ip] = jp;
        } else {
            a[jp] = ip;
        }

    }

    private int find(int i) {
        while (a[i] != i) {
            a[i] = a[a[i]];
            i = a[i];

        }
        return i;
    }

    public boolean connected(int i, int j) {
        return find(i) == find(j);
    }
}
