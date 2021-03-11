package com.my.vrp.SSBH;

public class Area {
    //记录长和宽，坐标仍然以长宽高的形式保留
    //记录的坐标当中，第三个维度高度是被忽略的
    //在块装载算法中，上下面的面积是一样的，所以可以用hold的surface来代替bottom的surface
    private Double lz;
    private Double lx;
    private Double[] Coords;

    public void setLength(Double lz) {
        this.lz = lz;
    }

    public void setWidth(Double lx) {
        this.lx = lx;
    }

    public void setCoords(Double[] coord) {
        this.Coords = coord.clone();
    }

    public Double getLength() { return lz;}
    public Double getWidth() { return lx;}
}
