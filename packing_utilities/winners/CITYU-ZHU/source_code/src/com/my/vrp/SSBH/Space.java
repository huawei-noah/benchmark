package com.my.vrp.SSBH;

public class Space {
    private Double vol;
    private Double lz;
    private Double lx;
    private Double ly;
    private Double[] Coord;
    private Space trans_space;
    private Area hold_surface;
    public Space(Double lz, Double lx, Double ly, Double[] Coord){
        this.lz = lz;
        this.lx = lx;
        this.ly = ly;
        this.vol = lz * lx * ly;
        this.Coord = Coord;
    }
    public Space(Double lz, Double lx, Double ly, Double[] Coord, Space trans_space){
        this.lz = lz;
        this.lx = lx;
        this.ly = ly;
        this.vol = lz * lx * ly;
        this.Coord = Coord;
        this.trans_space = trans_space;
    }
    public Space(Double lz, Double lx, Double ly, Double[] Coord, Space trans_space, Area hold_surface){
        this.lz = lz;
        this.lx = lx;
        this.ly = ly;
        this.vol = lz * lx * ly;
        this.Coord = Coord;
        this.trans_space = trans_space;
        this.hold_surface = hold_surface;
    }
    public Space(Double lz, Double lx, Double ly, Double[] Coord, Area hold_surface){
        this.lz = lz;
        this.lx = lx;
        this.ly = ly;
        this.vol = lz * lx * ly;
        this.Coord = Coord;
        this.hold_surface = hold_surface;
    }

    public Double getLength() {return lz;}
    public void setLength(Double length){this.lz = length;}
    public Double getWidth() {return lx;}
    public void setWidth(Double width){this.lx = width;}
    public Double getHeight() {return ly;}
    public Double getZCoor() {return Coord[0];}
    public Double getXCoor() {return Coord[1];}
    public Double getYCoor() {return Coord[2];}
    public Double[] getSize(){
        Double[] return_size = new Double[3];
        return_size[0] = this.lz;
        return_size[1] = this.lx;
        return_size[2] = this.ly;
        return return_size;
    }
    public Double[] getCoord(){return this.Coord;}
    public Area getHold_surface(){return this.hold_surface;}
    public Space getTrans_space(){return this.trans_space;}
}
