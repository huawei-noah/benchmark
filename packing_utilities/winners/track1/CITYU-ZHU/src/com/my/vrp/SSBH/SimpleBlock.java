package com.my.vrp.SSBH;

/**
 * 用于拼接箱子的简单块
 */
public class SimpleBlock {
    private Integer nz;
    private Integer nx;
    private Integer ny;
    private Double[] item_size;
    private Double[] size;
    private Integer box_num;//箱子的总个数
    private Double vol;//块的总体积
    private Double lz;
    private Double lx;
    private Double ly;
    private Area hold_surface;
    /**
     * @param nz z方向箱子个数，即长方向
     * @param nx x方向箱子个数，即宽方向
     * @param ny y方向箱子个数，即高方向
     * @param sizes 每一个组成块的箱子的种类
     */
    public SimpleBlock(Integer nz, Integer nx, Integer ny, Double[] sizes){
        //仍然是长宽高的顺序来处理
        this.nz = nz;
        this.nx = nx;
        this.ny = ny;
        this.item_size = sizes;
        this.box_num = nz * nx * ny;
        this.vol = this.box_num * sizes[0] * sizes[1] * sizes[2];
        this.lz = sizes[0] * nz;
        this.lx = sizes[1] * nx;
        this.ly = sizes[2] * ny;
        this.size = new Double[3];
        this.size[0] = this.lz;
        this.size[1] = this.lx;
        this.size[2] = this.ly;
    }
    public Double getLength(){return this.lz;}
    public Double getWidth(){return this.lx;}
    public Double getHeight(){return this.ly;}
    public Double[] getItemSizes(){return this.item_size;}
    public Double[] getSize(){return this.size;}

    public Integer getBox_num() { return box_num; }
    public String getStringsize() {
        String to_return = new String();
        to_return = this.item_size[0].toString()+"+"+this.item_size[1].toString()+"+"+this.item_size[2].toString();
        return to_return;
    }
    public Integer getNz(){return this.nz;}
    public Integer getNx(){return this.nx;}
    public Integer getNy(){return this.ny;}
    public Area getHold_surface(){return this.hold_surface;}

    public void setHold_surface(Space space_using) {
        this.hold_surface = new Area();
        this.hold_surface.setLength(this.lz);
        this.hold_surface.setWidth(this.lx);
        this.hold_surface.setCoords(space_using.getCoord());
    }
}
