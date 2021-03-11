package com.quadtalent.quadx.algrithmDataStructure;

import com.quadtalent.quadx.algrithmDataStructure.BoxSpace;
import com.quadtalent.quadx.algrithmDataStructure.Size;
import com.quadtalent.quadx.inputEntity.Box;
import lombok.Data;

import java.util.List;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/1/25
 * @mail zhangyan.zy@quadtalent.com
 */
@Data
public class SimpleBlock extends Group{
    private int nx;
    private int ny;
    private int nz;
    private Size size;


    public SimpleBlock(int nx,int ny,int nz,Size size,List<BoxSpace> elements,List<Box> boxes,double blockWeight){
        super(elements,boxes,blockWeight,nx * ny * nz * size.getLx() * size.getLy() * size.getLz());
        this.nx = nx;
        this.ny = ny;
        this.nz = nz;
        this.size = size;
    }


}
