package com.quadtalent.quadx.algrithmDataStructure;

import com.quadtalent.quadx.inputEntity.Truck;
import lombok.Data;

import java.util.UUID;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/1/26
 * @mail zhangyan.zy@quadtalent.com
 */
@Data
public class BoxSpace extends Space {
    private UUID binId;
    private int direction;
    private int innerOrder;

    public BoxSpace(UUID binId,float x,float y,float z,float lx,float ly,float lz,boolean vehicleCrossSectionMark,int direction, int innerOrder){
        super(x,y,z,lx,ly,lz,vehicleCrossSectionMark);
        this.binId = binId;
        this.direction = direction;
        this.innerOrder = innerOrder;
    }

    public void changeCoordinate(Truck truck){
        this.x -= truck.getLength()/2;
        this.y -= truck.getWidth()/2;
        this.z -= truck.getHeight()/2;

        this.x += this.lx /2;
        this.y += this.ly/2;
        this.z += this.lz/2;
        float tmp = this.x;
        this.x = this.y;
        this.y = this.z;
        this.z = tmp;
        if (this.direction == 0) {
            this.direction = 100;
        }
        else{
            this.direction = 200;
        }
    }
}
