package com.quadtalent.quadx.algrithmDataStructure;

import lombok.Data;
import lombok.NoArgsConstructor;

import javax.swing.plaf.metal.MetalBorders;
import java.util.UUID;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/1/19
 * @mail zhangyan.zy@quadtalent.com
 */
@Data
@NoArgsConstructor
public class Face {
    private UUID binId;
    private String platformCode;
    private float faceLength;
    private float faceWidth;
    private float binWeight;
    private double area;

    public Face(UUID binId,float binWeight,float faceLength,float faceWidth,String platformCode){
        this.binId = binId;
        this.binWeight = binWeight;
        this.faceLength = faceLength;
        this.faceWidth = faceWidth;
        this.area = faceLength * faceWidth;
        this.platformCode = platformCode;
    }
}
