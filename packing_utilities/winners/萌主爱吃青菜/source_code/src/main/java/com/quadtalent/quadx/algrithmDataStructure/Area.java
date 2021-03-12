package com.quadtalent.quadx.algrithmDataStructure;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/1/20
 * @mail zhangyan.zy@quadtalent.com
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Area {
    private String binId;
    private float x;
    private float y;
    private float width;
    private float length;

    public boolean isContained(Area obj){
        return (x>=obj.getX()&&y>=obj.getY()&&x+width<=obj.getX()+obj.getWidth()&&y+length<=obj.getY()+obj.getLength());
    }
}
