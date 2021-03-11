package com.quadtalent.quadx.algrithmDataStructure;

import com.quadtalent.quadx.inputEntity.Box;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/1/29
 * @mail zhangyan.zy@quadtalent.com
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Group {
    public List<BoxSpace> elements;
    public List<Box> boxes;
    public double blockWeight;
    public double blockVolume;

    public int updateBoxOrder(int originOrder){
        for (BoxSpace boxSpace:elements){
            boxSpace.setInnerOrder(boxSpace.getInnerOrder()+originOrder);
        }
        return originOrder + elements.size();
    }
}
