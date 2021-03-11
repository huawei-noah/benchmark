package com.quadtalent.quadx.algrithmDataStructure;

import com.quadtalent.quadx.algrithmDataStructure.Area;
import com.quadtalent.quadx.algrithmDataStructure.Face;
import lombok.AllArgsConstructor;
import lombok.Builder;
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
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Info {
    private Area virtualRect;
    private double[] bestOrder;
    private Face chosen;

}
