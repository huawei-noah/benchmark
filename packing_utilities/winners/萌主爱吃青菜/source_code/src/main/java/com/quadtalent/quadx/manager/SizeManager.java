package com.quadtalent.quadx.manager;

import com.quadtalent.quadx.algrithmDataStructure.Face;
import com.quadtalent.quadx.algrithmDataStructure.Size;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/1/19
 * @mail zhangyan.zy@quadtalent.com
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SizeManager {
    private Map<Float,List<Face>> boxLengthMap;
    private Map<Float,Double> boxLengthArea;
}
