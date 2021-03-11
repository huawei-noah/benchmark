package com.quadtalent.quadx.outputEntity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/1/27
 * @mail zhangyan.zy@quadtalent.com
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class LoadTruck {
    private String truckTypeId;
    private String truckTypeCode;
    private int piece;
    private float volume;
    private float weight;
    private float innerLength;
    private float innerWidth;
    private float innerHeight;
    private float maxLoadWeight;
    private List<String> platformArray;
    private List<PackedBox> spuArray;

}
