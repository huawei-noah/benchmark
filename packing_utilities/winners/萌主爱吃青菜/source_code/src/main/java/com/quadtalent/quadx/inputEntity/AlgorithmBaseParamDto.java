package com.quadtalent.quadx.inputEntity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2020/11/25
 * @mail zhangyan.zy@quadtalent.com
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AlgorithmBaseParamDto implements Serializable {
    private List<Platform> platformDtoList;
    private List<Truck> truckTypeDtoList;
    private Map<String,Truck> truckTypeMap;
    private Map<String,Double> distanceMap;

}
