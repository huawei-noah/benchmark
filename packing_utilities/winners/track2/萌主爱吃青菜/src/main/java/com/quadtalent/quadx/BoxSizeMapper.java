package com.quadtalent.quadx;

import com.quadtalent.quadx.algrithmDataStructure.Size;
import com.quadtalent.quadx.inputEntity.Box;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/1/26
 * @mail zhangyan.zy@quadtalent.com
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class BoxSizeMapper {
    private Map<Size, List<Box>> sizeListMap;
    private Map<UUID, Set<Size>> boxSetMap;
}
