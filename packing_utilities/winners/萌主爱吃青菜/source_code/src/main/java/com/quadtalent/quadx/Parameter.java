package com.quadtalent.quadx;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/2/27
 * @mail zhangyan.zy@quadtalent.com
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Parameter {
    public int totalIterationTimes;
    public int treeStart;
    public long timeLimitation;
}
