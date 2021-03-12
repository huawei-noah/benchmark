package com.quadtalent.quadx;

import com.quadtalent.quadx.algrithmDataStructure.Area;
import com.quadtalent.quadx.algrithmDataStructure.Face;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.util.List;

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
public class Container implements Serializable {
    private static final long serialVersionUID = -7708956222565291553L;
    private double length;
    private double width;
    private double maxLoad;
    private double leftLoad;
    private List<List<Area>> usedRect;
    private List<List<Face>> usedFace;
}
