package com.quadtalent.quadx.algrithmDataStructure;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Objects;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/2/2
 * @mail zhangyan.zy@quadtalent.com
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PlatformPair {
    private String from;
    private String to;

    @Override
    public boolean equals(Object obj){
        if(this == obj){    //先判断是否是同一个对象
            return true;
        }

        if(obj == null){    //如果对象为null，返回false
            return false;
        }

        //判断两个对象是否属于同一个类
        if(!(obj instanceof PlatformPair)){
            return false;
        }

        PlatformPair other = (PlatformPair) obj;
        if (from.equals(other.from) && to.equals(other.to)){
            return true;
        }
        return false;
    }

    @Override
    public int hashCode() {
        return Objects.hash(from, to);
    }
}
