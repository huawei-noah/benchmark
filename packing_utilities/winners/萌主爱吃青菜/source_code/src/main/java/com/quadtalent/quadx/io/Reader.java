package com.quadtalent.quadx.io;

import com.quadtalent.quadx.inputEntity.Box;
import com.quadtalent.quadx.inputEntity.RawInput;
import com.quadtalent.quadx.inputEntity.Truck;

import java.io.File;
import java.util.List;
import java.util.UUID;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/1/27
 * @mail zhangyan.zy@quadtalent.com
 */
public class Reader {
    public static RawInput getRawInput(String path){
//        "E:\\code\\emo-huawei\\dataset\\E1594518281316"
        RawInput rawInput = JacksonUtils.fromFile(new File(path
                        ),
                RawInput.class);
        setBoxUUID(rawInput.getBoxes());
        setTruckMaxLoad(rawInput.getEnv().getTruckTypeDtoList());
        return rawInput;
    }

    public static void setBoxUUID(List<Box> boxList){
        for (Box box:boxList){
            box.setUuid(UUID.randomUUID());
        }
    }

    public static void setTruckMaxLoad(List<Truck> truckList){
        for (Truck truck:truckList){
            truck.setMaxLoad(truck.getMaxLoad()-(float) 0.01);
        }
    }
}
