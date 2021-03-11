/*
 *    Copyright (c) 2020. Huawei Technologies Co., Ltd.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
 */

package CheckSet.entity;

import CheckSet.util.ParseJsonList;

import java.util.List;
import java.util.Map;


public class Bin {
    /**
     * The length of the bin.
     */
    protected double l;

    /**
     * The width of the bin.
     */
    protected double w;

    /**
     * The height of the bin.
     */
    protected double h;

    /**
     * The type of the bin.
     */
    private String type;

    /**
     * The maximum loading weight of the tray.
     */
    private double maxLoadWeight = Double.MAX_VALUE;

    /**
     * Initialize from a bin map.
     */
    @SuppressWarnings("unchecked")
    public Bin(Map binMap, Map messageMap) {
        String binType = (String) binMap.get("container_type_name");
        Map algorithmBaseParamDto = (Map) messageMap.get("algorithmBaseParamDto");
        List<Map> truckTypeDtoList = ParseJsonList.parseJsonList(algorithmBaseParamDto.get("truckTypeDtoList"), Map.class);
        for (Map truckTypeDto: truckTypeDtoList) {
            String truckTypeName = (String) truckTypeDto.get("truckTypeName");
            if (binType.equals(truckTypeName)) {
                this.l = (Double.parseDouble(truckTypeDto.get("length").toString())) / 10;
                this.w = (Double.parseDouble(truckTypeDto.get("width").toString())) / 10;
                this.h = (Double.parseDouble(truckTypeDto.get("height").toString())) / 10;
                break;
            }
        }
        this.type = (String) binMap.get("container_type_name");
        List<Map> truckTypeMap = ParseJsonList.parseJsonList(algorithmBaseParamDto.get("truckTypeDtoList"), Map.class);
        for (Map truckType: truckTypeMap) {
            if (this.type.equals(truckType.get("truckTypeName"))) {
                this.maxLoadWeight = Double.parseDouble(truckType.get("maxLoad").toString());
                break;
            }
        }
    }

    public double getLength() {
        return this.l;
    }

    public double getWidth() {
        return this.w;
    }

    public double getHeight() {
        return this.h;
    }

    public String getType() {
        return this.type;
    }

    public double getMaxLoadWeight() {
        return this.maxLoadWeight;
    }
}
