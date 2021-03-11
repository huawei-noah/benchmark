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

import java.util.Map;


public class BoxInTruck extends Box {

    /**
     * The direction of the box.
     * For rectangles,
     * 100: horizontal with the length of the bin; 200: vertical with the length of the bin.
     * For cylinders,
     * 1: Standing cylinder in "Pin" style; 2: Standing cylinder in "Tian" style; 0: Lying cylinder.
     */
    private int direction;

    /**
     * Initialize from a box map.
     */
    public BoxInTruck(Map boxMap, Map messageMap) {
        super(boxMap, messageMap);
        this.direction = (int) boxMap.get("place");
        this.initLWH();
        double x = Double.parseDouble(boxMap.get("z").toString());
        double y = Double.parseDouble(boxMap.get("x").toString());
        double z = Double.parseDouble(boxMap.get("y").toString());
        this.x1 = x - this.l / 2;
        this.y1 = y - this.w / 2;
        this.z1 = z - this.h / 2;
        this.x2 = x + this.l / 2;
        this.y2 = y + this.w / 2;
        this.z2 = z + this.h / 2;
    }

    /**
     * Initialize the information of the boxes settled.
     */
    private void initLWH() {
        if (this.direction == 100) {
            this.l = this.length;
            this.w = this.width;
        }
        else {
            this.l = this.width;
            this.w = this.length;
        }
        this.h = this.height;
    }

    public int getDirection() {
        return this.direction;
    }
}
