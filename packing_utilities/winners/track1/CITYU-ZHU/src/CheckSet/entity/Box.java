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


public class Box {
    /**
     * The identity of the box.
     */
    protected String id;

    /**
     * The code of the platform the box belonging to.
     */
    protected String platformCode;

    /**
     * The length of the box.
     */
    protected double length;

    /**
     * The width of the box.
     */
    protected double width;

    /**
     * The height of the box.
     */
    protected double height;

    /**
     * The length of the box after rotation.
     */
    protected double l;

    /**
     * The width of the box after rotation.
     */
    protected double w;

    /**
     * The height of the box after rotation.
     */
    protected double h;

    /**
     * The smallest x1 coordinate of the box.
     */
    protected double x1;

    /**
     * The smallest y1 coordinate of the box.
     */
    protected double y1;

    /**
     * The smallest z1 coordinate of the box.
     */
    protected double z1;

    /**
     * The largest x1 coordinate of the box.
     */
    protected double x2;

    /**
     * The largest y1 coordinate of the box.
     */
    protected double y2;

    /**
     * The largest z1 coordinate of the box.
     */
    protected double z2;

    /**
     * The weight of the box.
     */
    protected double weight;

    /**
     * The packing order of the box.
     */
    private int order;

    /**
     * The corresponding box in the input message file.
     */
    protected Map spuBox = null;

    /**
     * Initialize from a box map.
     */
    @SuppressWarnings("unchecked")
    public Box(Map boxMap, Map messageMap) {
        List<Map> boxes = ParseJsonList.parseJsonList(messageMap.get("boxes"), Map.class);
        this.id = (String) boxMap.get("boxId");
        this.order = (int) boxMap.get("order");
        if (boxes != null) {
            for (Map spuBox: boxes) {
                if (spuBox.get("spuBoxId").equals(this.id)) {
                    this.spuBox = spuBox;
                    break;
                }
            }
        }
        this.platformCode = (String) boxMap.get("platform");
        this.length = Double.parseDouble(boxMap.get("length").toString());
        this.width = Double.parseDouble(boxMap.get("width").toString());
        this.height = Double.parseDouble(boxMap.get("height").toString());
        this.l = this.length;
        this.w = this.width;
        this.h = this.height;
        double x = Double.parseDouble(boxMap.get("z").toString());
        double y = Double.parseDouble(boxMap.get("x").toString());
        double z = Double.parseDouble(boxMap.get("y").toString());
        this.x1 = x - this.l / 2;
        this.y1 = y - this.w / 2;
        this.z1 = z - this.h / 2;
        this.x2 = x + this.l / 2;
        this.y2 = y + this.w / 2;
        this.z2 = z + this.h / 2;
        this.weight = Double.parseDouble(boxMap.get("weight").toString());
    }

    /**
     * Initialize from a SPUBoxDto object.
     */
    public Box(Map estimateBox) {
        this.id = (String) estimateBox.get("spuBoxId");
        this.platformCode = (String) estimateBox.get("platformCode");
        this.length = Double.parseDouble(estimateBox.get("length").toString());
        this.width = Double.parseDouble(estimateBox.get("width").toString());
        this.height = Double.parseDouble(estimateBox.get("height").toString());
        this.weight = Double.parseDouble(estimateBox.get("weight").toString());
    }

    public void resetCoordinate(Map binMap) {
        double extraLength = Double.parseDouble(binMap.get("container_length").toString()) / 2;
        double extraWidth = Double.parseDouble(binMap.get("container_width").toString()) / 2;
        double extraHeight = Double.parseDouble(binMap.get("container_height").toString()) / 2;
        this.x1 += extraLength;
        this.y1 += extraWidth;
        this.z1 += extraHeight;
        this.x2 += extraLength;
        this.y2 += extraWidth;
        this.z2 += extraHeight;
    }

    public String getId() {
        return this.id;
    }

    public String getPlatformCode() {
        return this.platformCode;
    }

    public double getLength() {
        return this.length;
    }

    public double getWidth() {
        return this.width;
    }

    public double getHeight() {
        return this.height;
    }

    public double getL() {
        return this.l;
    }

    public double getW() {
        return this.w;
    }

    public double getH() {
        return this.h;
    }

    public double getX1() {
        return this.x1;
    }

    public double getY1() {
        return this.y1;
    }

    public double getZ1() {
        return this.z1;
    }

    public double getX2() {
        return this.x2;
    }

    public double getY2() {
        return this.y2;
    }

    public double getZ2() {
        return this.z2;
    }

    public double getWeight() {
        return this.weight;
    }

    public int getOrder() {
        return this.order;
    }
}
