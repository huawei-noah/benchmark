package com.quadtalent.quadx;

import com.quadtalent.quadx.algrithmDataStructure.*;
import com.quadtalent.quadx.inputEntity.Box;
import com.quadtalent.quadx.inputEntity.Truck;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/1/22
 * @mail zhangyan.zy@quadtalent.com
 */
@Data
public class Vehicle {
    private String vehicleId;
    private float height;
    private float width;
    private float length;
    private float volume;
    private double areaRatio;
    private float weight;
    private List<BoxSpace> usedRects;
    private List<Space> freeRects;
    private List<Box> insertedBins;
    private int order;
    private Truck truck;

    public Vehicle(String vehicleId,Truck truck)
    {

        this.vehicleId = vehicleId;
        this.truck = truck;
        this.length = truck.getLength();
        this.width = truck.getWidth();
        this.height = truck.getHeight();
        this.weight = truck.getMaxLoad();
        Init();
    }

    private void Init()
    {
        order = 1;
        Space empty = new Space(0,0,0,length,width,height,true);
        usedRects = new ArrayList<>();
        freeRects = new ArrayList<>();
        insertedBins = new ArrayList<>();
        freeRects.add(empty);
    }

    private SimpleBlock findBlockForSpace(Space space, Map<Size,List<Box>> sizeListMap){
        List<SimpleBlock> availableBlocks = Utils.simpleBlockSizeManager(sizeListMap,space,weight);
        double bestVolume = 0;
        SimpleBlock chosen = null;
        for (SimpleBlock block:availableBlocks){
            if(block.getBlockWeight()>weight){
                continue;
            }

            double blockVolume = block.getBlockVolume();
            if (blockVolume > bestVolume){
                bestVolume = blockVolume;
                chosen = block;
            }
        }
        return chosen;
    }

    private Group findLayerGroupForSpace(Space space,List<Layer> layers,Map<String,Box> boxMap){
        Layer targetLayer = null;
        for (Layer layer:layers){
            if(layer.getWeight() > weight){
                continue;
            }
            if (layer.getLayerThickness()<=space.getLx()){
                targetLayer = layer;
                break;
            }
        }
        if (targetLayer==null){
            return null;
        }
        else{
            layers.remove(targetLayer);
            return Utils.layerTransfer(space,targetLayer,boxMap);
        }
    }
    private void placeRect(Space tmpSpace,SimpleBlock chosen){
        freeRects.remove(tmpSpace);
        float minCoordX = tmpSpace.getX();
        float minCoordY = tmpSpace.getY();
        float minCoordZ = tmpSpace.getZ();
        float blockLx = chosen.getNx() * chosen.getSize().getLx();
        float blockLy = chosen.getNy() * chosen.getSize().getLy();
        float blockLz = chosen.getNz() * chosen.getSize().getLz();
        float deltaX = tmpSpace.getLx() - blockLx;
        float deltaY = tmpSpace.getLy() - blockLy;
        float deltaZ = tmpSpace.getLz() - blockLz;
        if (deltaX > 0) {
            Space space1 = null;
            if(tmpSpace.isVehicleCrossSectionMark()){
                space1 = new Space(minCoordX+blockLx,minCoordY,minCoordZ,deltaX,tmpSpace.getLy(),tmpSpace.getLz(),true);
            }
            else{
                space1 = new Space(minCoordX+blockLx,minCoordY,minCoordZ,deltaX,tmpSpace.getLy(),tmpSpace.getLz(),false);
            }
            freeRects.add(space1);
        }
        if (deltaY > 0){
            Space space2 = new Space(minCoordX,minCoordY+blockLy,minCoordZ,blockLx,deltaY,tmpSpace.getLz(),false);
            freeRects.add(space2);
        }
        if (deltaZ > 0){
            Space space3 = new Space(minCoordX,minCoordY,minCoordZ+blockLz,blockLx,blockLy,deltaZ,false);
            freeRects.add(space3);
        }
    }

    private void placeLayer(Space tmpSpace,Group chosen){
        freeRects.remove(tmpSpace);
        BoxSpace sample = chosen.elements.get(0);
        Space newSpace = new Space(tmpSpace.getX()+sample.lx,tmpSpace.getY(),tmpSpace.getZ(),tmpSpace.lx-sample.lx,tmpSpace.ly,tmpSpace.lz,true);
        freeRects.add(newSpace);

    }

    public List<Box> insertBlock(List<Box> binList){
        List<Box> binListCopy = new ArrayList<>(binList);
        BoxSizeMapper boxSizeMapper = Utils.getSizeBoxList(binListCopy);
        while (binListCopy.size() > 0) {
            if (freeRects.size()==0){
                break;
            }
            Space tmpSpace = freeRects.get(freeRects.size()-1);
            SimpleBlock chosen = findBlockForSpace(tmpSpace,boxSizeMapper.getSizeListMap());
            if (chosen == null) {
                freeRects.remove(tmpSpace);
                if(freeRects.size()==0){
                    break;
                }
            }
            else{
                weight -= chosen.getBlockWeight();
                order = chosen.updateBoxOrder(order);
                insertedBins.addAll(chosen.getBoxes());
                usedRects.addAll(chosen.getElements());
                placeRect(tmpSpace, chosen);
                updateBoxSizeMapper(binListCopy, boxSizeMapper, chosen);
            }
        }
        updateRatio();
        return binListCopy;
    }

    public List<Box> insertLayer(List<Box> binList,List<Layer> layers,Map<String,Box> boxMap){
        List<Box> binListCopy = new ArrayList<>(binList);
        BoxSizeMapper boxSizeMapper = Utils.getSizeBoxList(binListCopy);
        boolean layerMark = false;
        if (layers!=null && layers.size()>0){
            layerMark = true;
        }
        while (binListCopy.size() > 0 || layerMark) {
            if (freeRects.size()==0){
                break;
            }
            Space tmpSpace = freeRects.get(freeRects.size()-1);
            Group chosen = null;
            if (layers !=null && tmpSpace.vehicleCrossSectionMark && layers.size()>0){
                chosen = findLayerGroupForSpace(tmpSpace,layers,boxMap);
                if (chosen==null){
                    chosen = findBlockForSpace(tmpSpace,boxSizeMapper.getSizeListMap());
                }
                else{
                    if (layers.size()==0){
                        layerMark = false;
                    }
                }
            }
            else{
                chosen = findBlockForSpace(tmpSpace,boxSizeMapper.getSizeListMap());
            }
            if (chosen == null) {
                freeRects.remove(tmpSpace);
                if(freeRects.size()==0){
                    break;
                }
            }
            else{
                weight -= chosen.getBlockWeight();
                order = chosen.updateBoxOrder(order);
                insertedBins.addAll(chosen.getBoxes());
                usedRects.addAll(chosen.getElements());
                if (chosen instanceof SimpleBlock) {
                    placeRect(tmpSpace,(SimpleBlock) chosen);
                    updateBoxSizeMapper(binListCopy, boxSizeMapper, chosen);
                }
                else{
                    placeLayer(tmpSpace,chosen);
                }


            }
        }
        updateRatio();
        return binListCopy;
    }

    private void updateBoxSizeMapper(List<Box> binListCopy, BoxSizeMapper boxSizeMapper, Group chosen) {
        binListCopy.removeAll(chosen.getBoxes());
        Box sample = chosen.getBoxes().get(0);
        Set<Size> sizeSet = boxSizeMapper.getBoxSetMap().get(sample.getSpuBoxId());
        for (Size size:sizeSet){
            List<Box> tmpBoxList = boxSizeMapper.getSizeListMap().get(size);
            tmpBoxList.removeAll(chosen.getBoxes());
            if (tmpBoxList.size()==0){
                boxSizeMapper.getSizeListMap().remove(size);
            }
        }
    }

    private void updateRatio(){
        float sumArea = 0;
        for (Box box:insertedBins){
            sumArea += box.getLength() * box.getWidth() * box.getHeight();
        }
        volume = sumArea;
        areaRatio = sumArea/(width*length*height);
    }


}
