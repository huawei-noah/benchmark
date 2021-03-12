package com.quadtalent.quadx.manager;

import com.quadtalent.quadx.*;
import com.quadtalent.quadx.algrithmDataStructure.Face;
import com.quadtalent.quadx.algrithmDataStructure.Size;
import com.quadtalent.quadx.inputEntity.Box;
import com.quadtalent.quadx.inputEntity.Truck;

import java.util.*;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/1/21
 * @mail zhangyan.zy@quadtalent.com
 */
public class LayerManager {
    public static List<Layer> getLayerList(List<Box> boxes, Truck truck, Map<String,Box> boxMap, double parameter){
        List<Layer> layers = new ArrayList<>();
        BoxSizeMapper boxSizeMapper = Utils.getSizeBoxList(boxes);
//        Set<String> usedBins = new HashSet<>();
        Set<Float> worseThickness = new HashSet<>();
        SizeManager manager = Utils.getSizeManager(boxSizeMapper.getSizeListMap(),truck,worseThickness,parameter);
        Set<Float> markSet = new HashSet<>();
        markSet.addAll(manager.getBoxLengthArea().keySet());
        markSet.removeAll(worseThickness);
        while (markSet.size()>0){
            for (float thickness:markSet){

                List<Face> faces = manager.getBoxLengthMap().get(thickness);
                Collections.sort(faces,new Comparator<Face>() {
                    @Override
                    public int compare(Face o1,Face o2){
                        return (int) ((o2.getBinWeight() - o1.getBinWeight()) * 100);
                    }
                });
                Layer layer = new Layer("tmpLayer",thickness,truck.getWidth(),truck.getHeight());
                List<Face> leftFaces = layer.insertBottomUp(faces);
                if(layer.getAreaRatio() >= parameter){
                    layers.add(layer);
                    for (Face face:layer.getInsertedBins()){
                        Box box = boxMap.get(face.getBinId());
                        for (Size size:boxSizeMapper.getBoxSetMap().get(face.getBinId())){
                            boxSizeMapper.getSizeListMap().get(size).remove(box);
                        }
                    }
                    break;
                }
                else{
                    worseThickness.add(thickness);
                }
            }
            manager = Utils.getSizeManager(boxSizeMapper.getSizeListMap(),truck,worseThickness,parameter);
            markSet.clear();
            markSet.addAll(manager.getBoxLengthArea().keySet());
            markSet.removeAll(worseThickness);
        }
        return layers;

    }
    public static PlatformLayerManager getPlatformLayers(List<Box> boxes, Truck truck, Map<String,Box> boxMap, double parameter){
        Map<String,List<Box>> platformBoxes = Utils.getPlatformBoxes(boxes);
        Set<Box> usedBins = new HashSet<>();
        Map<String, List<Layer>> platformLayers = new HashMap<>();
        for (String platformId:platformBoxes.keySet()){
            List<Layer> layers = LayerManager.getLayerList(platformBoxes.get(platformId),truck,boxMap,parameter);
            if(layers.size()>0){
                platformLayers.put(platformId,layers);
                for (Layer layer:layers){
                    for (Face face:layer.getInsertedBins()){
                        usedBins.add(boxMap.get(face.getBinId()));
                    }
                }
            }
        }
        return new PlatformLayerManager(platformLayers,usedBins);
    }
}
