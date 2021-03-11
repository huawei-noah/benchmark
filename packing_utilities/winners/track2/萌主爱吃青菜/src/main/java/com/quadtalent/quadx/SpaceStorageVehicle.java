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
 * @date 2021/2/3
 * @mail zhangyan.zy@quadtalent.com
 */
@Data
public class SpaceStorageVehicle {
    private String vehicleId;
    private float height;
    private float width;
    private float length;
    private float volume;
    private float areaRatio;
    private float weight;
    private List<BoxSpace> usedRects;
    private List<Space> freeRects;
    private List<Space> storageSpaces;
    private List<Box> insertedBins;
    private int order;
    private Truck truck;

    public SpaceStorageVehicle(String vehicleId,Truck truck)
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
        storageSpaces = new ArrayList<>();
        freeRects.add(empty);
    }

    private SimpleBlock findBlockForSpace(Space space, Map<Size,List<Box>> sizeListMap){
        List<SimpleBlock> availableBlocks = Utils.simpleBlockSizeManager(sizeListMap,space,weight);
        return chooseFromPool(availableBlocks);
    }

    private SimpleBlock chooseFromPool(List<SimpleBlock> availableBlocks){
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
            else if (chosen!=null && blockVolume == bestVolume && (block.getSize().getLy() > chosen.getSize().getLy())){
                chosen = block;
            }
        }
        return chosen;
    }

    private SimpleBlock findOverSizeBlockForSpace(Space space, Map<Size,List<Box>> sizeListMap){
        List<SimpleBlock> availableBlocks = Utils.holdSingleBigBox(sizeListMap,space,width,length,usedRects);
        return chooseFromPool(availableBlocks);
    }



    private void updateSpace(float minCoordY,float blockLy,Space wasteSpace){
        float tmpLy = wasteSpace.ly - (minCoordY + blockLy - wasteSpace.y);
        wasteSpace.ly = tmpLy;
        wasteSpace.y = minCoordY + blockLy;
    }

    private List<Space> updateOverlapSpaces(Space tmpSpace,SimpleBlock chosen){
        float minCoordY = tmpSpace.getY();
        float minCoordZ = tmpSpace.getZ();
        float blockLy = chosen.getNy() * chosen.getSize().getLy();
        float blockLz = chosen.getNz() * chosen.getSize().getLz();
        List<Space> refreshSpaces = new ArrayList<>();
        for (int i=0;i<storageSpaces.size();i++){
            Space space = storageSpaces.get(i);
            if (Utils.overlap(minCoordY,minCoordZ,blockLy,blockLz,space)){
                updateSpace(minCoordY,blockLy,space);
                if (space.ly <= 0){
                    storageSpaces.remove(i);
                    i--;
                }
                else{
                    refreshSpaces.add(space);
                }
            }
        }
        return refreshSpaces;
    }

    private void updateEnqueueSpaces(float minCoordX,float minCoordY,float minCoordZ,float blockLx,float blockLy,float blockLz){
        List<Space> available = new ArrayList<>();
        available.addAll(freeRects);
        available.addAll(storageSpaces);
        for (int i=0;i<available.size();i++) {
            Space space = available.get(i);
            if (Utils.overlap3D(minCoordX,minCoordY,minCoordZ,blockLx,blockLy,blockLz,space)){
                float deltaY = minCoordY + blockLy - space.y;
                float deltaX = minCoordX + blockLx - space.x;
                if (deltaY < Math.min(150,blockLy/0.8 - blockLy)){
                    float tmpLy = space.ly - deltaY;
                    space.ly = tmpLy;
                    space.y = minCoordY + blockLy;
                }
                if (deltaX < Math.min(150,blockLx/0.8 - blockLx)){
                    float tmpLx = space.lx - deltaX;
                    space.lx = tmpLx;
                    space.x = minCoordX + blockLx;
                }
                if (space.ly <= 0 || space.lx<=0){
                    if (freeRects.contains(space)){
                        freeRects.remove(space);
                    }
                    if (storageSpaces.contains(space)){
                        storageSpaces.remove(space);
                    }
                }
            }
            if (space.x<minCoordX && Utils.overlap(minCoordY,minCoordZ,blockLy,blockLz,space)){
                float deltaY = minCoordY + blockLy - space.y;
                if (deltaY < Math.min(150,blockLy/0.8 - blockLy)){
                    float tmpLy = space.ly - deltaY;
                    space.ly = tmpLy;
                    space.y = minCoordY + blockLy;
                }
                if (space.ly <= 0){
                    if (freeRects.contains(space)){
                        freeRects.remove(space);
                    }
                    if (storageSpaces.contains(space)){
                        storageSpaces.remove(space);
                    }
                }
            }
        }
    }

    private Space merge(List<Space> refreshSpaces,Space targetSpace){
        Space available = null;
        for (Space space:refreshSpaces){
            if (space.x+space.lx==targetSpace.x && space.y == targetSpace.y && space.z==targetSpace.z && space.ly == targetSpace.ly && space.lz == targetSpace.lz){
                available = space;
                break;
            }
        }
        return available;
    }

    private List<BoxSpace> findOverlapRects(Space space){
        List<BoxSpace> overlapRects = new ArrayList<>();
        for (BoxSpace boxSpace:usedRects){
            if (Utils.overlap3D(space.x,space.y,space.z,space.lx,space.ly,space.lz,boxSpace)){
                overlapRects.add(boxSpace);
            }
        }
        return overlapRects;
    }

    private void placeRect(Space tmpSpace,SimpleBlock chosen,boolean overSizeMark){
        freeRects.remove(tmpSpace);
        List<Space> refreshSpaces = updateOverlapSpaces(tmpSpace,chosen);
        float minCoordX = tmpSpace.getX();
        float minCoordY = tmpSpace.getY();
        float minCoordZ = tmpSpace.getZ();
        float blockLx = chosen.getNx() * chosen.getSize().getLx();
        float blockLy = chosen.getNy() * chosen.getSize().getLy();
        float blockLz = chosen.getNz() * chosen.getSize().getLz();
        float deltaX = tmpSpace.getLx() - blockLx;
        float deltaY = tmpSpace.getLy() - blockLy;
        float deltaZ = tmpSpace.getLz() - blockLz;

        updateEnqueueSpaces(minCoordX,minCoordY,minCoordZ,blockLx,blockLy,blockLz);

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
            Space space2 = new Space(minCoordX,minCoordY+blockLy,minCoordZ,Math.min(blockLx,tmpSpace.getLx()),deltaY,tmpSpace.getLz(),false);
            if (refreshSpaces.size()>0){
                Space available = merge(refreshSpaces,space2);
                if(available!=null){
                    space2.x = available.x;
                    space2.lx = space2.lx + available.lx;
                    storageSpaces.remove(available);
                }
            }
            freeRects.add(space2);
        }
        if (deltaZ > 0){
            Space space3 = new Space(minCoordX,minCoordY,minCoordZ+blockLz,blockLx,blockLy,deltaZ,false);
            if (overSizeMark){
                List<BoxSpace> overlapRects = findOverlapRects(space3);
                if (overlapRects.size()>0){
                    float minY = Float.POSITIVE_INFINITY;
                    for (BoxSpace boxSpace:overlapRects){
                        if (boxSpace.y<minY){
                            minY = boxSpace.y;
                        }
                    }
                    space3.ly = minY - space3.y;
                }
            }
            freeRects.add(space3);
        }
//        System.out.println("debug");
    }

    public List<Box> insertBlock(List<Box> binList,boolean updateMark){
        List<Box> binListCopy = new ArrayList<>(binList);
        BoxSizeMapper boxSizeMapper = Utils.getSizeBoxList(binListCopy);
        while (binListCopy.size() > 0) {
            if (freeRects.size()==0){
                break;
            }
            if (updateMark) {
                freeRects.addAll(storageSpaces);
                storageSpaces = new ArrayList<>();
                updateMark = false;
            }
            Space tmpSpace = freeRects.get(freeRects.size()-1);
            SimpleBlock chosen = findBlockForSpace(tmpSpace,boxSizeMapper.getSizeListMap());
            SimpleBlock overSizeChosen = findOverSizeBlockForSpace(tmpSpace,boxSizeMapper.getSizeListMap());
            if (chosen == null && overSizeChosen == null){
                freeRects.remove(tmpSpace);
                storageSpaces.add(0,tmpSpace);
                if(freeRects.size()==0){
                    break;
                }
            }
            else if (chosen == null && overSizeChosen != null){
                weight -= overSizeChosen.getBlockWeight();
                order = overSizeChosen.updateBoxOrder(order);
                insertedBins.addAll(overSizeChosen.getBoxes());
                usedRects.addAll(overSizeChosen.getElements());
                placeRect(tmpSpace, overSizeChosen,true);
                updateBoxSizeMapper(binListCopy, boxSizeMapper, overSizeChosen);
            }
            else if (chosen != null && overSizeChosen == null){
                weight -= chosen.getBlockWeight();
                order = chosen.updateBoxOrder(order);
                insertedBins.addAll(chosen.getBoxes());
                usedRects.addAll(chosen.getElements());
                placeRect(tmpSpace, chosen,false);
                updateBoxSizeMapper(binListCopy, boxSizeMapper, chosen);
            }
            else{
                if (chosen.getBlockVolume()>overSizeChosen.getBlockVolume()){
                    weight -= chosen.getBlockWeight();
                    order = chosen.updateBoxOrder(order);
                    insertedBins.addAll(chosen.getBoxes());
                    usedRects.addAll(chosen.getElements());
                    placeRect(tmpSpace, chosen,false);
                    updateBoxSizeMapper(binListCopy, boxSizeMapper, chosen);
                }
                else{
                    weight -= overSizeChosen.getBlockWeight();
                    order = overSizeChosen.updateBoxOrder(order);
                    insertedBins.addAll(overSizeChosen.getBoxes());
                    usedRects.addAll(overSizeChosen.getElements());
                    placeRect(tmpSpace, overSizeChosen,true);
                    updateBoxSizeMapper(binListCopy, boxSizeMapper, overSizeChosen);
                }

            }
        }
        updateRatio();
        return binListCopy;
    }

    private void updateBoxSizeMapper(List<Box> binListCopy, BoxSizeMapper boxSizeMapper, Group chosen) {
        binListCopy.removeAll(chosen.getBoxes());
        Box sample = chosen.getBoxes().get(0);
        Set<Size> sizeSet = boxSizeMapper.getBoxSetMap().get(sample.getUuid());
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

    public List<String> getPlatformList(){
        List<String> platformNameList = new ArrayList<>();
        for (Box tmpBox:insertedBins){
            if (!platformNameList.contains(tmpBox.getPlatformCode())){
                platformNameList.add(tmpBox.getPlatformCode());
            }
        }
        return platformNameList;
    }

}
