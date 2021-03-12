package com.quadtalent.quadx;

import com.quadtalent.quadx.algrithmDataStructure.*;
import com.quadtalent.quadx.inputEntity.Box;
import com.quadtalent.quadx.inputEntity.Truck;
import com.quadtalent.quadx.manager.SizeManager;

import java.util.*;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/1/18
 * @mail zhangyan.zy@quadtalent.com
 */
public class Utils {
    public static Map<String, Box> getBoxMap(List<Box> boxes){
        Map<String,Box> boxMap = new HashMap<>();
        for (Box box:boxes){
            boxMap.put(box.getSpuBoxId(),box);
        }
        return boxMap;
    }
    public static double getBoxTotalWeight(List<Box> boxes){
        double totalWeight = 0;
        for (Box box: boxes){
            totalWeight += box.getWeight();
        }
        return totalWeight;
    }
    public static Map<String,List<Box>> getPlatformBoxes(List<Box> boxes){
        Map<String,List<Box>> platformBoxes = new HashMap<>();
        for (Box box:boxes){
            if(!platformBoxes.containsKey(box.getPlatformCode())){
                List<Box> boxList = new ArrayList<>();
                platformBoxes.put(box.getPlatformCode(),boxList);
            }
            List<Box> boxList = platformBoxes.get(box.getPlatformCode());
            boxList.add(box);
        }
        return platformBoxes;
    }

    public static Truck getMaxLoadTruck(List<Truck> truckList){
        Truck truck = null;
        float maxLoad = 0;
        for (Truck truck1:truckList){
            float tmpLoad = truck1.getWidth() * truck1.getHeight() * truck1.getLength();
            if (tmpLoad > maxLoad){
                truck = truck1;
                maxLoad = tmpLoad;
            }
        }
        return truck;
    }
    public static BoxSizeMapper getSizeBoxList(List<Box> boxes){
        Map<Size,List<Box>> sizeListMap = new HashMap<>();
        Map<UUID,Set<Size>> boxSetMap = new HashMap<>();
        int[] directions = new int[]{0,1};
        for (Box item:boxes){
            Set<Size> sizes = new HashSet<>();
            for (int direction:directions){
                Size tmp = chooseSizeByDirection(item,direction);
                sizes.add(tmp);
            }
            boxSetMap.put(item.getUuid(),sizes);
            for (Size size:sizes){
                if (!sizeListMap.containsKey(size)){
                    List<Box> names = new ArrayList<>();
                    sizeListMap.put(size,names);
                }
                List<Box> names = sizeListMap.get(size);
                names.add(item);
            }
        }
        return new BoxSizeMapper(sizeListMap,boxSetMap);
    }

    public static Size chooseSizeByDirection(Box box, int direction){
        if (direction==0){
            return new Size(box.getLength(),box.getWidth(),box.getHeight());
        }
        else if (direction == 1){
            return new Size(box.getWidth(),box.getLength(),box.getHeight());
        }
        else{
            return null;
        }
    }
    public static int getBoxDirection(Box box,Size size){
        if (box.getLength()==size.getLx()){
            return 0;
        }
        return 1;
    }

    public static SizeManager getSizeManager(Map<Size,List<Box>> sizeListMap, Truck truck, Set<Float> worseThickness, double parameter){
        Map<Float,List<Face>> boxLengthMap = new HashMap<>();
        Map<Float,Double> boxLengthArea = new HashMap<>();
        Set<Size> banSize = new HashSet<>();
        Set<Size> sizes = new HashSet<>();
        for (Size tmp:sizeListMap.keySet()){
            if(banSize.contains(tmp)){
                continue;
            }
            if (tmp.getLy()<=truck.getWidth() && !worseThickness.contains(tmp.getLx())){
                sizes.add(tmp);
            }
            else{
                banSize.add(tmp);
            }
        }
        for (Size size:sizes){
            double area = 0;
            List<Face> faces = new ArrayList<>();
            for(Box item:sizeListMap.get(size)){
                Face face = new Face(item.getUuid(),item.getWeight(),size.getLz(),size.getLy(),item.getPlatformCode());
                faces.add(face);
                area += face.getArea();
            }
            if (!boxLengthArea.containsKey(size.getLx())){
                boxLengthMap.put(size.getLx(),faces);
                boxLengthArea.put(size.getLx(),area);
            }
            else{
                List<Face> originFaces = boxLengthMap.get(size.getLx());
                double originArea = boxLengthArea.get(size.getLx());
                originFaces.addAll(faces);
                boxLengthArea.put(size.getLx(),originArea+area);
            }

        }

        double crossSectionArea = truck.getWidth() * truck.getHeight();
        for (float thickness:boxLengthArea.keySet()){
            if (boxLengthArea.get(thickness)/crossSectionArea<parameter){
                worseThickness.add(thickness);
            }
        }
        return new SizeManager(boxLengthMap,boxLengthArea);
    }
    public static int getMaxNumberOfBoxList(List<Box> boxes,float weight){
        float tmpWeight = 0;
        int number = 0;
        for (Box box:boxes){
            tmpWeight += box.getWeight();
            if (tmpWeight > weight){
                break;
            }
            number += 1;
        }
        return number;
    }

    public static List<SimpleBlock> simpleBlockSizeManager(Map<Size,List<Box>> sizeListMap, Space space, float weight){
        List<SimpleBlock> simpleBlocks = new ArrayList<>();
        float baseX = space.getX();
        float baseY = space.getY();
        float baseZ = space.getZ();
        Set<Size> banSize = new HashSet<>();
        Set<Size> sizes = new HashSet<>();
        for (Size tmp:sizeListMap.keySet()){
            if(banSize.contains(tmp)){
                continue;
            }
            if (tmp.getLx()<=space.getLx() && tmp.getLy()<=space.getLy() && tmp.getLz() <= space.getLz()){
                sizes.add(tmp);
            }
            else{
                banSize.add(tmp);
            }
        }
        for (Size size:sizes){
            List<Box> boxList = sizeListMap.get(size);
            Collections.sort(boxList,new Comparator<Box>() {
                @Override
                public int compare(Box o1,Box o2){
                    return (int) ((o2.getWeight() - o1.getWeight()) * 1000);
                }
            });
            int availableNumber = getMaxNumberOfBoxList(boxList,weight);
            int ny = (int) (space.getLy() / size.getLy());
            int nz = (int) (space.getLz() / size.getLz());
            int nx = (int) (space.getLx() / size.getLx());
//            if (availableNumber <= ny){
//                ny = availableNumber;
//                nz = 1;
//                nx = 1;
//            }
            if (availableNumber <= nz){
                nz = availableNumber;
                ny = 1;
                nx = 1;
            }
//            else if(availableNumber <= ny * nz){
//                nz = availableNumber / ny;
//                nx = 1;
//            }
            else if(availableNumber <= ny * nz){
                ny = availableNumber / nz;
                nx = 1;
            }
            else if(availableNumber <= nx * ny * nz){
                nx = availableNumber / (ny * nz);
            }

            int index = 0;
            List<BoxSpace> spaceList = new ArrayList<>();
            double blockWeight = 0;
            for (int i = 0;i<nz;i++){
                for (int j=0;j<nx;j++){
                    for (int k=0;k<ny;k++){
                        Box item = boxList.get(index);
                        blockWeight += item.getWeight();
                        BoxSpace tmpSpace = new BoxSpace(item.getUuid(),baseX + j*size.getLx(),baseY+k*size.getLy(),baseZ+i*size.getLz(),
                                size.getLx(),size.getLy(),size.getLz(),false,getBoxDirection(item,size),index);
                        index ++;
                        spaceList.add(tmpSpace);
                    }
                }
            }
            List<Box> blockBoxes = new ArrayList<>(boxList.subList(0,index));
            SimpleBlock simpleBlock = new SimpleBlock(nx,ny,nz,size,spaceList,blockBoxes,blockWeight);
            simpleBlocks.add(simpleBlock);
        }
        return simpleBlocks;
    }
    public static boolean canHold(Space space,Size tmp,float width,float length){

        if (tmp.getLz() > space.getLz() || space.x + tmp.getLx() > length || space.y + tmp.getLy() > width){
            return false;
        }
        //这里要在某个尺寸上较小的放置较大的箱子，所以完全包住是不能容忍的！
        else if(tmp.getLx()<=space.getLx() && tmp.getLy()<=space.getLy()){
            return false;
        }
        else if(tmp.getLx()<=space.getLx() && tmp.getLy()>space.getLy()){
            float deltaY = tmp.getLy() - space.getLy();
            if (deltaY < 150 && space.getLy()/tmp.getLy() > 0.8){
                return true;
            }
            else{
                return false;
            }
        }
        else if(tmp.getLx()>space.getLx() && tmp.getLy()<=space.getLy()){
            float deltaX = tmp.getLx() - space.getLx();
            if (deltaX < 150 && space.getLx()/tmp.getLx() > 0.8){
                return true;
            }
            else{
                return false;
            }
        }
        else{
            float deltaX = tmp.getLx() - space.getLx();
            float deltaY = tmp.getLy() - space.getLy();
            if (deltaX < 150 && deltaY < 150 && space.getLx()*space.getLy()/(tmp.getLx()*tmp.getLy())>0.8){
                return true;
            }
            else{
                return false;
            }
        }
    }

    public static boolean overlap(float minCoordY,float minCoordZ,float blockLy,float blockLz,Space wasteSpace){

        if (minCoordY >=wasteSpace.getY()+wasteSpace.getLy() ||
                minCoordY+blockLy<=wasteSpace.getY() ||
                minCoordZ>=wasteSpace.getZ()+wasteSpace.getLz() ||
                minCoordZ+blockLz<=wasteSpace.getZ()){
            return false;
        }
        return true;
    }

    public static boolean overlap3D(float minCoordX,float minCoordY,float minCoordZ,float blockLx,float blockLy,float blockLz,Space wasteSpace){
        if (minCoordX >= wasteSpace.getX() + wasteSpace.getLx() ||
                minCoordX+blockLx<=wasteSpace.getX() ||
                minCoordY >=wasteSpace.getY()+wasteSpace.getLy() ||
                minCoordY+blockLy<=wasteSpace.getY() ||
                minCoordZ>=wasteSpace.getZ()+wasteSpace.getLz() ||
                minCoordZ+blockLz<=wasteSpace.getZ()){
            return false;
        }
        return true;
    }

    public static boolean noOverlapWithOnBoard(List<BoxSpace> usedRects,float minCoordX,float minCoordY,float minCoordZ,float blockLx,float blockLy,float blockLz){
        for (BoxSpace space:usedRects){
            if (overlap3D(minCoordX,minCoordY,minCoordZ,blockLx,blockLy,blockLz,space)){
                return false;
            }
            if (space.x>minCoordX && overlap(minCoordY,minCoordZ,blockLy,blockLz,space)){
                return false;
            }
        }
        return true;
    }

    public static List<SimpleBlock> holdSingleBigBox(Map<Size,List<Box>> sizeListMap, Space space,float width,float length,List<BoxSpace> usedRects){
        List<SimpleBlock> simpleBlocks = new ArrayList<>();
        float baseX = space.getX();
        float baseY = space.getY();
        float baseZ = space.getZ();
        for (Size tmp:sizeListMap.keySet()){
            if (canHold(space,tmp,width,length) && noOverlapWithOnBoard(usedRects,baseX,baseY,baseZ,tmp.getLx(),tmp.getLy(),tmp.getLz())){
                List<Box> boxList = sizeListMap.get(tmp);
                Collections.sort(boxList,new Comparator<Box>() {
                    @Override
                    public int compare(Box o1,Box o2){
                        return (int) ((o2.getWeight() - o1.getWeight()) * 1000);
                    }
                });
                List<BoxSpace> spaceList = new ArrayList<>();
                double blockWeight = 0;
                Box item = boxList.get(0);
                blockWeight += item.getWeight();
                BoxSpace tmpSpace = new BoxSpace(item.getUuid(),baseX ,baseY,baseZ,
                        tmp.getLx(),tmp.getLy(),tmp.getLz(),false,getBoxDirection(item,tmp),0);
                spaceList.add(tmpSpace);
                List<Box> blockBoxes = new ArrayList<>(boxList.subList(0,1));
                SimpleBlock simpleBlock = new SimpleBlock(1,1,1,tmp,spaceList,blockBoxes,blockWeight);
                simpleBlocks.add(simpleBlock);
            }
        }
        return simpleBlocks;
    }

    public static Group layerTransfer(Space space, Layer layer,Map<String,Box> boxMap){
        float baseX = space.getX();
        float baseY = space.getY();
        float baseZ = space.getZ();
        List<BoxSpace> boxSpaceList = new ArrayList<>();
        List<Box> boxList = new ArrayList<>();
        for(int i=0;i<layer.getInsertedBins().size();i++){
            Face tmpFace = layer.getInsertedBins().get(i);
            Area tmpArea = layer.getUsedRects().get(i);
            Box item = boxMap.get(tmpFace.getBinId());
            Size tmpSize = new Size(layer.getLayerThickness(),tmpFace.getFaceWidth(),tmpFace.getFaceLength());
            BoxSpace tmpSpace = new BoxSpace(tmpFace.getBinId(),baseX,baseY+tmpArea.getX(),baseZ+tmpArea.getY(),
                    layer.getLayerThickness(),tmpArea.getWidth(),tmpArea.getLength(),false,getBoxDirection(item,tmpSize),i);
            boxSpaceList.add(tmpSpace);
            boxList.add(item);
        }
        return new Group(boxSpaceList,boxList,layer.getWeight(),layer.getAreaRatio());
    }

    public static Map<PlatformPair,Double> getDistMap(Map<String,Double> distanceMap){
        Map<PlatformPair,Double> distMap = new HashMap<>();
        for (String key:distanceMap.keySet()){
            String[] pair = key.split("\\+");
            PlatformPair platformPair = new PlatformPair(pair[0],pair[1]);
            distMap.put(platformPair,distanceMap.get(key));
        }
        return distMap;
    }
}
