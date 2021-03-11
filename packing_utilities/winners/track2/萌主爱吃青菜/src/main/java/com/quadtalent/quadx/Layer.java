package com.quadtalent.quadx;
import com.quadtalent.quadx.algrithmDataStructure.Area;
import com.quadtalent.quadx.algrithmDataStructure.Face;
import com.quadtalent.quadx.algrithmDataStructure.Info;
import com.quadtalent.quadx.algrithmDataStructure.Line;
import lombok.Data;

import java.util.*;

/**
 * Copyright © 2021 QuadTalent. All rights reserved.
 * 代码版权声明：深圳坤湛科技有限公司
 * @author 张岩
 * @date 2021/1/20
 * @mail zhangyan.zy@quadtalent.com
 */
@Data
public class Layer {
    private String layerId;
    private float layerThickness;
    private float width;
    private float length;
    private float areaValue;
    private double areaRatio;
    private float weight;
    private List<Area> usedRects;
    private List<Area> freeRects;
    private List<Face> insertedBins;

    public Layer(String layerId,float layerThickness,float width,float length)
    {
        this.layerThickness = layerThickness;
        this.layerId = layerId;
        this.width = width;
        this.length = length;
        Init();
    }

    private void Init()
    {
        weight = 0;
        Area empty = new Area("EmptyArea",0,0,width,length);
        usedRects = new ArrayList<>();
        freeRects = new ArrayList<>();
        insertedBins = new ArrayList<>();
        freeRects.add(empty);
    }

    private Info findBinForSpace(Area tmpRect, List<Face> binList){
        Area virtualRect = new Area("virtual",0,0,0,0);
        Face chosen = null;
        Face tempBin;
        double[] bestOrder = new double[] {Double.MAX_VALUE,Double.MAX_VALUE};
        double bestArea = 0;
        for (int i = 0;i<binList.size();i++){
            tempBin = binList.get(i);
//            if(tempBin.getBinWeight()>weight){
//                continue;
//            }
            double binArea = tempBin.getArea();
            if (tmpRect.getWidth()>=tempBin.getFaceWidth()&&tmpRect.getLength()>=tempBin.getFaceLength()){
                double leftOverX = Math.abs(tmpRect.getWidth() - tempBin.getFaceWidth());
                double leftOverY = Math.abs(tmpRect.getLength() - tempBin.getFaceLength());
                double[] tmpOrder = new double[]{leftOverX,leftOverY};
                Arrays.sort(tmpOrder);
                if (tmpOrder[0]<bestOrder[0] ||
                        (tmpOrder[0]==bestOrder[0] && tmpOrder[1] < bestOrder[1]) ||
                        (tmpOrder[0]==bestOrder[0] && tmpOrder[1] == bestOrder[1] && binArea > bestArea))
                {
                    Area currentRect = getProperArea(tmpRect,tempBin);
                    if (currentRect.getWidth() > 0) {
                        virtualRect = currentRect;
                        bestOrder = tmpOrder;
                        bestArea = binArea;
                        chosen = tempBin;
                    }

                }
            }
        }
        return new Info(virtualRect,bestOrder,chosen);
    }

    private Area getProperArea(Area tmpSpace,Face tmpFace){
        if (tmpSpace.getY()==0){
            return new Area(tmpFace.getBinId().toString(),tmpSpace.getX(),tmpSpace.getY(),tmpFace.getFaceWidth(),tmpFace.getFaceLength());
        }
        else{
            List<Line> supportLines = getSupport(tmpSpace);
            double ratio = 0;
            Area target = new Area("virtual",0,0,0,0);
            for (int i=0;i<supportLines.size();i++){
                Line tmpLine = supportLines.get(i);
                if(tmpLine.getStart()+tmpFace.getFaceWidth()<=tmpSpace.getX()+tmpSpace.getWidth()){
                    Area tmpArea = new Area(tmpFace.getBinId().toString(),tmpLine.getStart(),tmpSpace.getY(),tmpFace.getFaceWidth(),tmpFace.getFaceLength());
                    double tmpRatio = calcSupportRatio(supportLines,tmpArea);
                    if (tmpRatio>=0.8&&tmpRatio>ratio){
                        ratio = tmpRatio;
                        target = tmpArea;
                    }
                }
                if(tmpLine.getEnd()-tmpFace.getFaceWidth()>=tmpSpace.getX()){
                    Area tmpArea = new Area(tmpFace.getBinId().toString(),tmpLine.getEnd()-tmpFace.getFaceWidth(),tmpSpace.getY(),tmpFace.getFaceWidth(),tmpFace.getFaceLength());
                    double tmpRatio = calcSupportRatio(supportLines,tmpArea);
                    if (tmpRatio>=0.8&&tmpRatio>ratio){
                        ratio = tmpRatio;
                        target = tmpArea;
                    }
                }
            }
            return target;
        }
    }

    List<Line> getSupport(Area tmpSpace){
        List<Line> supportLines = new ArrayList<>();
        Area tmpRect;
        for (int i=0;i<usedRects.size();i++){
            tmpRect = usedRects.get(i);
            if(tmpRect.getY()+tmpRect.getLength()==tmpSpace.getY() && tmpRect.getX()+tmpRect.getWidth()>tmpSpace.getX() && tmpSpace.getX()+tmpSpace.getWidth()>tmpRect.getX()){
                float start = Math.max(tmpRect.getX(),tmpSpace.getX());
                float end = Math.min(tmpRect.getX()+tmpRect.getWidth(),tmpSpace.getX()+tmpSpace.getWidth());
                Line tmpLine = new Line(start,end);
                supportLines.add(tmpLine);
            }
        }
        return supportLines;
    }

    double calcSupportRatio(List<Line> supportLines, Area tmpArea){
        double overlap = 0;
        for (int i=0;i<supportLines.size();i++){
            Line tmpLine = supportLines.get(i);
            if (tmpArea.getX()+tmpArea.getWidth()>tmpLine.getStart()&&tmpLine.getEnd()>tmpArea.getX()){
                overlap += Math.min(tmpArea.getX()+tmpArea.getWidth(),tmpLine.getEnd())-Math.max(tmpArea.getX(),tmpLine.getStart());
            }
        }
        return overlap/tmpArea.getWidth();
    }

    private List<Area> splitRectByRect(Area spaceRect, Area binRect){
        List<Area> spaceRectList = new ArrayList<>();
        Area tmpRect;
        if (binRect.getX()>=spaceRect.getX()+spaceRect.getWidth() ||
                binRect.getX()+binRect.getWidth()<=spaceRect.getX() ||
                binRect.getY()>=spaceRect.getY()+spaceRect.getLength() ||
                binRect.getY()+binRect.getLength()<=spaceRect.getY()){
            return spaceRectList;
        }
        if (binRect.getY()>spaceRect.getY()){
            tmpRect = new Area("emptyCon",spaceRect.getX(),spaceRect.getY(),spaceRect.getWidth(),binRect.getY()-spaceRect.getY());
            spaceRectList.add(tmpRect);
        }
        if (binRect.getY() + binRect.getLength() < spaceRect.getY() + spaceRect.getLength()) {
            tmpRect = new Area("emptyCon",spaceRect.getX(),binRect.getY()+binRect.getLength(),spaceRect.getWidth(),spaceRect.getY()+spaceRect.getLength()-(binRect.getY()+binRect.getLength()));
            spaceRectList.add(tmpRect);
        }
        if (binRect.getX() > spaceRect.getX()) {
            tmpRect = new Area("emptyCon",spaceRect.getX(),spaceRect.getY(),binRect.getX()-spaceRect.getX(),spaceRect.getLength());
            spaceRectList.add(tmpRect);
        }
        if (binRect.getX() + binRect.getWidth() < spaceRect.getX() + spaceRect.getWidth()) {
            tmpRect = new Area("emptyCon",binRect.getX()+binRect.getWidth(),spaceRect.getY(),spaceRect.getX()+spaceRect.getWidth()-(binRect.getX()+binRect.getWidth()),spaceRect.getLength());
            spaceRectList.add(tmpRect);
        }
        if (spaceRectList.size() == 0) {
            tmpRect = new Area("zeroCon",binRect.getX(),binRect.getY(),0,0);
            spaceRectList.add(tmpRect);
        }
        return spaceRectList;
    }

    private void placeRect(Area binRect){
        List<Area> pool = new ArrayList<>();
        List<Area> newMaximalSpace = new ArrayList<>();
        Area freeRect;
        for (int i=0;i<freeRects.size();i++){
            freeRect = freeRects.get(i);
            newMaximalSpace = splitRectByRect(freeRect,binRect);
            if (newMaximalSpace.size() > 0) {
                freeRects.remove(i);
                i--;
                pool.addAll(newMaximalSpace);
            }
        }
        freeRects.addAll(pool);
        Area rect1,rect2;
        for (int i=0;i<freeRects.size();i++){
            for (int j=i+1;j<freeRects.size();j++){
                rect1 = freeRects.get(i);
                rect2 = freeRects.get(j);
                if (rect1.isContained(rect2)){
                    freeRects.remove(i);
                    i--;
                    break;
                }
                if (rect2.isContained(rect1)){
                    freeRects.remove(j);
                    j--;
                }
            }
        }
    }

    private boolean spaceAvailable(Area space,List<Face> binList){
        boolean mark = false;

        for (int i=0;i<binList.size();i++){
            Face tmpBin = binList.get(i);
            if (space.getWidth()>=tmpBin.getFaceWidth()&&space.getLength()>=tmpBin.getFaceLength()){
                mark = true;
                break;
            }
        }
        return mark;
    }

    private void filterRect(List<Face> binList){
        for (int i=0;i<freeRects.size();i++){
            Area area = freeRects.get(i);
            if (!spaceAvailable(area,binList)){
                freeRects.remove(i);
                i--;
            }
        }
    }

    public List<Face> insertBottomUp(List<Face> binList){
        List<Face> binListCopy = new ArrayList<>(binList);
        while (binListCopy.size() > 0) {
            if (freeRects.size()==0){
                break;
            }
            Collections.sort(freeRects,new Comparator<Area>() {
                @Override
                public int compare(Area o1,Area o2){
                    return (int) ((o1.getY() - o2.getY()) * 100);
                }
            });
            Area tmpSpace = freeRects.get(0);
            Info pac = findBinForSpace(tmpSpace,binListCopy);
            if (pac.getBestOrder()[0]==Double.MAX_VALUE){
                freeRects.remove(0);
                if(freeRects.size()==0){
                    break;
                }
            }
            else{
                weight += pac.getChosen().getBinWeight();
                insertedBins.add(pac.getChosen());
                placeRect(pac.getVirtualRect());
                usedRects.add(pac.getVirtualRect());
                binListCopy.remove(pac.getChosen());
                filterRect(binListCopy);
            }
        }
        updateRatio();
        return binListCopy;
    }

    private void updateRatio(){
        float sumArea = 0;
        for (Face face:insertedBins){
            sumArea += face.getArea();
        }
        areaValue = sumArea;
        areaRatio = sumArea/(width*length);
    }
}
