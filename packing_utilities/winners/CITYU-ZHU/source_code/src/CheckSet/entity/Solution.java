package CheckSet.entity;

import java.util.Map;

public class Solution {
    private Map<Integer, Bin> allBins;
    private Map<Integer, Map<Integer, BoxInTruck>> allBoxesInTruck;
    private int index;

    public Solution(Map<Integer, Bin> allBins, Map<Integer, Map<Integer, BoxInTruck>> allBoxesInTruck, int index) {
        this.allBins = allBins;
        this.allBoxesInTruck = allBoxesInTruck;
        this.index = index;
    }

    public Map<Integer, Bin> getAllBins() {
        return allBins;
    }

    public Map<Integer, Map<Integer, BoxInTruck>> getAllBoxesInTruck() {
        return allBoxesInTruck;
    }

    public int getIndex() {
        return index;
    }
}
