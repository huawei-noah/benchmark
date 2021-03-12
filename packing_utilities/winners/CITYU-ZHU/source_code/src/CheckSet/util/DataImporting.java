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

package CheckSet.util;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.TypeReference;

import CheckSet.entity.*;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;


public class DataImporting {
    /**
     * All input information read from an input message file.
     */
    private Map messageMap;

    /**
     * Initialize the message map.
     * @param message: the path of an input message file.
     */
    public DataImporting(String message) throws IOException {
        this.messageMap = fromJson(new String(Files.readAllBytes(Paths.get(message))), new TypeReference<Map>(){});
    }

    /**
     * Generate boxes from a html file.
     */
    public Map<Integer, BoxInTruck> genBoxesInTruck(String htmlFile) throws IOException {
        // In the html file, the variable jsonstr is the list of box maps.
        String boxesStr = htmlParse(htmlFile, "jsonstr");
        ArrayList<Map> boxesMap = fromJson(boxesStr, new TypeReference<ArrayList<Map>>(){});
        String binStr = htmlParse(htmlFile, "jsonstr2");
        Map<String, Object> binMap = fromJson(binStr, new TypeReference<Map>(){});
        Map<Integer, BoxInTruck> boxes = new HashMap<>();
        for (Map boxMap: boxesMap) {
            BoxInTruck boxInTruck = new BoxInTruck(boxMap, this.messageMap);
            boxInTruck.resetCoordinate(binMap);
            boxes.put(boxInTruck.getOrder(), boxInTruck);
        }
        return boxes;
    }

    /**
     * Generate boxes in all bins from the directory of a html files.
     */
    public Map<Integer, Map<Integer, BoxInTruck>> genAllBoxesInTruck(String directory) throws IOException {
        File dir = new File(directory);
        Map<Integer, Map<Integer, BoxInTruck>> allBoxes = new HashMap<>();
        if (dir.listFiles() != null){
            for(File file: Objects.requireNonNull(dir.listFiles())) {
                String fileStr = file.getPath();
                Path path = Paths.get(fileStr);
                if (fileStr.contains(".html")) {
                    int index = getIndex(file.getName());
                    Map<Integer, BoxInTruck> boxMap = genBoxesInTruck(new String(Files.readAllBytes(path)));
                    allBoxes.put(index, boxMap);
                }
            }
        }
        return allBoxes;
    }

    /**
     * Generate a bin from a html file and a list of truck objects.
     */
    public Bin genBin(String htmlFile) throws IOException {
        // In the html file, the variable jsonstr2 is the bin map.
        String binStr = htmlParse(htmlFile, "jsonstr2");
        Map<String, Object> binMap = fromJson(binStr, new TypeReference<Map<String, Object>>(){});
        return new Bin(binMap, this.messageMap);
    }

    /**
     * Generate all bins from the directory of a html files and the path of a message file.
     */
    public Map<Integer, Bin> genAllBins(String directory) throws IOException {
        File dir = new File(directory);
        Map<Integer, Bin> allBins = new HashMap<>();
        if (dir.listFiles() != null){
            for(File file: Objects.requireNonNull(dir.listFiles())) {
                String fileStr = file.getPath();
                Path path = Paths.get(fileStr);
                if (fileStr.contains(".html")) {
                    int index = getIndex(file.getName());
                    allBins.put(index, genBin(new String(Files.readAllBytes(path))));
                }
            }
        }
        return allBins;
    }

    /**
     * Generate all bins and boxes in all solutions.
     */
    public ArrayList<Solution> genAllSolutions(String directory) throws IOException {
        ArrayList<Solution> solutions = new ArrayList<>();
        File dir = new File(directory);
        if (dir.listFiles() != null) {
            int solutionIndex = 1;
            for (File file: Objects.requireNonNull(dir.listFiles())) {
                if (file.isDirectory()) {
                    Map<Integer, Bin> allBins = genAllBins(file.getPath());
                    Map<Integer, Map<Integer, BoxInTruck>> allBoxesInTruck =  genAllBoxesInTruck(file.getPath());
                    solutions.add(new Solution(allBins, allBoxesInTruck, solutionIndex));
                    solutionIndex++;
                }
            }
        }
        return solutions;
    }

    /**
     * Generate all boxes of the order from an input file.
     */
    @SuppressWarnings("unchecked")
    public ArrayList<Box> genInputBoxes() {
        List<Map> boxesToBePacked = ParseJsonList.parseJsonList(messageMap.get("boxes"), Map.class);
        ArrayList<Box> boxes = new ArrayList<>();
        if (boxesToBePacked != null) {
            for (Map estimateBox: boxesToBePacked) {
                Box box = new Box(estimateBox);
                boxes.add(box);
            }
        }
        return boxes;
    }

    /**
     * Generate a route parameter map from an input file.
     */
    public Route genRoute() {
        return new Route(this.messageMap);
    }

    /**
     * Parse a html file and extract the variables of boxes or a bin.
     * @param option: the name of the variable to be matched.
     */
    private static String htmlParse(String htmlFile, String option) {
        Document doc = Jsoup.parse(htmlFile);
        Elements scripts = doc.select("script");
        for (Element script: scripts) {
            if (script.html().contains("var " + option)) {
                String scriptStr = script.html();
                // Match and return the content of the variables.
                String pattern = "var " + option + " =\\s?'(.*)'";
                Pattern r = Pattern.compile(pattern);
                Matcher m = r.matcher(scriptStr);
                if (m.find()) {
                    return m.group(1);
                }
                else {
                    throw new IllegalArgumentException("Do not find " + option + " in the html file.");
                }
            }
        }
        throw new IllegalArgumentException("Do not find " + option + " in the html file.");
    }

    /**
     * Transform a json string to an object.
     */
    private static <T> T fromJson(String jsonStr, TypeReference type) {
        return (T) JSON.parseObject(jsonStr, type);
    }

    /**
     * Get the index of truck or tray from a message file.
     */
    private static int getIndex(String fileStr) {
        // Match and return the content of the variables.
        String pattern;
        pattern = "(\\d+)_";
        Pattern r = Pattern.compile(pattern);
        Matcher m = r.matcher(fileStr);
        if (m.find()) {
            return Integer.parseInt(m.group(1));
        }
        else {
            throw new IllegalArgumentException("Do not find the index of the html file.");
        }
    }
}
