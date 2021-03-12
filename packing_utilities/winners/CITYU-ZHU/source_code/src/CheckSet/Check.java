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

package CheckSet;

import CheckSet.entity.*;
import CheckSet.util.Config;
import CheckSet.util.DataImporting;
import CheckSet.util.OutputJson2Html;

import java.io.File;
import java.io.IOException;
import java.nio.file.NoSuchFileException;
import java.nio.file.Paths;
import java.util.*;

import static java.lang.Math.*;


public class Check {
    private Map<Integer, ArrayList<String>> errorMessages = new HashMap<>();
    private ArrayList<Solution> solutions;
    private ArrayList<Box> inputBoxes;
    private Route route;

    /**
     * Generate boxes, bins and trays from the directory of html files or a single html file.
     * @param outputPath: the directory of output files.
     * @param messageFile: the path of an input message file.
     */
    public Check(String outputPath, String messageFile) throws IOException {
        this(outputPath, messageFile, "json", null);
    }

    public Check(String outputPath, String messageFile, String outputMode) throws IOException {
        this(outputPath, messageFile, outputMode, null);
    }

    /**
     * Generate boxes, bins and trays from an output json file or html files.
     * @param outputPath: the directory of output files.
     * @param messageFile: the path of an input message file.
     * @param outputMode: Choose the recognition mode: "html" or "json".
     */
    public Check(String outputPath, String messageFile, String outputMode, String visualizationPath) throws IOException {
        // Import the data.
        File inputFile = new File(outputPath);
        DataImporting dataImporting = new DataImporting(messageFile);
        // If import data from an output json file, transfer it to html files first.
        if (outputMode.equals("json")) {
            if (visualizationPath == null || visualizationPath.equals("")) {
                visualizationPath = Paths.get("result", "visualization").toString();
            }
            File visualizationDir = new File(visualizationPath);
            if (!visualizationDir.exists()) {
                visualizationDir.mkdir();
            }
            String outputName = Paths.get(outputPath).getFileName().toString();
            String visualizationFilesPath = Paths.get(visualizationPath, outputName).toString();
            OutputJson2Html.Json2Html(
                    messageFile,
                    outputPath,
                    Config.TRUCK_TEMPLATE_PATH,
                    visualizationFilesPath);
            outputPath = visualizationFilesPath;
            inputFile = new File(visualizationFilesPath);
        }
        // check all the html files for a solution.
        if (inputFile.isDirectory()) {
            this.solutions = dataImporting.genAllSolutions(outputPath);
            this.inputBoxes = dataImporting.genInputBoxes();
            this.route = dataImporting.genRoute();
        }
        else {
            throw new IllegalArgumentException("Please input a correct path of a directory");
        }
    }

    /**
     * Main programme for checks.
     * @return true: passed; false: unpassed.
     */
    public boolean check() {
        boolean pass = true;
        for (Solution solution: this.solutions) {
            pass = checkSingleSolution(solution);
        }
        return pass;
    }

    /**
     * Check one solution.
     * @return true: passed; false: unpassed.
     */
    public boolean checkSingleSolution(Solution solution) {
        boolean pass = true;
        if (!checkInput(solution)) {
            pass = false;
        }
        if (!solution.getAllBins().isEmpty() && !checkTruck(solution)) {
            pass = false;
        }
        return pass;
    }

    /**
     * Check if the boxes are not packed.
     * @return true: passed; false: unpassed.
     */
    @SuppressWarnings("unchecked")
    private boolean checkInput(Solution solution) {
        ArrayList<String> solutionErrorMessages = this.errorMessages.getOrDefault(solution.getIndex(), new ArrayList<>());
        Map<Integer, Map<Integer, BoxInTruck>> allBoxesInTruck = solution.getAllBoxesInTruck();
        boolean pass = true;
        ArrayList<String> resultIds = new ArrayList<>();
        ArrayList<String> inputIds = new ArrayList<>();
        for (Map<Integer, BoxInTruck> boxes: allBoxesInTruck.values()) {
            for (BoxInTruck boxInTruck : boxes.values()) {
                resultIds.add(boxInTruck.getId());
            }
        }
        for (Box inputBox: this.inputBoxes) {
            inputIds.add(inputBox.getId());
        }
        // Compare two lists.
        boolean equal;
        if (resultIds.size() != inputIds.size()){
            equal = false;
        }
        else {
            Collections.sort(resultIds);
            Collections.sort(inputIds);
            equal = resultIds.equals(inputIds);
        }
        if (!equal) {
            solutionErrorMessages.add("The result boxes are not consistent with the input boxes.");
            solutionErrorMessages.add("Input boxes: size " + inputIds.size());
            ArrayList<String> inputIdsCopy = (ArrayList<String>) inputIds.clone();
            resultIds.forEach(inputIdsCopy::remove);
            solutionErrorMessages.add("Unpacked boxes: ");
            solutionErrorMessages.add(inputIdsCopy.toString());
            solutionErrorMessages.add("Result boxes: size " + resultIds.size());
            solutionErrorMessages.add("Extra packed boxes: ");
            ArrayList<String> resultIdsCopy = (ArrayList<String>) resultIds.clone();
            inputIds.forEach(resultIdsCopy::remove);
            solutionErrorMessages.add(resultIdsCopy.toString());
            pass = false;
        }
        this.errorMessages.put(solution.getIndex(), solutionErrorMessages);
        return pass;
    }

    /**
     * Check all the trucks.
     * @return true: passed; false: unpassed.
     */
    private boolean checkTruck(Solution solution) {
        ArrayList<String> solutionErrorMessages = this.errorMessages.getOrDefault(solution.getIndex(), new ArrayList<>());
        boolean pass = true;
        for (int truckIndex = 1; truckIndex <= solution.getAllBoxesInTruck().size(); truckIndex++){
            Map<Integer, BoxInTruck> boxes = solution.getAllBoxesInTruck().get(truckIndex);
            Bin bin = solution.getAllBins().get(truckIndex);
            ArrayList<String> platformOrder = new ArrayList<>();
            double totalWeight = 0.;
            for (int i = 1; i <= boxes.size(); i++) {
                BoxInTruck boxInTruck = boxes.get(i);
                // Check the order of the platform the boxInTruck belonging to.
                if (!checkPlatformOrder(boxInTruck, platformOrder)) {
                    solutionErrorMessages.add("Unpassed: The platform of the box " + i + " in truck " + truckIndex
                            + " is duplicated with the previous.");
                    pass = false;
                }
            }
            for (int i = 1; i <= boxes.size(); i++) {
                BoxInTruck boxInTruck = boxes.get(i);
                // Check the boundary of the bin.
                if (!checkBoundary(boxInTruck, bin)) {
                    solutionErrorMessages.add("Unpassed: The box " + i + " in truck " + truckIndex
                            + " is beyond the boundary of the bin.");
                    pass = false;
                }
                // Check if the platform order of the front box is behind of the platform orders of the boxes behind of it.
                // AKA: check if the box is needed to be pulled out to pack the latter boxes.
                ArrayList<Box> behindBoxesInTruck = getBehindBoxes(boxInTruck, boxes);
                if (!checkPullOut(boxInTruck, behindBoxesInTruck, platformOrder)) {
                    solutionErrorMessages.add(
                            "Unpassed: the box " + i + " is needed to be pulled out to pack the latter boxes in truck"
                                    + truckIndex + ".");
                    pass = false;
                }
                // Check if the total weight of the boxes is beyond the max loading weight of the truck.
                totalWeight += boxInTruck.getWeight();
                if (totalWeight > bin.getMaxLoadWeight()) {
                    solutionErrorMessages.add(
                            "Unpassed: the total weight of the boxes is beyond the max loading weight of the truck "
                                    + truckIndex + ".");
                    pass = false;
                }
                for (int j = 1; j < i; j++) {
                    BoxInTruck boxInTruck2 = boxes.get(j);
                    // Check the overlap of the two boxes.
                    if (!checkOverlap(boxInTruck2, boxInTruck)) {
                        solutionErrorMessages.add(
                                "Unpassed: The box " + i + " and the box " + j + " in the truck " + truckIndex
                                        + " are overlapped.");
                        pass = false;
                    }
                }
                ArrayList<Box> belowBoxesInTruck = getBelowBoxes(boxInTruck, boxes);
                // Check the support area ratio.
                if (!checkSupportAreaRatio(boxInTruck, belowBoxesInTruck)) {
                    solutionErrorMessages.add(
                            "Unpassed: the support area below the box " + i + " in the truck " + truckIndex
                                    + " is smaller than " + String.format("%.0f", Config.SUPPORT_RATIO * 100) + "%.");
                    pass = false;
                }
            }
            // Check if the bonded warehouse is the first platform.
            if (!checkFirstPlatform(platformOrder)) {
                solutionErrorMessages.add(
                        "Unpassed: the truck " + truckIndex + " does not visit some bonded warehouses first.");
                pass = false;
            }
        }
        this.errorMessages.put(solution.getIndex(), solutionErrorMessages);
        return pass;
    }

    /**
     * Check if the boxInTruck is beyond the boundary of the bin.
     * @return true: passed; false: unpassed.
     */
    private boolean checkBoundary(BoxInTruck boxInTruck, Bin bin) {
        return (boxInTruck.getX1() >= -Config.BOUNDARY_ERROR
                && boxInTruck.getY1() >= -Config.BOUNDARY_ERROR
                && boxInTruck.getZ1() >= -Config.BOUNDARY_ERROR
                && boxInTruck.getX2() <= bin.getLength() + Config.BOUNDARY_ERROR
                && boxInTruck.getY2() <= bin.getWidth() + Config.BOUNDARY_ERROR
                && boxInTruck.getZ2() <= bin.getHeight() + Config.BOUNDARY_ERROR);
    }

    /**
     * Check if the two boxes are overlapped.
     * @return true: passed; false: unpassed.
     */
    private boolean checkOverlap(Box box1, Box box2) {
        return (box2.getX2() <= box1.getX1() + Config.OVERLAP_ERROR
                || box1.getX2() <= box2.getX1() + Config.OVERLAP_ERROR
                || box2.getY2() <= box1.getY1() + Config.OVERLAP_ERROR
                || box1.getY2() <= box2.getY1() + Config.OVERLAP_ERROR
                || box2.getZ2() <= box1.getZ1() + Config.OVERLAP_ERROR
                || box1.getZ2() <= box2.getZ1() + Config.OVERLAP_ERROR);
    }

    /**
     * Check the platform order.
     * @return true: passed; false: unpassed.
     */
    private boolean checkPlatformOrder(BoxInTruck boxInTruck, ArrayList<String> platformOrder) {
        String platform = boxInTruck.getPlatformCode();
        if (!platformOrder.contains(platform)) {
            platformOrder.add(platform);
            return true;
        }
        else return (platform.equals(platformOrder.get(platformOrder.size() - 1)));
    }

    /**
     * Check if the box is needed to be pulled out to pack the latter boxes.
     * @param box: current box being checked.
     * @param behindBoxes: the boxes behind of the current boxes.
     * @return true: passed; false: unpassed.
     */
    private boolean checkPullOut(Box box, ArrayList<Box> behindBoxes, ArrayList<String> route) {
        int current_platform_ind = route.indexOf(box.getPlatformCode());
        for (Box behindBox: behindBoxes) {
            int platform_ind = route.indexOf(behindBox.getPlatformCode());
            if (current_platform_ind < platform_ind) {
                return false;
            }
        }
        return true;
    }

    /**
     * Check if the support area ratio is larger than a certain percentage in a truck.
     * @param box: current box being checked.
     * @param belowBoxes: the boxes below the current boxes.
     * @return true: passed; false: unpassed.
     */
    private boolean checkSupportAreaRatio(Box box, ArrayList<Box> belowBoxes) {
        if (belowBoxes.isEmpty()) {
            return box.getZ1() < Config.CONTACT_ERROR;
        }
        double SupportArea = 0.;
        for (Box belowBox: belowBoxes) {
            // Calculate the total support area.
            SupportArea += (min(belowBox.getX2() - box.getX1(), box.getX2() - belowBox.getX1())
                    * min(belowBox.getY2() - box.getY1(), box.getY2() - belowBox.getY1()));
        }
        // Check the support area ratio.
        return belowBoxes.isEmpty() || SupportArea / (box.getL() * box.getW()) >= Config.SUPPORT_RATIO - 1e-4;
    }

    /**
     * Get the the boxes below a given box.
     */
    private ArrayList<Box> getBelowBoxes(Box boxInTruck, Map<Integer, BoxInTruck> boxes) {
        ArrayList<Box> belowBoxesInTruck = new ArrayList<>();
        for (Box boxInTruck2 : boxes.values()) {
            if (isContact(boxInTruck, boxInTruck2)) {
                belowBoxesInTruck.add(boxInTruck2);
            }
        }
        return belowBoxesInTruck;
    }

    /**
     * Check if two boxes are in contact.
     */
    private boolean isContact(Box upperBox, Box lowerBox) {
        return (abs(upperBox.getZ1() - lowerBox.getZ2()) < Config.CONTACT_ERROR
                && lowerBox.getX2() > upperBox.getX1()
                && upperBox.getX2() > lowerBox.getX1()
                && lowerBox.getY2() > upperBox.getY1()
                && upperBox.getY2() > lowerBox.getY1());
    }

    /**
     * Get the boxes behind a given box.
     */
    private ArrayList<Box> getBehindBoxes(Box boxInTruck, Map<Integer, BoxInTruck> boxes) {
        ArrayList<Box> behindBoxes = new ArrayList<>();
        for (Box boxInTruck2 : boxes.values()) {
            if (isBehind(boxInTruck, boxInTruck2)) {
                behindBoxes.add(boxInTruck2);
            }
        }
        return behindBoxes;
    }

    /**
     * Check if two boxes are in contact.
     */
    private boolean isBehind(Box frontBox, Box rearBox) {
        return (frontBox.getX1() > rearBox.getX2() - Config.CONTACT_ERROR
                && rearBox.getZ2() > frontBox.getZ1() + Config.OVERLAP_ERROR
                && frontBox.getZ2() > rearBox.getZ1() + Config.OVERLAP_ERROR
                && rearBox.getY2() > frontBox.getY1() + Config.OVERLAP_ERROR
                && frontBox.getY2() > rearBox.getY1() + Config.OVERLAP_ERROR);
    }

    /**
     * Check if the bonded warehouse is the first platform.
     */
    private boolean checkFirstPlatform(ArrayList<String> platformOrder) {
        for (int i = 1; i < platformOrder.size(); i++) {
            Map platformMap = (Map) this.route.getAllPlatformMap().get(platformOrder.get(i));
            if ((boolean) platformMap.get("mustFirst")) {
                return false;
            }
        }
        return true;
    }

    /**
     * Get a Check object from the given addresses and order name.
     */
    public static Check getOrderCheck(String inputDir, String outputDir, String orderName) throws IOException {
        return getOrderCheck(inputDir, outputDir, orderName, "json", null);
    }

    public static Check getOrderCheck(String inputDir, String outputDir, String orderName, String outputMode) throws IOException {
        return getOrderCheck(inputDir, outputDir, orderName, outputMode, null);
    }

    public static Check getOrderCheck(String inputDir, String outputDir, String orderName, String outputMode, String visualizationDir) throws IOException {
        String orderNameOri;
        if (orderName.endsWith("_d")) {
            orderNameOri = orderName.replace("_d", "");
        }
        else {
            orderNameOri = orderName;
            orderName = orderName + "_d";
        }
        String orderInput = Paths.get(inputDir, orderName).toString();
        String orderOutput = Paths.get(outputDir,orderNameOri).toString();
        Check orderCheck;
        try {
            try {
                orderCheck = new Check(orderOutput, orderInput, outputMode, visualizationDir);
            }
            catch (NoSuchFileException e) {
                orderInput = Paths.get(inputDir, orderNameOri).toString();
                orderCheck = new Check(orderOutput, orderInput, outputMode, visualizationDir);
            }
        }
        catch (IllegalArgumentException e) {
            try {
                orderOutput = Paths.get(outputDir, orderNameOri + "_runSmartPackingSite").toString();
                orderCheck = new Check(orderOutput, orderInput, outputMode, visualizationDir);
            }
            catch (IllegalArgumentException e1) {
                orderOutput = Paths.get(outputDir, orderNameOri + "_runPackingForWeightBalance").toString();
                orderCheck = new Check(orderOutput, orderInput, outputMode, visualizationDir);
            }
        }
        return orderCheck;
    }

    public Map<Integer, ArrayList<String>> getErrorMessages() {
        return this.errorMessages;
    }

    public ArrayList<Box> getInputBoxes() {
        return this.inputBoxes;
    }

    public Route getRoute() {
        return this.route;
    }
}
