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

package feasibilityCheck;

import feasibilityCheck.entity.*;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;


public class Statistics {
    private String inputDir;
    private String outputDirOld;
    private String outputDirNew;
    private File orders;
    private FileWriter resultWriter;

    public Statistics(String inputDir, String outputDirOld, String outputDirNew, String resultFile) throws IOException {
        this.inputDir = inputDir;
        this.outputDirOld = outputDirOld;
        this.outputDirNew = outputDirNew;
        this.orders = new File(inputDir);
        if (!this.orders.isDirectory()) {
            throw new IOException("inputDir and outputDir should be directories.");
        }
        // Write the results to file.
        if (resultFile == null || resultFile.equals("")) {
            resultWriter = null;
        }
        else {
            resultWriter = new FileWriter(resultFile, false);
        }
    }

    public Statistics(String inputDir, String outputDirOld, String outputDirNew) throws IOException {
        this(inputDir, outputDirOld, outputDirNew, null);
    }

    public void calc() throws IOException {
        // Calculate the volumes of boxes and bins by groups of the numbers of bins
        Map<Integer, Double> boxesVolumeStat = new HashMap<>();
        Map<Integer, Double> binsVolumeStatOld = new HashMap<>();
        Map<Integer, Double> binsVolumeStatNew = new HashMap<>();
        Map<Integer, Integer> orderNumStat = new HashMap<>();
        for (File order: Objects.requireNonNull(this.orders.listFiles())) {
            String orderName = order.getName();
            String orderNameOri = orderName.replace("_d", "");
            Check orderCheckOld;
            try {
                orderCheckOld = Check.getOrderCheck(this.inputDir, this.outputDirOld, orderName);
            }
            catch (IllegalArgumentException e) {
                print("Order missed in old results: " + orderNameOri);
                continue;
            }
            Check orderCheckNew;
            try {
                orderCheckNew = Check.getOrderCheck(this.inputDir, this.outputDirNew, orderName);
            }
            catch (IllegalArgumentException e) {
                print("Order missed in new results: " + orderNameOri);
                continue;
            }
            Map<Integer, Bin> allBinsOld = orderCheckOld.getAllBins();
            Map<Integer, Bin> allBinsNew = orderCheckNew.getAllBins();
            ArrayList<Box> inputBoxes = orderCheckOld.getInputBoxes();
            double boxesVolume = 0.;
            for (Box box: inputBoxes) {
                boxesVolume += box.getLength() * box.getWidth() * box.getHeight();
            }
            double binsVolume1 = 0.;
            for (Bin bin: allBinsOld.values()) {
                binsVolume1 += bin.getLength() * bin.getWidth() * bin.getHeight() * 1000;
            }
            double binsVolume2 = 0.;
            for (Bin bin: allBinsNew.values()) {
                binsVolume2 += bin.getLength() * bin.getWidth() * bin.getHeight() * 1000;
            }
            int binNum = allBinsOld.size();
            boxesVolumeStat.put(binNum, boxesVolumeStat.getOrDefault(binNum, 0.) + boxesVolume);
            binsVolumeStatOld.put(binNum, binsVolumeStatOld.getOrDefault(binNum, 0.) + binsVolume1);
            binsVolumeStatNew.put(binNum, binsVolumeStatNew.getOrDefault(binNum, 0.) + binsVolume2);
            orderNumStat.put(binNum, orderNumStat.getOrDefault(binNum, 0) + 1);
        }
        double totalBoxesVolume = 0.;
        double totalBinsVolumeOld = 0.;
        double totalBinsVolumeNew = 0.;
        Integer totalOrderNum = 0;
        print("truck_num order_num old_rate new_rate diff");
        for (Integer binNum: boxesVolumeStat.keySet()) {
            totalBoxesVolume += boxesVolumeStat.get(binNum);
            totalBinsVolumeOld += binsVolumeStatOld.get(binNum);
            totalBinsVolumeNew += binsVolumeStatNew.get(binNum);
            totalOrderNum += orderNumStat.get(binNum);
            double packingRateOld = boxesVolumeStat.get(binNum) / binsVolumeStatOld.get(binNum);
            double packingRateNew = boxesVolumeStat.get(binNum) / binsVolumeStatNew.get(binNum);
            String statMessage = String.join(
                    " ",
                    binNum.toString(),
                    orderNumStat.get(binNum).toString(),
                    String.format("%.4f", packingRateOld),
                    String.format("%.4f", packingRateNew),
                    String.format("%.4f", packingRateNew - packingRateOld)
            );
            print(statMessage);
        }
        double totalPackingRateOld = totalBoxesVolume / totalBinsVolumeOld;
        double totalPackingRateNew = totalBoxesVolume / totalBinsVolumeNew;
        String totalStatMessage = String.join(
                " ",
                "overall",
                totalOrderNum.toString(),
                String.format("%.4f", totalPackingRateOld),
                String.format("%.4f", totalPackingRateNew),
                String.format("%.4f", totalPackingRateNew - totalPackingRateOld)
        );
        print(totalStatMessage);
        if (this.resultWriter != null) {
            this.resultWriter.close();
        }
    }

    /**
     * Print to the console or the file.
     */
    private void print(String message) throws IOException {
        if (this.resultWriter == null) {
            System.out.println(message);
        }
        else {
            this.resultWriter.write(message + '\n');
        }
    }

    public static void main(String[] args) throws IOException {
        String inputDir = ".\\data\\data0923\\input";
        String outputDirOld = ".\\data\\result1";
        String outputDirNew = ".\\data\\result3";
        String resultFile = ".\\result\\statResult3.txt";
        Statistics statistics = new Statistics(inputDir, outputDirOld, outputDirNew, resultFile);
        statistics.calc();
    }
}
