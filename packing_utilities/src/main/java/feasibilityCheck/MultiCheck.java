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

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;
import java.util.Objects;


public class MultiCheck {
    private String inputDir;
    private String outputDir;
    private File orders;
    private FileWriter resultWriter;
    private String visualizationDir;

    public MultiCheck(String inputDir, String outputDir, String resultFile, String visualizationDir) throws IOException {
        this.inputDir = inputDir;
        this.outputDir = outputDir;
        this.orders = new File(outputDir);
        if (!this.orders.isDirectory()) {
            throw new IOException("inputDir and outputDir should be directories.");
        }
        // Write the results to file.
        if (resultFile == null || resultFile.equals("")) {
            this.resultWriter = null;
        }
        else {
            this.resultWriter = new FileWriter(resultFile, false);
        }
        // Generate visualization files.
        this.visualizationDir = visualizationDir;
    }

    public MultiCheck(String inputDir, String outputDir, String resultFile) throws IOException {
        this(inputDir, outputDir, resultFile, null);
    }

    public MultiCheck(String inputDir, String outputDir) throws IOException {
        this(inputDir, outputDir, null, null);
    }

    /**
     * Check multiple results.
     */
    public void check(String outputMode) throws IOException {
        long startTime = System.currentTimeMillis();
        int orderNum = 1;
        int uncheckedOrderNum = 0;
        for (File order: Objects.requireNonNull(this.orders.listFiles())) {
            System.out.println("Checking order no. " + orderNum);
            String orderName = order.getName();
            String orderNameOri = orderName.replace("_d", "");
            Check orderCheck = Check.getOrderCheck(this.inputDir, this.outputDir, orderName, outputMode, this.visualizationDir);
            try {
                orderCheck.check();
                Map<Integer, ArrayList<String>> errorMessages = orderCheck.getErrorMessages();
                int errorNum = 0;
                for (ArrayList<String> solutionErrors: errorMessages.values()) {
                    errorNum += solutionErrors.size();
                }
                if (errorNum > 0) {
                    print(orderNameOri);
                    for (int i = 1; i <= errorMessages.size(); i++) {
                        ArrayList<String> solutionErrors = errorMessages.get(i);
                        if (!solutionErrors.isEmpty()) {
                            print("Solution " + i + " errors:");
                            for (String errorMessage: solutionErrors) {
                                print(errorMessage);
                            }
                            print("");
                        }
                    }
                    print("");
                }
                orderNum += 1;
            }
            catch (StackOverflowError stackOverflowError) {
                uncheckedOrderNum += 1;
                print(orderNameOri);
                print("Stack overflow.\n");
            }
        }
        long runTime = (System.currentTimeMillis() - startTime) / 1000;
        print("Number of orders checked: " + (orderNum-1));
        print("Number of orders unchecked: " + uncheckedOrderNum);
        print("\nRuntime: " + runTime + "s");
        if (this.resultWriter != null) {
            this.resultWriter.close();
        }
    }

    public void check() throws IOException {
        check("json");
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
        String inputDir = args[0];
        String outputDir = args[1];
        String resultFile = ".\\result\\checkResult.txt";
        if (args.length >= 3) {
            resultFile = args[2];
        }
        String visualizationDir = null;
        if (args.length >= 4) {
            visualizationDir = args[3];
        }
        MultiCheck multiCheck = new MultiCheck(inputDir, outputDir, resultFile, visualizationDir);
        multiCheck.check("json");
    }
}
